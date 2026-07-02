#!/usr/bin/env python3
"""学生早期挂科预警系统 — Web 应用"""

import os
import sys
import threading
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

# ==================== 项目路径 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_DIR)  # 确保能导入项目根目录模块

from config import DATA_DIR, OUTPUT_DIR, FIGURES_DIR, LABEL_COL, EXCLUDED_COLS, CATEGORICAL_COLS, RANDOM_STATE
from preprocess import load_wide_table, classify_columns
from models import init_models, prepare_data, train_and_predict, build_preprocessor
from evaluate import run_evaluation

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ==================== 训练状态（线程安全） ====================
_train_lock = threading.Lock()
_train_in_progress = False
_train_result = None
_train_error = None


def _get_dataset_info():
    """读取宽表基本信息，用于页面展示"""
    try:
        df = load_wide_table()
        label_col, excluded, cat_cols, num_cols = classify_columns(df)
        return {
            'filename': 'OULAD_Final_Feature_Matrix.csv',
            'rows': len(df),
            'columns': len(df.columns),
            'features': len(cat_cols) + len(num_cols),
            'feature_list': cat_cols + num_cols,
            'drop_cols': excluded,
            'label': label_col,
            'positive_ratio': f'{df[label_col].mean()*100:.1f}%',
        }
    except Exception:
        return None


def _load_training_results():
    """加载上次训练的指标"""
    path = os.path.join(OUTPUT_DIR, 'metrics.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).to_dict(orient='records')


# ==================== 页面路由 ====================
@app.route('/')
def index():
    return render_template('index.html',
                           dataset=_get_dataset_info(),
                           metrics=_load_training_results())


@app.route('/train')
def train_page():
    return render_template('train.html',
                           dataset=_get_dataset_info(),
                           metrics=_load_training_results())


@app.route('/predict')
def predict_page():
    return render_template('predict.html',
                           dataset=_get_dataset_info())


# ==================== 静态资源服务 ====================
@app.route('/outputs/<path:filename>')
def serve_output(filename):
    """提供 outputs/ 目录下的 CSV 文件"""
    return send_from_directory(OUTPUT_DIR, filename)


@app.route('/figures/<path:filename>')
def serve_figure(filename):
    """提供 outputs/figures/ 目录下的图表"""
    return send_from_directory(FIGURES_DIR, filename)


# ==================== 训练 API ====================
@app.route('/api/train/run', methods=['POST'])
def api_train_run():
    """异步启动训练"""
    global _train_in_progress
    with _train_lock:
        if _train_in_progress:
            return jsonify({'error': '训练正在进行中'}), 400
        _train_in_progress = True
        global _train_result, _train_error
        _train_result = None
        _train_error = None
        thread = threading.Thread(target=_run_training, daemon=True)
        thread.start()
    return jsonify({'status': 'started'})


@app.route('/api/train/status')
def api_train_status():
    """轮询训练进度"""
    with _train_lock:
        return jsonify({
            'in_progress': _train_in_progress,
            'result': _train_result,
            'error': _train_error,
        })


def _run_training():
    """后台训练流程：加载 → 训练 → 评估 → SHAP → 保存"""
    global _train_in_progress, _train_result, _train_error
    try:
        df = load_wide_table()
        label_col, excluded, cat_cols, num_cols = classify_columns(df)

        # 数据划分
        X_train, X_test, y_train, y_test, feature_cols = prepare_data(
            df, label_col, cat_cols, num_cols
        )

        # 初始化并训练所有可用模型
        model_map = init_models()
        if not model_map:
            raise ValueError('无可用模型')

        models_results = train_and_predict(
            model_map, cat_cols, num_cols, X_train, y_train, X_test
        )

        # 评估 + 图表
        metrics_df = run_evaluation(
            models_results, X_test, y_test, feature_cols, df_full=df
        )

        # SHAP 可解释性（可选，失败不影响主流程）
        try:
            from interpretability import run_shap_analysis
            run_shap_analysis(models_results, X_test, cat_cols, num_cols)
        except Exception:
            pass

        # 保存最佳模型（按 ROC AUC 排序）
        os.makedirs(os.path.join(OUTPUT_DIR, 'models'), exist_ok=True)
        best_idx = metrics_df['ROC_AUC'].idxmax()
        best_name = metrics_df.loc[best_idx, 'Model']
        best_pipe = models_results[best_name]['pipeline']
        joblib.dump(best_pipe, os.path.join(OUTPUT_DIR, 'models', 'best_model.pkl'))
        joblib.dump(feature_cols, os.path.join(OUTPUT_DIR, 'models', 'feature_columns.pkl'))

        # 读取 Top5 特征重要性
        imp_path = os.path.join(OUTPUT_DIR, f'feature_importance_{best_name}.csv')
        top_features = []
        if os.path.exists(imp_path):
            imp_df = pd.read_csv(imp_path)
            top_features = imp_df.head(5).to_dict(orient='records')

        with _train_lock:
            _train_result = {
                'metrics': metrics_df.to_dict(orient='records'),
                'best_model': best_name,
                'best_auc': round(float(metrics_df.loc[best_idx, 'ROC_AUC']), 4),
                'feature_importance': top_features,
                'model_count': len(model_map),
            }

    except Exception as e:
        import traceback
        with _train_lock:
            _train_error = str(e) + '\n' + traceback.format_exc()
    finally:
        with _train_lock:
            _train_in_progress = False


# ==================== 预测 API ====================
@app.route('/api/predict/upload', methods=['POST'])
def api_predict_upload():
    """上传 CSV 文件并检查列是否完整"""
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith('.csv'):
        return jsonify({'error': '仅支持 CSV 文件'}), 400

    # 保存上传文件
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(filepath):
        base, ext = os.path.splitext(filename)
        filename = f'{base}_{int(pd.Timestamp.now().timestamp())}{ext}'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='gbk')

    # 检查模型是否存在
    model_path = os.path.join(OUTPUT_DIR, 'models', 'best_model.pkl')
    if not os.path.exists(model_path):
        return jsonify({'error': '模型不存在，请先训练'}), 400

    # 检查特征列是否完整
    feature_cols = joblib.load(os.path.join(OUTPUT_DIR, 'models', 'feature_columns.pkl'))
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        return jsonify({
            'error': f'数据缺少所需特征列: {missing}',
            'required_columns': feature_cols,
            'existing_columns': list(df.columns),
        }), 400

    preview = df.head(20).to_html(classes='data-table', index=False, escape=False)
    return jsonify({
        'file_id': filename,
        'rows': len(df),
        'columns': list(df.columns),
        'preview_html': preview,
    })


@app.route('/api/predict/run', methods=['POST'])
def api_predict_run():
    """执行预测并返回结果"""
    data = request.get_json()
    file_id = data.get('file_id')
    if not file_id:
        return jsonify({'error': '缺少文件标识'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file_id))
    if not os.path.exists(filepath):
        return jsonify({'error': '文件不存在'}), 400

    pipeline = joblib.load(os.path.join(OUTPUT_DIR, 'models', 'best_model.pkl'))
    feature_cols = joblib.load(os.path.join(OUTPUT_DIR, 'models', 'feature_columns.pkl'))

    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='gbk')

    # 预处理：类别编码 + 缺失值填充
    X = df[feature_cols].copy()
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category').cat.codes
    X = X.fillna(0).astype(float)

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    result = df.copy()
    result['预测结果'] = y_pred
    result['危机概率'] = np.round(y_proba, 4)

    result_filename = f'predicted_{file_id}'
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    result.to_csv(result_path, index=False, encoding='utf-8-sig')

    preview = result.head(20).to_html(classes='data-table', index=False, escape=False)
    n_total = len(result)
    n_pos = int(y_pred.sum())

    return jsonify({
        'preview_html': preview,
        'total': n_total,
        'positive_count': n_pos,
        'positive_rate': round(n_pos / n_total * 100, 1),
        'positive_label': '学业危机',
        'download_url': f'/api/download/{result_filename}',
    })


@app.route('/api/download/<filename>')
def api_download(filename):
    """下载预测结果 CSV"""
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        secure_filename(filename),
        as_attachment=True,
        download_name=filename,
    )


if __name__ == '__main__':
    app.run(debug=True, port=5001)
