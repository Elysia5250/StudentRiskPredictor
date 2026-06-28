#!/usr/bin/env python3
"""高校学生学业风险预测 — Web 应用"""

import os
import sys
import threading
import glob as pyglob
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
sys.path.insert(0, PROJECT_DIR)

import train
import joblib

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(BASE_DIR, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')

_train_lock = threading.Lock()
_train_in_progress = False
_train_result = None
_train_error = None


def _get_dataset_info():
    if not os.path.exists(DATA_DIR):
        return None
    csv_files = sorted(pyglob.glob(os.path.join(DATA_DIR, '*.csv')))
    if not csv_files:
        return None
    try:
        df = pd.read_csv(csv_files[0], encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_files[0], encoding='gbk')
    _, feature_cols, drop_cols = train.validate_and_resolve_columns(df)
    return {
        'filename': os.path.basename(csv_files[0]),
        'rows': len(df),
        'columns': len(df.columns),
        'features': len(feature_cols),
        'feature_list': feature_cols,
        'drop_cols': drop_cols,
        'label': train.LABEL_COLUMN,
    }


def _load_training_results():
    path = os.path.join(OUTPUT_DIR, 'model_metrics.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).to_dict(orient='records')


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


@app.route('/output/<path:filename>')
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route('/api/train/run', methods=['POST'])
def api_train_run():
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
    with _train_lock:
        return jsonify({
            'in_progress': _train_in_progress,
            'result': _train_result,
            'error': _train_error,
        })


def _run_training():
    global _train_in_progress, _train_result, _train_error
    try:
        csv_files = train.find_csv_files()
        if not csv_files:
            raise ValueError('未在 data/ 目录下找到 CSV 文件')
        csv_path = csv_files[0]

        df = pd.read_csv(csv_path, encoding='utf-8')
        label_col, feature_cols, drop_cols = train.validate_and_resolve_columns(df)

        train.task1_analyze(df, csv_path, label_col, feature_cols, drop_cols)
        results_df, pipelines, preprocessor, numeric_cols, categorical_cols, le = \
            train.task2_3_train_and_evaluate(df, label_col, feature_cols)
        imp_df = train.task4_feature_importance(preprocessor, pipelines, numeric_cols, categorical_cols)
        train.task5_charts(imp_df, results_df, df)
        train.task6_report(label_col, feature_cols, drop_cols, results_df, imp_df, df)

        os.makedirs(MODEL_DIR, exist_ok=True)
        best_idx = results_df['F1'].idxmax()
        best_name = results_df.loc[best_idx, 'Model']
        best_pipe = pipelines[best_name]
        joblib.dump(best_pipe, os.path.join(MODEL_DIR, 'best_model.pkl'))
        joblib.dump(le, os.path.join(MODEL_DIR, 'label_encoder.pkl'))
        joblib.dump(feature_cols, os.path.join(MODEL_DIR, 'feature_columns.pkl'))

        top_features = []
        if imp_df is not None:
            top_features = imp_df.head(5).to_dict(orient='records')
            for item in top_features:
                item['importance'] = round(item['importance'], 4)

        with _train_lock:
            _train_result = {
                'metrics': results_df.to_dict(orient='records'),
                'best_model': best_name,
                'best_f1': round(float(results_df.loc[best_idx, 'F1']), 4),
                'feature_importance': top_features,
            }
    except Exception as e:
        with _train_lock:
            _train_error = str(e)
    finally:
        with _train_lock:
            _train_in_progress = False


@app.route('/api/predict/upload', methods=['POST'])
def api_predict_upload():
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400

    filename = secure_filename(file.filename)
    if not filename.lower().endswith('.csv'):
        return jsonify({'error': '仅支持 CSV 文件'}), 400

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

    if not os.path.exists(os.path.join(MODEL_DIR, 'best_model.pkl')):
        return jsonify({'error': '模型不存在，请先训练'}), 400

    feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))
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
    data = request.get_json()
    file_id = data.get('file_id')
    if not file_id:
        return jsonify({'error': '缺少文件标识'}), 400

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file_id))
    if not os.path.exists(filepath):
        return jsonify({'error': f'文件不存在'}), 400

    pipeline = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))

    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='gbk')

    X = df[feature_cols].copy()
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]
    pred_labels = le.inverse_transform(y_pred)

    result = df.copy()
    result['预测结果'] = pred_labels
    result['挂科概率'] = np.round(y_proba, 4)

    result_filename = f'predicted_{file_id}'
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], result_filename)
    result.to_csv(result_path, index=False, encoding='utf-8-sig')

    preview = result.head(20).to_html(classes='data-table', index=False, escape=False)

    n_total = len(result)
    pos_label = le.classes_[1] if len(le.classes_) > 1 else le.classes_[0]
    n_pos = int((result['预测结果'] == pos_label).sum())

    return jsonify({
        'preview_html': preview,
        'total': n_total,
        'positive_count': n_pos,
        'positive_rate': round(n_pos / n_total * 100, 1),
        'positive_label': str(pos_label),
        'download_url': f'/api/download/{result_filename}',
    })


@app.route('/api/download/<filename>')
def api_download(filename):
    return send_from_directory(
        app.config['UPLOAD_FOLDER'],
        secure_filename(filename),
        as_attachment=True,
        download_name=filename,
    )


if __name__ == '__main__':
    app.run(debug=True, port=5000)
