"""
模型可解释性模块 — SHAP 分析
==============================
对最佳树模型（LightGBM / XGBoost / RandomForest）生成 SHAP 可解释性图表。
若 shap 未安装则自动提示，不影响主程序运行。
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import BASE_DIR, FIGURES_DIR, MODEL_NAME_ZH

# ==================== 加载字段中英文映射 ====================
_COLUMN_ZH = {}
_doc_path = os.path.join(BASE_DIR, 'OULAD_字段说明.csv')
if os.path.exists(_doc_path):
    _doc_df = pd.read_csv(_doc_path)
    for _, _r in _doc_df.iterrows():
        _COLUMN_ZH[str(_r.iloc[0]).strip()] = str(_r.iloc[1]).strip()

# 类别取值映射（OULAD 常见取值 → 中文）
_CAT_VALUE_ZH = {
    'M': '男', 'F': '女',
    '0': '否', '1': '是', 'Y': '是', 'N': '否',
    'Unknown': '未知',
}


def _translate_feature_name(raw_name):
    """
    将预处理后的特征名翻译为中文：
      - 数值列: total_clicks_120 → 前120天总点击量
      - 类别展开列: gender_F → 性别=女
    """
    # 先尝试整字段匹配
    if raw_name in _COLUMN_ZH:
        return _COLUMN_ZH[raw_name]

    # 尝试 OneHot 展开列: 取最后一个 _ 前的部分为基础列名，后面的为取值
    if '_' in raw_name:
        base = raw_name.rsplit('_', 1)[0]
        val = raw_name.rsplit('_', 1)[1]
        if base in _COLUMN_ZH:
            zh_val = _CAT_VALUE_ZH.get(val, val)
            return f'{_COLUMN_ZH[base]}={zh_val}'

    # 兜底：去除 _120 后缀后返回
    return raw_name.replace('_120', '')

# ==================== 中文字体 ====================
_ZH_FONTS = ['Microsoft YaHei', 'SimHei', 'DengXian', 'Noto Sans SC', 'PingFang SC']
for _f in _ZH_FONTS:
    try:
        plt.rcParams['font.sans-serif'] = [_f]
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue


# ==================== 提取预处理后的特征名 ====================
def _get_feature_names(preprocessor, raw_feature_cols, cat_cols, num_cols):
    """
    从 ColumnTransformer 中提取经过 OneHotEncoder 展开后的完整特征名。
    数值特征保持原列名，类别特征追加 OneHot 后的各取值列名。
    """
    names = []
    for name, trans, columns in preprocessor.transformers_:
        if trans == 'drop' or trans is None:
            continue
        if name == 'cat':
            ohe = trans.named_steps['ohe']
            names.extend(ohe.get_feature_names_out(columns))
        elif name == 'num':
            names.extend(columns)
    # 全部翻译为中文
    return [_translate_feature_name(n) for n in names]


def _try_shap():
    """尝试导入 shap，失败返回 None"""
    try:
        import shap
        return shap
    except ImportError:
        return None


# ==================== SHAP 分析主入口 ====================
def run_shap_analysis(models_results, X_test, cat_cols, num_cols, best_model_name=None):
    """
    对最佳树模型运行 SHAP 分析，生成 4 张图表：
      1. shap_summary.png    — SHAP 蜜蜂图
      2. shap_bar.png        — 平均绝对 SHAP 值排序
      3. shap_waterfall.png  — 单样本瀑布图
      4. shap_dependence.png — 最重要特征的依赖图
    若 shap 未安装则提示 pip install shap 并跳过。
    """
    shap = _try_shap()
    if shap is None:
        print('\n  [!] SHAP 未安装，跳过可解释性分析')
        print('     安装命令: pip install shap')
        return

    # 自动选择最佳树模型（优先 LightGBM > XGBoost > RandomForest）
    if best_model_name is None:
        for pref in ['LightGBM', 'XGBoost', 'RandomForest']:
            if pref in models_results and hasattr(models_results[pref]['model'], 'feature_importances_'):
                best_model_name = pref
                break
        if best_model_name is None:
            print('\n  无可解释的树模型，跳过 SHAP')
            return

    res = models_results.get(best_model_name)
    if res is None:
        print(f'\n  模型 {best_model_name} 不存在，跳过 SHAP')
        return

    model = res['model']
    if not hasattr(model, 'feature_importances_'):
        print(f'\n  {best_model_name} 不是树模型，跳过 SHAP')
        return

    pipeline = res['pipeline']
    preprocessor = pipeline.named_steps['prep']

    # 关键优化：抽样 200 条用于 SHAP 分析（全量 6000+ 条计算极慢）
    n_samples = min(200, len(X_test))
    X_test_sample = X_test.sample(n=n_samples, random_state=42)

    # 对测试集应用与训练时相同的预处理（类别编码、缺失值填充等）
    try:
        feature_names = _get_feature_names(preprocessor, [], cat_cols, num_cols)
        X_test_transformed = preprocessor.transform(X_test_sample)
    except Exception as e:
        print(f'\n  [!]  预处理转换失败: {e}')
        return

    display_name = MODEL_NAME_ZH.get(best_model_name, best_model_name)
    print(f'\n  生成 SHAP 可解释性分析 ({display_name})...')

    # shap 要求输入为稠密 numpy 数组
    if hasattr(X_test_transformed, 'toarray'):
        X_test_transformed = X_test_transformed.toarray()
    if not isinstance(X_test_transformed, np.ndarray):
        X_test_transformed = np.asarray(X_test_transformed)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_transformed)

        # 二分类兼容处理：shap 输出可能是 list(2) 或 3D 数组
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_pos = shap_values[1]
        elif hasattr(shap_values, 'shape') and shap_values.ndim == 3:
            shap_values_pos = shap_values[:, :, 1]
        else:
            shap_values_pos = shap_values

        # =============================================
        # 1. SHAP Summary Plot（蜜蜂图）
        # 展示每个特征对预测的影响方向与强度
        # =============================================
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_pos, X_test_transformed,
            feature_names=feature_names,
            show=False,
            plot_size=(10, 8),
            color_bar_label='特征值',
        )
        plt.title(f'SHAP 模型解释摘要 - {display_name}', fontsize=14, fontweight='bold')
        path = os.path.join(FIGURES_DIR, 'shap_summary.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'    SHAP 蜜蜂图 -> {path}')

        # =============================================
        # 2. SHAP Bar Plot（平均绝对 SHAP 值排序）
        # 相当于特征重要性的 SHAP 版本
        # =============================================
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_pos, X_test_transformed,
            feature_names=feature_names,
            plot_type='bar',
            show=False,
            plot_size=(10, 8),
            color='#4C72B0',
        )
        plt.title(f'SHAP 特征重要性排序 - {display_name}', fontsize=14, fontweight='bold')
        path = os.path.join(FIGURES_DIR, 'shap_bar.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'    SHAP 条形图 -> {path}')

        # =============================================
        # 3. SHAP Waterfall Plot（单样本瀑布图）
        # 展示各特征如何共同影响某一条样本的最终预测
        # =============================================
        sample_idx = np.random.randint(0, n_samples)
        expected_value = explainer.expected_value
        if isinstance(expected_value, np.ndarray) and expected_value.ndim == 1 and len(expected_value) == 2:
            ev = expected_value[1]
        elif isinstance(expected_value, np.ndarray) and expected_value.ndim == 0:
            ev = float(expected_value)
        else:
            ev = expected_value

        shap_values_single = shap_values_pos[sample_idx]
        data_single = X_test_transformed[sample_idx]
        if shap_values_single.ndim > 1:
            shap_values_single = shap_values_single.flatten()
        if data_single.ndim > 1:
            data_single = data_single.flatten()

        fig = plt.figure(figsize=(10, 7))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_single,
                base_values=ev,
                data=data_single,
                feature_names=feature_names,
            ),
            show=False,
            max_display=15,
        )
        plt.title(f'SHAP 单样本决策路径 - {display_name}', fontsize=14, fontweight='bold', y=1.02)
        path = os.path.join(FIGURES_DIR, 'shap_waterfall.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'    SHAP 瀑布图 -> {path}')

        # =============================================
        # 4. SHAP Dependence Plot（依赖图）
        # 展示最重要特征的值变化如何影响 SHAP 值
        # =============================================
        mean_abs_shap = np.mean(np.abs(shap_values_pos), axis=0)
        top2_idx = np.argsort(mean_abs_shap)[-2:][::-1]
        for rank, idx in enumerate(top2_idx):
            feat_name = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
            fig = plt.figure(figsize=(8, 6))
            shap.dependence_plot(
                idx, shap_values_pos, X_test_transformed,
                feature_names=feature_names,
                show=False,
            )
            plt.title(f'SHAP 依赖关系 - {feat_name}', fontsize=14, fontweight='bold')
            path = os.path.join(FIGURES_DIR, f'shap_dependence_{rank+1}.png')
            plt.tight_layout()
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'    SHAP 依赖图 ({feat_name}) -> {path}')

        # 保存 Top1 特征依赖图为 shap_dependence.png（用户指定文件名）
        idx = top2_idx[0]
        feat_name = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
        fig = plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            idx, shap_values_pos, X_test_transformed,
            feature_names=feature_names,
            show=False,
        )
        plt.title(f'SHAP 依赖关系 - {feat_name}', fontsize=14, fontweight='bold')
        path = os.path.join(FIGURES_DIR, 'shap_dependence.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'    SHAP 依赖图 (top1) -> {path}')

    except Exception as e:
        print(f'  [!]  SHAP 分析失败: {e}')
        import traceback
        traceback.print_exc()
