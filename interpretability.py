"""
模型可解释性模块 — SHAP 分析
==============================
对最佳树模型生成 SHAP 可解释性图表。
仅支持 RandomForest / LightGBM / XGBoost 等树模型。
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import FIGURES_DIR, MODEL_NAME_ZH

# 中文字体
_ZH_FONTS = ['Microsoft YaHei', 'SimHei', 'DengXian', 'Noto Sans SC', 'PingFang SC']
for _f in _ZH_FONTS:
    try:
        plt.rcParams['font.sans-serif'] = [_f]
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue


def _get_feature_names(preprocessor, raw_feature_cols, cat_cols, num_cols):
    """从 ColumnTransformer 中提取预处理后的特征名"""
    names = []
    for name, trans, columns in preprocessor.transformers_:
        if trans == 'drop' or trans is None:
            continue
        if name == 'cat':
            ohe = trans.named_steps['ohe']
            names.extend(ohe.get_feature_names_out(columns))
        elif name == 'num':
            names.extend(columns)
    return names


def _try_shap():
    try:
        import shap
        return shap
    except ImportError:
        return None


def run_shap_analysis(models_results, X_test, cat_cols, num_cols, best_model_name=None):
    """
    对最佳树模型运行 SHAP 分析。
    若 shap 未安装则提示安装并跳过。
    """
    shap = _try_shap()
    if shap is None:
        print('\n  [!] SHAP 未安装，跳过可解释性分析')
        print('     安装命令: pip install shap')
        return

    if best_model_name is None:
        # 自动选择最佳树模型（优先 LightGBM > XGBoost > RandomForest）
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

    # 抽样 200 条用于 SHAP 分析（全量计算太慢）
    n_samples = min(200, len(X_test))
    X_test_sample = X_test.sample(n=n_samples, random_state=42)

    # 获取预处理后的特征名和数据
    try:
        feature_names = _get_feature_names(preprocessor, [], cat_cols, num_cols)
        X_test_transformed = preprocessor.transform(X_test_sample)
    except Exception as e:
        print(f'\n  [!]  预处理转换失败: {e}')
        return

    display_name = MODEL_NAME_ZH.get(best_model_name, best_model_name)
    print(f'\n  生成 SHAP 分析 ({display_name})...')

    # 确保数据是 dense（Explainer 要求）
    if hasattr(X_test_transformed, 'toarray'):
        X_test_transformed = X_test_transformed.toarray()
    # 确保是 numpy 或 pandas
    if not isinstance(X_test_transformed, np.ndarray):
        X_test_transformed = np.asarray(X_test_transformed)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_transformed)

        # 兼容不同 shap 输出格式
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_pos = shap_values[1]
        elif hasattr(shap_values, 'shape') and shap_values.ndim == 3:
            # shape (n_samples, n_features, n_classes) → 取正类
            shap_values_pos = shap_values[:, :, 1]
        else:
            shap_values_pos = shap_values

        # 1. SHAP Summary Plot (蜜蜂图)
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_pos, X_test_transformed,
            feature_names=feature_names,
            show=False,
            plot_size=(10, 8),
            color_bar_label='特征值',
        )
        plt.title(f'SHAP Summary - {display_name}', fontsize=14, fontweight='bold')
        path = os.path.join(FIGURES_DIR, 'shap_summary.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'    SHAP Summary -> {path}')

        # 2. SHAP Bar Plot (平均绝对 SHAP 值)
        fig = plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values_pos, X_test_transformed,
            feature_names=feature_names,
            plot_type='bar',
            show=False,
            plot_size=(10, 8),
            color='#4C72B0',
        )
        plt.title(f'SHAP Feature Importance - {display_name}', fontsize=14, fontweight='bold')
        path = os.path.join(FIGURES_DIR, 'shap_bar.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'    SHAP Bar -> {path}')

        # 3. SHAP Waterfall Plot (随机抽取一条样本)
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
        # 确保都是 1D
        if shap_values_single.ndim > 1:
            shap_values_single = shap_values_single.flatten()
        if data_single.ndim > 1:
            data_single = data_single.flatten()

        fig = plt.figure(figsize=(10, 6))
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
        path = os.path.join(FIGURES_DIR, 'shap_waterfall.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'    SHAP Waterfall -> {path}')

        # 4. SHAP Dependence Plot (前两个最重要特征)
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
            plt.title(f'SHAP Dependence - {feat_name}', fontsize=14, fontweight='bold')
            path = os.path.join(FIGURES_DIR, f'shap_dependence_{rank+1}.png')
            plt.tight_layout()
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f'    SHAP Dependence ({feat_name}) -> {path}')

        # 同时也保存一个包含两个 dependence 的整体图
        # （用户要求 shap_dependence.png，这里将 top1 保存为此文件名）
        idx = top2_idx[0]
        feat_name = feature_names[idx] if idx < len(feature_names) else f'feature_{idx}'
        fig = plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            idx, shap_values_pos, X_test_transformed,
            feature_names=feature_names,
            show=False,
        )
        plt.title(f'SHAP Dependence - {feat_name}', fontsize=14, fontweight='bold')
        path = os.path.join(FIGURES_DIR, 'shap_dependence.png')
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'    SHAP Dependence (top1) -> {path}')

    except Exception as e:
        print(f'  [!]  SHAP 分析失败: {e}')
        import traceback
        traceback.print_exc()
