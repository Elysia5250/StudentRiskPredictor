#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学生早期挂科预警系统 — 完整流程
===================================
流程: 加载 → 字段识别 → 特征筛选 → 交叉验证 → 训练 + 校准
      → 阈值优化 → 集成 → 评估 → SHAP
"""

import warnings
warnings.filterwarnings('ignore')

from config import OUTPUT_DIR
from preprocess import load_wide_table, classify_columns, check_missing, check_duplicates
from models import (
    init_models, prepare_data, train_and_predict,
    cross_validate_models, select_features,
    find_optimal_threshold, ensemble_predict,
)
from evaluate import run_evaluation


def main():
    print('=' * 60)
    print('  学生早期挂科预警系统')
    print('  v2.0 — 特征筛选 + 交叉验证 + 校准 + 阈值优化 + 集成')
    print('=' * 60)

    # ==================== 1. 加载宽表 ====================
    print('\n[1/6] 加载特征宽表...')
    df = load_wide_table()
    label_col, excluded, cat_cols, num_cols = classify_columns(df)
    check_missing(df, cat_cols, num_cols)
    check_duplicates(df)

    # ==================== 2. 数据划分 ====================
    print('\n[2/6] 划分训练集/测试集...')
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(
        df, label_col, cat_cols, num_cols
    )
    print(f'  训练集: {len(X_train):,}  测试集: {len(X_test):,}')

    # ==================== 3. 特征筛选 ====================
    print('\n[3/6] 特征筛选...')
    cat_cols, num_cols = select_features(
        X_train, y_train, cat_cols, num_cols, keep_ratio=0.8
    )
    feature_cols = cat_cols + num_cols
    print(f'  筛选后特征: {len(feature_cols)}')

    # ==================== 4. 交叉验证 + 模型训练 ====================
    print('\n[4/6] 初始化模型...')
    model_map = init_models()
    if not model_map:
        print('  无可用模型')
        return

    print('\n  5折交叉验证...')
    cv_results = cross_validate_models(model_map, cat_cols, num_cols, X_train, y_train)

    print(f'\n  训练 {len(model_map)} 个模型（含概率校准）...')
    models_results = train_and_predict(
        model_map, cat_cols, num_cols, X_train, y_train, X_test,
        calibrate=True, return_val=True
    )

    # ==================== 5. 阈值优化 ====================
    print('\n[5/6] 阈值优化...')
    # 用最佳模型的验证集预测来搜索最优阈值
    best_model_name = max(cv_results, key=lambda k: cv_results[k]['mean'])
    val_entry = models_results.get(best_model_name, {})
    if 'val_proba' in val_entry:
        best_threshold, threshold_search = find_optimal_threshold(
            val_entry['y_val'], val_entry['val_proba'], metric='f1'
        )
        print(f'  最优截断阈值: {best_threshold}（基于 {best_model_name} 验证集）')
    else:
        best_threshold = 0.5
        threshold_search = None
        print('  无法获取验证集预测，使用默认阈值 0.5')

    # ==================== 6. 集成 ====================
    print('\n[6/6] 集成与评估...')
    # 对 LightGBM + XGBoost 做概率平均
    ensemble_names = [n for n in ['LightGBM', 'XGBoost', 'RandomForest'] if n in model_map]
    if len(ensemble_names) >= 2:
        ensemble_result = ensemble_predict(models_results, ensemble_names[:2], X_test, cat_cols, num_cols)
        if ensemble_result is not None:
            models_results['Ensemble'] = ensemble_result
            ensemble_names_zh = '+'.join([models_results[n]['model'].__class__.__name__ for n in ensemble_names[:2]])
            print(f'  集成模型: {ensemble_names_zh} 概率平均')

    # ==================== 评估 ====================
    run_evaluation(
        models_results, X_test, y_test, feature_cols, df_full=df,
        optimal_threshold=best_threshold,
        cv_results=cv_results,
        threshold_search=threshold_search,
    )

    # ==================== SHAP ====================
    print('\n  SHAP 可解释性分析...')
    try:
        from interpretability import run_shap_analysis
        run_shap_analysis(models_results, X_test, cat_cols, num_cols)
    except Exception:
        pass

    print(f'\n  {"=" * 60}')
    print(f'  完成！输出目录: {OUTPUT_DIR}/')
    print(f'  {"=" * 60}')


if __name__ == '__main__':
    main()
