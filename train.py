#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学生早期挂科预警系统
=====================
基于 OULAD 特征宽表，使用学期前期数据预测学生学业风险。
宽表中所有 _120 特征已做好时间截断，确保无数据泄露。

使用方式:
    python train.py

输出:
    outputs/metrics.csv                    - 各模型评估指标
    outputs/prediction.csv                 - 测试集预测结果
    outputs/feature_importance_*.csv       - 特征重要性
    outputs/figures/                       - 所有图表
"""

import warnings
warnings.filterwarnings('ignore')

from config import OUTPUT_DIR, RANDOM_STATE
from preprocess import load_wide_table, classify_columns, check_missing, check_duplicates
from models import init_models, prepare_data, train_and_predict
from evaluate import run_evaluation
from interpretability import run_shap_analysis


def main():
    print('=' * 60)
    print('  学生早期挂科预警系统')
    print('  基于 OULAD 特征宽表')
    print('=' * 60)

    # ==================== 1. 加载宽表 ====================
    print('\n[1/5] 加载特征宽表...')
    df = load_wide_table()

    # ==================== 2. 字段识别 ====================
    print('\n[2/5] 识别字段类型...')
    label_col, excluded, cat_cols, num_cols = classify_columns(df)

    # ==================== 3. 数据质量检查 ====================
    print('\n[3/5] 数据质量检查...')
    check_missing(df, cat_cols, num_cols)
    check_duplicates(df)

    # ==================== 4. 数据准备 ====================
    print('\n[4/5] 划分训练集/测试集...')
    X_train, X_test, y_train, y_test, feature_cols = prepare_data(
        df, label_col, cat_cols, num_cols
    )
    print(f'  训练集: {len(X_train):,}  测试集: {len(X_test):,}')
    print(f'  特征总数: {len(feature_cols)}')

    # ==================== 5. 训练与评估 ====================
    print('\n[5/5] 初始化模型...')
    model_map = init_models()
    if not model_map:
        print('  无可用模型，请检查依赖安装')
        return

    print(f'\n  训练 {len(model_map)} 个模型...')
    models_results = train_and_predict(
        model_map, cat_cols, num_cols, X_train, y_train, X_test
    )

    run_evaluation(models_results, X_test, y_test, feature_cols, df_full=df)

    # ==================== 6. SHAP 可解释性分析 ====================
    print('\n[6/6] SHAP 可解释性分析...')
    run_shap_analysis(models_results, X_test, cat_cols, num_cols)

    print(f'\n  {"=" * 60}')
    print(f'  完成！输出目录: {OUTPUT_DIR}/')
    print(f'  {"=" * 60}')


if __name__ == '__main__':
    main()
