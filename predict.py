#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高校学生学业风险预测 — 批量预测脚本
======================================
读取训练好的模型，对新数据进行预测。

使用方式:
  python predict.py data/new_students.csv

输出:
  data/new_students_predicted.csv（在原数据上追加预测结果和概率）
"""

import os
import sys
import warnings

import joblib
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'output', 'models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_model():
    """加载训练好的最佳模型、标签编码器和特征列名"""
    required = ['best_model.pkl', 'label_encoder.pkl', 'feature_columns.pkl']
    for f in required:
        path = os.path.join(MODEL_DIR, f)
        if not os.path.exists(path):
            print(f'错误：未找到模型文件 {f}')
            print(f'请先运行 python train.py 完成模型训练。')
            sys.exit(1)

    pipeline = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
    label_encoder = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    feature_cols = joblib.load(os.path.join(MODEL_DIR, 'feature_columns.pkl'))

    print(f'模型加载成功')
    print(f'特征列 ({len(feature_cols)}): {feature_cols}')
    return pipeline, label_encoder, feature_cols


def load_data(csv_path):
    """读取输入 CSV，自动识别编码"""
    if not os.path.exists(csv_path):
        print(f'错误：文件不存在: {csv_path}')
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')
    except Exception as e:
        print(f'读取失败: {e}')
        sys.exit(1)

    print(f'读取数据: {os.path.basename(csv_path)}  ({df.shape[0]} 条)')
    return df


def validate_columns(df, feature_cols):
    """检查输入数据是否包含所有必需的特征列"""
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f'错误：输入数据缺少以下特征列: {missing}')
        print(f'需要的特征列: {feature_cols}')
        print(f'当前列: {list(df.columns)}')
        sys.exit(1)

    # 提示未知列（不会被使用，但也不阻断）
    extra = [c for c in df.columns if c not in feature_cols]
    if extra:
        print(f'提示：以下列不在模型特征中，将被忽略: {extra}')

    print('特征列校验通过')
    return True


def predict(pipeline, label_encoder, feature_cols, df):
    """执行预测，返回带结果的 DataFrame"""
    X = df[feature_cols].copy()

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)

    # 获取正类（挂科）的概率索引
    pos_idx = 1 if len(label_encoder.classes_) > 1 else 0
    pos_proba = y_proba[:, pos_idx]

    pred_labels = label_encoder.inverse_transform(y_pred)

    result = df.copy()
    result['预测结果'] = pred_labels
    result['挂科概率'] = np.round(pos_proba, 4)

    return result


def save_result(result, csv_path, label_encoder):
    """保存预测结果到 CSV"""
    base, ext = os.path.splitext(csv_path)
    output_path = f'{base}_predicted{ext}'
    result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'预测结果已保存: {output_path}')

    # 统计摘要
    n_total = len(result)
    classes = label_encoder.classes_
    positive_label = classes[1] if len(classes) > 1 else classes[0]
    n_positive = (result['预测结果'] == positive_label).sum()

    print(f'\n--- 预测摘要 ---')
    print(f'总样本数: {n_total}')
    print(f'预测 [{positive_label}]: {n_positive} ({n_positive / n_total * 100:.1f}%)')
    other_label = classes[0] if len(classes) > 1 else None
    if other_label:
        print(f'预测 [{other_label}]: {n_total - n_positive} ({(n_total - n_positive) / n_total * 100:.1f}%)')
    return output_path


def main():
    print('=' * 60)
    print('  高校学生学业风险预测 — 批量预测')
    print('=' * 60)

    if len(sys.argv) < 2:
        print('\n使用方式:')
        print('  python predict.py <csv文件路径>')
        print('\n示例:')
        print('  python predict.py data/new_students.csv')
        sys.exit(0)

    csv_path = sys.argv[1]

    pipeline, label_encoder, feature_cols = load_model()
    df = load_data(csv_path)
    validate_columns(df, feature_cols)
    result = predict(pipeline, label_encoder, feature_cols, df)
    save_result(result, csv_path, label_encoder)


if __name__ == '__main__':
    main()
