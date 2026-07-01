#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import joblib
import pandas as pd
import numpy as np
from config import OUTPUT_DIR

warnings.filterwarnings('ignore')

MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')


def load_model(model_name='LogisticRegression'):
    model_path = os.path.join(MODEL_DIR, f'{model_name}.pkl')
    feat_path = os.path.join(MODEL_DIR, f'{model_name}_features.pkl')

    if not os.path.exists(model_path):
        print(f'Error: Model not found at {model_path}')
        print('Please run train.py first.')
        sys.exit(1)

    pipeline = joblib.load(model_path)
    feature_cols = joblib.load(feat_path)

    print(f'Model loaded: {model_name}')
    print(f'Features ({len(feature_cols)}): {feature_cols}')
    return pipeline, feature_cols


def load_data(csv_path):
    if not os.path.exists(csv_path):
        print(f'Error: File not found: {csv_path}')
        sys.exit(1)

    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')
    except Exception as e:
        print(f'Read failed: {e}')
        sys.exit(1)

    print(f'Loaded: {os.path.basename(csv_path)} ({len(df)} rows)')
    return df


def predict(pipeline, feature_cols, df):
    required = [c for c in feature_cols if c in df.columns]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f'Warning: Missing columns (will use 0): {missing}')

    X = pd.DataFrame(index=df.index)
    for col in feature_cols:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = 0

    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = X[col].astype('category').cat.codes

    X = X.fillna(0).astype(float)

    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)

    pos_idx = 1 if y_proba.shape[1] > 1 else 0
    pos_proba = y_proba[:, pos_idx]

    result = df.copy()
    result['prediction'] = y_pred
    result['risk_probability'] = np.round(pos_proba, 4)

    return result


def save_result(result, csv_path):
    base, ext = os.path.splitext(csv_path)
    output_path = f'{base}_predicted{ext}'
    result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f'Results saved: {output_path}')

    n_total = len(result)
    n_risk = int(result['prediction'].sum())
    print(f'\n--- Summary ---')
    print(f'Total: {n_total}')
    print(f'At risk: {n_risk} ({n_risk / n_total * 100:.1f}%)')
    print(f'Not at risk: {n_total - n_risk} ({(n_total - n_risk) / n_total * 100:.1f}%)')
    return output_path


def main():
    print('=' * 60)
    print('  Student Risk Predictor - Batch Prediction')
    print('=' * 60)

    if len(sys.argv) < 2:
        print('\nUsage:')
        print('  python predict.py <csv_path> [model_name]')
        print('\nExample:')
        print('  python predict.py data/new_students.csv LogisticRegression')
        sys.exit(0)

    csv_path = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else 'LogisticRegression'

    pipeline, feature_cols = load_model(model_name)
    df = load_data(csv_path)
    result = predict(pipeline, feature_cols, df)
    save_result(result, csv_path)


if __name__ == '__main__':
    main()
