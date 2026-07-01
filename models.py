import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE, MODEL_DIR, AVAILABLE_MODELS


# ==================== 模型工厂 ====================
def get_model(model_name):
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    return models.get(model_name)


def _try_import_lgb():
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(
            n_estimators=100, random_state=RANDOM_STATE, verbose=-1, n_jobs=-1
        )
    except ImportError:
        return None


def _try_import_xgb():
    try:
        import xgboost as xgb
        return xgb.XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE, use_label_encoder=False,
            eval_metric='logloss', n_jobs=-1
        )
    except ImportError:
        return None


def init_models():
    model_map = {}
    for name in AVAILABLE_MODELS:
        if name == 'LightGBM':
            m = _try_import_lgb()
        elif name == 'XGBoost':
            m = _try_import_xgb()
        else:
            m = get_model(name)

        if m is not None:
            model_map[name] = m
        else:
            print(f'  [{name}] 未安装，已跳过')
    return model_map


# ==================== 预处理流水线 ====================
def build_preprocessor(cat_cols, num_cols, scale_numeric=False):
    """
    构建 ColumnTransformer:
      - 类别: 缺失填 Unknown → OneHotEncoder
      - 数值: 中位数填充 → (可选) StandardScaler
    """
    transformers = []
    if cat_cols:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        transformers.append(('cat', cat_pipe, cat_cols))
    if num_cols:
        steps = [('imputer', SimpleImputer(strategy='median'))]
        if scale_numeric:
            steps.append(('scaler', StandardScaler()))
        transformers.append(('num', Pipeline(steps), num_cols))
    return ColumnTransformer(transformers, remainder='drop')


# ==================== 数据准备与划分 ====================
def prepare_data(df, label_col, cat_cols, num_cols):
    """分离 X / y，分层抽样划分训练集和测试集"""
    if label_col not in df.columns:
        raise ValueError(f'标签列 {label_col} 不在数据中')

    y = df[label_col].copy()
    feature_cols = cat_cols + num_cols
    X = df[feature_cols].copy()

    valid_idx = y.notna()
    X = X.loc[valid_idx]
    y = y.loc[valid_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    return X_train, X_test, y_train, y_test, feature_cols


# ==================== 模型训练 ====================
def train_and_predict(model_map, cat_cols, num_cols, X_train, y_train, X_test):
    """
    为每个模型构建完整 Pipeline（预处理 + 分类器），训练并预测。
    逻辑回归使用 StandardScaler，树模型不使用。
    """
    results = {}
    for name, model in model_map.items():
        scale = (name == 'LogisticRegression')
        pre = build_preprocessor(cat_cols, num_cols, scale_numeric=scale)
        pipe = Pipeline([('prep', pre), ('clf', model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)
        results[name] = {
            'pipeline': pipe,
            'model': model,
            'y_pred': y_pred,
            'y_proba': y_proba,
        }
    return results


# ==================== 模型持久化 ====================
def save_model(pipeline, name, feature_cols):
    path = os.path.join(MODEL_DIR, f'{name}.pkl')
    joblib.dump(pipeline, path)
    feat_path = os.path.join(MODEL_DIR, f'{name}_features.pkl')
    joblib.dump(feature_cols, feat_path)
    print(f'  已保存 {name} 到 {path}')
    return path
