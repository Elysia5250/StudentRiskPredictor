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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
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


# ==================== 数据准备 ====================
def prepare_data(df, label_col, cat_cols, num_cols):
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


# ==================== 缺失值预处理 ====================
_SENTINEL_COLS = [
    'first_submission_day_120', 'last_submission_day_120',
    'first_click_day_120', 'last_click_day_120',
]

def _fill_sentinel(X):
    X = X.copy()
    for col in _SENTINEL_COLS:
        if col in X.columns:
            X[col] = X[col].fillna(-1)
    return X


def _configure_model(model, name, scale_pos_weight_val):
    if name == 'LogisticRegression':
        model.set_params(class_weight='balanced')
    elif name == 'RandomForest':
        model.set_params(class_weight='balanced_subsample')
    elif name == 'LightGBM':
        model.set_params(
            scale_pos_weight=scale_pos_weight_val,
            num_leaves=63, learning_rate=0.05, n_estimators=300,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
        )
    elif name == 'XGBoost':
        model.set_params(
            scale_pos_weight=scale_pos_weight_val,
            learning_rate=0.05, n_estimators=300,
            subsample=0.8, colsample_bytree=0.8,
            max_depth=6, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=0.1,
        )


# ==================== 1. 特征筛选 ====================
def select_features(X_train, y_train, cat_cols, num_cols, keep_ratio=0.8):
    """
    用 LightGBM 训练一次，按累积重要性保留 top keep_ratio 的特征。
    返回筛选后的 cat_cols, num_cols。
    """
    try:
        import lightgbm as lgb
    except ImportError:
        return cat_cols, num_cols

    all_cols = cat_cols + num_cols
    # 对类别列简单编码用于初步筛选
    X_enc = X_train[all_cols].copy()
    for c in cat_cols:
        X_enc[c] = X_enc[c].astype('category').cat.codes
    X_enc = X_enc.fillna(0)

    model = lgb.LGBMClassifier(
        n_estimators=100, random_state=RANDOM_STATE,
        verbose=-1, n_jobs=-1
    )
    model.fit(X_enc, y_train)

    imp = pd.DataFrame({'feature': all_cols, 'importance': model.feature_importances_})
    imp = imp.sort_values('importance', ascending=False).reset_index(drop=True)
    imp['cumsum'] = imp['importance'].cumsum() / imp['importance'].sum()
    keep = imp[imp['cumsum'] <= keep_ratio]['feature'].tolist()
    # 确保至少保留一半特征
    if len(keep) < len(all_cols) // 2:
        keep = imp.head(max(len(all_cols) // 2, 1))['feature'].tolist()

    removed = [c for c in all_cols if c not in keep]
    print(f'  特征筛选: {len(all_cols)} → {len(keep)}（移除 {len(removed)} 个低贡献特征）')

    cat_cols_new = [c for c in cat_cols if c in keep]
    num_cols_new = [c for c in num_cols if c in keep]
    return cat_cols_new, num_cols_new


# ==================== 2. 交叉验证 ====================
def cross_validate_models(model_map, cat_cols, num_cols, X_train, y_train, n_folds=5):
    """
    对每个模型做 n_folds 折交叉验证，返回 {name: {fold_auc: [], mean_auc, std_auc}}。
    """
    X = _fill_sentinel(X_train)
    y = y_train
    neg, pos = y.value_counts().sort_index()
    scale_pos = neg / pos if pos > 0 else 1.0

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, model in model_map.items():
        _configure_model(model, name, scale_pos)
        aucs = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            scale = (name == 'LogisticRegression')
            pre = build_preprocessor(cat_cols, num_cols, scale_numeric=scale)
            pipe = Pipeline([('prep', pre), ('clf', model)])
            pipe.fit(X_fold_train, y_fold_train)
            proba = pipe.predict_proba(X_fold_val)[:, 1]
            from sklearn.metrics import roc_auc_score
            aucs.append(roc_auc_score(y_fold_val, proba))
        results[name] = {
            'aucs': [round(a, 4) for a in aucs],
            'mean': round(np.mean(aucs), 4),
            'std': round(np.std(aucs), 4),
        }
        print(f'    {name}: CV AUC = {results[name]["mean"]} ± {results[name]["std"]}')

    return results


# ==================== 3. 训练 + 校准 ====================
def train_and_predict(model_map, cat_cols, num_cols, X_train, y_train, X_test,
                      calibrate=False, return_val=False):
    """
    训练所有模型，可选返回校准后的结果。
    if return_val: 从 X_train 留出 val 集用于阈值搜索。
    """
    if return_val:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train
        )
    else:
        X_val, y_val = None, None

    X_train = _fill_sentinel(X_train)
    X_test = _fill_sentinel(X_test)
    if X_val is not None:
        X_val = _fill_sentinel(X_val)

    neg, pos = y_train.value_counts().sort_index()
    scale_pos = neg / pos if pos > 0 else 1.0

    results = {}
    for name, model in model_map.items():
        _configure_model(model, name, scale_pos)
        scale = (name == 'LogisticRegression')
        pre = build_preprocessor(cat_cols, num_cols, scale_numeric=scale)

        if calibrate and name in ('LightGBM', 'XGBoost', 'RandomForest'):
            # 用内部交叉验证校准概率
            inner_model = model.__class__(**model.get_params())
            inner_pipe = Pipeline([('prep', pre), ('clf', inner_model)])
            cal = CalibratedClassifierCV(inner_pipe, cv=3, method='sigmoid')
            cal.fit(X_train, y_train)
            pipe = cal
            y_pred = cal.predict(X_test)
            y_proba = cal.predict_proba(X_test)
        else:
            pipe = Pipeline([('prep', pre), ('clf', model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = pipe.predict_proba(X_test)

        entry = {'pipeline': pipe, 'model': model, 'y_pred': y_pred, 'y_proba': y_proba}

        # 验证集预测（用于阈值搜索）
        if X_val is not None:
            if calibrate and name in ('LightGBM', 'XGBoost', 'RandomForest'):
                val_proba = pipe.predict_proba(X_val)
            else:
                val_proba = pipe.predict_proba(X_val)
            entry['val_proba'] = val_proba[:, 1]
            entry['y_val'] = y_val.values

        results[name] = entry
    return results


# ==================== 4. 集成预测 ====================
def ensemble_predict(models_results, model_names, X_test, cat_cols, num_cols):
    """
    对指定模型列表做概率平均集成，返回和 train_and_predict 相同格式的 entry。
    """
    valid_names = [n for n in model_names if n in models_results]
    if len(valid_names) < 2:
        return None

    probas = []
    for name in valid_names:
        probas.append(models_results[name]['y_proba'][:, 1])
    avg_proba = np.mean(probas, axis=0)
    y_pred = (avg_proba >= 0.5).astype(int)

    # 构造 y_proba 二维数组
    y_proba = np.zeros((len(avg_proba), 2))
    y_proba[:, 1] = avg_proba
    y_proba[:, 0] = 1 - avg_proba

    return {
        'pipeline': None,
        'model': None,
        'y_pred': y_pred,
        'y_proba': y_proba,
    }


# ==================== 5. 阈值优化 ====================
def find_optimal_threshold(y_true, y_proba, metric='f1', n_thresholds=100):
    """
    在验证集上搜索最优概率截断点。
    metric: 'f1' | 'recall90' (在 recall>=0.9 前提下最大化 precision)
    返回 best_threshold, best_score, 及所有阈值的结果列表。
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    results = []
    best = {'threshold': 0.5, 'score': 0}

    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if metric == 'recall90':
            score = prec if rec >= 0.90 else 0
        else:
            score = f1

        results.append({'threshold': round(t, 3), 'recall': round(rec, 4),
                        'precision': round(prec, 4), 'f1': round(f1, 4)})
        if score > best['score']:
            best = {'threshold': round(t, 3), 'score': round(score, 4),
                    'recall': round(rec, 4), 'precision': round(prec, 4)}

    return best['threshold'], results


# ==================== 模型持久化 ====================
def save_model(pipeline, name, feature_cols):
    path = os.path.join(MODEL_DIR, f'{name}.pkl')
    joblib.dump(pipeline, path)
    feat_path = os.path.join(MODEL_DIR, f'{name}_features.pkl')
    joblib.dump(feature_cols, feat_path)
    print(f'  已保存 {name} 到 {path}')
    return path
