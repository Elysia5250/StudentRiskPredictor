import os
import pandas as pd
import numpy as np
from config import DATA_DIR, WIDE_TABLE, LABEL_COL, EXCLUDED_COLS, CATEGORICAL_COLS


# ==================== 加载特征宽表 ====================
def load_wide_table():
    """
    直接读取预处理好的特征宽表 OULAD_Final_Feature_Matrix.csv。
    不再从七张原始 OULAD 表合并。
    """
    path = os.path.join(DATA_DIR, WIDE_TABLE)
    df = pd.read_csv(path, low_memory=False)
    print(f'  宽表加载完成: {len(df):,} 行, {len(df.columns)} 列')
    return df


# ==================== 自动识别字段类型 ====================
def classify_columns(df):
    """
    自动识别字段类型：
      - label_col: academic_risk
      - excluded: 禁止入模的列
      - cat_cols: 已定义的类别特征（只取表中实际存在的）
      - num_cols: 其余全部作为数值特征
    """
    label_col = LABEL_COL if LABEL_COL in df.columns else None

    excluded = [c for c in EXCLUDED_COLS if c in df.columns]

    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    feature_candidates = [c for c in df.columns
                          if c not in excluded and c != label_col]
    num_cols = [c for c in feature_candidates if c not in cat_cols]

    print(f'  标签列: {label_col}')
    print(f'  排除列 ({len(excluded)}): {excluded}')
    print(f'  类别特征 ({len(cat_cols)}): {cat_cols}')
    print(f'  数值特征 ({len(num_cols)}): {num_cols[:5]}{"..." if len(num_cols) > 5 else ""}')

    return label_col, excluded, cat_cols, num_cols


def check_missing(df, cat_cols, num_cols):
    """检查缺失值概况"""
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f'  存在缺失值的列 ({len(missing)}):')
        for col, v in missing.items():
            print(f'    {col}: {v} ({v/len(df)*100:.2f}%)')
    else:
        print('  无缺失值')


def check_duplicates(df, key_cols=['id_student', 'code_module', 'code_presentation']):
    """检查主键重复"""
    existing = [c for c in key_cols if c in df.columns]
    if existing:
        dups = df.duplicated(subset=existing).sum()
        if dups > 0:
            print(f'  警告: 发现 {dups} 条主键重复记录')
        else:
            print('  主键无重复')
