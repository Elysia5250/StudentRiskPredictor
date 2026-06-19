#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高校学生学业风险预测系统
===========================
功能：
  1. 自动分析数据集
  2. 机器学习建模（逻辑回归/决策树/随机森林）
  3. 模型评估
  4. 特征重要性分析
  5. 可视化图表
  6. 生成课程报告素材
"""

import os
import sys
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

# ============================================================
# 中文字体自动适配
# ============================================================
_ZH_FONTS = ['Arial Unicode MS', 'PingFang SC', 'Microsoft YaHei', 'SimHei', 'WenQuanYi Micro Hei']
for _f in _ZH_FONTS:
    try:
        plt.rcParams['font.sans-serif'] = [_f]
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 辅助函数
# ============================================================

def find_csv_files():
    """查找 data/ 目录下所有 CSV 文件"""
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    return files


def auto_detect_label_column(df):
    """自动推断标签列"""
    keywords = [
        'target', 'label', 'class', 'grade', 'result',
        '挂科', 'fail', 'pass', 'y', 'output',
        '成绩', '等级', '是否挂科', 'is_fail', 'is_pass',
        'status', 'flag', 'category', 'outcome',
    ]
    for col in df.columns:
        cl = col.lower().strip()
        for kw in keywords:
            if cl == kw or (len(kw) > 1 and kw in cl):
                return col
    last = df.columns[-1]
    if df[last].nunique() <= 10:
        return last
    for col in df.columns:
        if col == last:
            continue
        if 2 <= df[col].nunique() <= 10:
            return col
    return last


def auto_detect_feature_columns(df, label_col):
    """自动识别特征列，排除 ID 类字段"""
    id_kw = ['id', '学号', 'student', '编号', 'code', 'num']
    features = []
    drops = []
    for col in df.columns:
        if col == label_col:
            continue
        cl = col.lower().strip()
        if any(kw in cl for kw in id_kw):
            drops.append(col)
        else:
            features.append(col)
    return features, drops


# ============================================================
# 任务1：数据集分析
# ============================================================

def task1_analyze(df, csv_path):
    print('=' * 60)
    print('【任务1】数据集分析')
    print('=' * 60)

    label_col = auto_detect_label_column(df)
    feature_cols, drop_cols = auto_detect_feature_columns(df, label_col)
    n_rows, n_cols = df.shape

    print(f'标签列: [{label_col}]')
    print(f'特征列 ({len(feature_cols)}): {feature_cols}')
    if drop_cols:
        print(f'忽略列: {drop_cols}')

    # 缺失值
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    # 数据类型
    dtype_counts = df.dtypes.value_counts()

    # 写入报告
    lines = []
    lines.append('数据集分析报告')
    lines.append('=' * 50)
    lines.append(f'')
    lines.append(f'数据文件:     {os.path.basename(csv_path)}')
    lines.append(f'数据行数:     {n_rows}')
    lines.append(f'数据列数:     {n_cols}')
    lines.append(f'特征字段:     {feature_cols}')
    lines.append(f'标签字段:     {label_col}')
    if drop_cols:
        lines.append(f'忽略的字段:   {drop_cols}')
    lines.append(f'')
    lines.append(f'缺失值统计:')
    if len(missing):
        for col, v in missing.items():
            lines.append(f'  {col}: {v} ({v / n_rows * 100:.2f}%)')
    else:
        lines.append(f'  无缺失值')
    lines.append(f'')
    lines.append(f'数据类型统计:')
    for dt, cnt in dtype_counts.items():
        lines.append(f'  {dt}: {cnt} 列')
    lines.append(f'')
    lines.append(f'前5行数据:')
    lines.append(df.head().to_string())

    report = '\n'.join(lines)
    rpath = os.path.join(OUTPUT_DIR, 'dataset_report.txt')
    with open(rpath, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\n报告已保存: {rpath}')

    return label_col, feature_cols, drop_cols, report


# ============================================================
# 任务2 & 3：建模 & 评估
# ============================================================

def task2_3_train_and_evaluate(df, label_col, feature_cols, drop_cols):
    print('\n' + '=' * 60)
    print('【任务2 & 3】机器学习建模与评估')
    print('=' * 60)

    X = df[feature_cols].copy()
    y = df[label_col].copy()

    # 编码标签
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # 区分数值/类别列
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f'数值特征: {numeric_cols}')
    print(f'类别特征: {categorical_cols}')

    # 预处理管道
    transformers = []
    if numeric_cols:
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        transformers.append(('num', num_pipe, numeric_cols))
    if categorical_cols:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ])
        transformers.append(('cat', cat_pipe, categorical_cols))

    preprocessor = ColumnTransformer(transformers, remainder='drop')

    # 划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc,
    )
    print(f'\n训练集: {X_train.shape[0]}  测试集: {X_test.shape[0]}')

    # 模型定义
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = []
    pipelines = {}

    for name, clf in models.items():
        pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe

        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        results.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
        })
        print(f'\n{name}')
        print(f'  Accuracy:  {acc:.4f}')
        print(f'  Precision: {prec:.4f}')
        print(f'  Recall:    {rec:.4f}')
        print(f'  F1 Score:  {f1:.4f}')

    rdf = pd.DataFrame(results)
    rpath = os.path.join(OUTPUT_DIR, 'model_metrics.csv')
    rdf.to_csv(rpath, index=False)
    print(f'\n评估结果已保存: {rpath}')

    return rdf, pipelines, preprocessor, numeric_cols, categorical_cols, le


# ============================================================
# 任务4：特征重要性
# ============================================================

def task4_feature_importance(preprocessor, pipelines, numeric_cols, categorical_cols):
    print('\n' + '=' * 60)
    print('【任务4】特征重要性分析')
    print('=' * 60)

    if 'Random Forest' not in pipelines:
        print('跳过：没有 Random Forest 模型')
        return None

    rf = pipelines['Random Forest'].named_steps['clf']
    imp = rf.feature_importances_

    # 重构特征名
    names = []
    if numeric_cols:
        names.extend(numeric_cols)
    if categorical_cols:
        cat_tfm = preprocessor.named_transformers_.get('cat')
        if cat_tfm is not None:
            ohe = cat_tfm.named_steps['ohe']
            names.extend(ohe.get_feature_names_out(categorical_cols))

    # 如果长度不匹配则截断
    n = min(len(imp), len(names))
    imp = imp[:n]
    names = names[:n]

    imp_df = pd.DataFrame({'feature': names, 'importance': imp})
    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)

    rpath = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
    imp_df.to_csv(rpath, index=False)
    print(f'特征重要性已保存: {rpath}')

    print('\nTop-10 重要特征:')
    for i, row in imp_df.head(10).iterrows():
        print(f'  {i+1}. {row["feature"]}: {row["importance"]:.4f}')

    return imp_df


# ============================================================
# 任务5：可视化
# ============================================================

def task5_charts(imp_df, results_df, df):
    print('\n' + '=' * 60)
    print('【任务5】生成可视化图表')
    print('=' * 60)

    colors_bar = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
                  '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']

    # ---------- 1. 特征重要性 ----------
    if imp_df is not None and len(imp_df):
        top_n = min(15, len(imp_df))
        plot_data = imp_df.head(top_n).iloc[::-1]
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(top_n), plot_data['importance'],
                       color=[colors_bar[i % len(colors_bar)] for i in range(top_n)])
        ax.set_yticks(range(top_n))
        ax.set_yticklabels(plot_data['feature'], fontsize=11)
        ax.set_xlabel('重要性', fontsize=13)
        ax.set_title('特征重要性排名 (Random Forest)', fontsize=15, fontweight='bold')
        for bar, val in zip(bars, plot_data['importance']):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9)
        plt.tight_layout()
        p1 = os.path.join(OUTPUT_DIR, 'feature_importance.png')
        plt.savefig(p1, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'特征重要性图: {p1}')

    # ---------- 2. 模型对比 ----------
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))
    width = 0.2
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    mcolors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    for i, m in enumerate(metrics):
        ax.bar(x + i * width, results_df[m], width, label=m, color=mcolors[i])

    ax.set_xlabel('模型', fontsize=13)
    ax.set_ylabel('得分', fontsize=13)
    ax.set_title('模型性能对比', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df['Model'], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.08)

    for i, m in enumerate(metrics):
        for j, v in enumerate(results_df[m]):
            ax.text(j + i * width, v + 0.008, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8, rotation=45)

    plt.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, 'model_compare.png')
    plt.savefig(p2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'模型对比图: {p2}')

    # ---------- 3. 相关性热力图 ----------
    ndf = df.select_dtypes(include=[np.number])
    if ndf.shape[1] >= 2:
        corr = ndf.corr()
        sz = max(10, corr.shape[1] * 0.7)
        sz2 = max(8, corr.shape[0] * 0.7)
        fig, ax = plt.subplots(figsize=(sz, sz2))
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8, label='相关系数')

        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(corr.columns, fontsize=9)
        ax.set_title('特征相关性热力图', fontsize=15, fontweight='bold')

        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                v = corr.iloc[i, j]
                c = 'white' if abs(v) > 0.5 else 'black'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=7, color=c)

        plt.tight_layout()
        p3 = os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')
        plt.savefig(p3, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'相关性热力图: {p3}')
    else:
        print('数值列不足2列，跳过相关性热力图')


# ============================================================
# 任务6：报告素材
# ============================================================

def task6_report(label_col, feature_cols, drop_cols, results_df, imp_df, df):
    print('\n' + '=' * 60)
    print('【任务6】生成课程报告素材')
    print('=' * 60)

    best = results_df.loc[results_df['F1'].idxmax()]
    top5 = []
    if imp_df is not None:
        top5 = imp_df.head(5)['feature'].tolist()

    lines = []
    lines.append('# 高校学生学业风险预测 — 课程报告素材')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 一、数据集介绍')
    lines.append('')
    lines.append(f'本实验采用的数据集为CSV格式，共包含 **{df.shape[0]:,}** 条样本记录和 **{df.shape[1]}** 个字段。经自动分析，标签列为 **{label_col}**，特征字段共 **{len(feature_cols)}** 个。')
    lines.append('')
    lines.append('**数据集概览：**')
    lines.append(f'- 样本总数：{df.shape[0]:,}')
    lines.append(f'- 特征总数：{len(feature_cols)}')
    lines.append(f'- 标签字段：{label_col}')
    if drop_cols:
        _drop_str = ', '.join(drop_cols)
        lines.append(f'- 排除的ID字段：{_drop_str}')
    lines.append('')
    lines.append('## 二、数据预处理方法')
    lines.append('')
    lines.append('本研究以 scikit-learn 为核心工具，预处理流程如下：')
    lines.append('')
    lines.append('1. **标签列自动识别** — 通过关键词匹配和列特性分析自动推断。')
    lines.append('2. **ID列过滤** — 自动排除学号、编号等无关字段。')
    lines.append('3. **缺失值处理**：')
    lines.append('   - 数值特征：中位数填充')
    lines.append('   - 类别特征：众数填充')
    lines.append('4. **类别特征编码** — One-Hot Encoding。')
    lines.append('5. **数值特征标准化** — StandardScaler（零均值单位方差）。')
    lines.append('6. **数据集划分** — 80% 训练 / 20% 测试（random_state=42，分层抽样）。')
    lines.append('')
    lines.append('## 三、三种模型简介')
    lines.append('')
    lines.append('### 3.1 逻辑回归 (Logistic Regression)')
    lines.append('')
    lines.append('广义线性模型，通过 Sigmoid 函数将线性输出映射到 [0,1] 区间。可解释性强、计算高效，可输出各特征的权重系数。')
    lines.append('')
    lines.append('### 3.2 决策树 (Decision Tree)')
    lines.append('')
    lines.append('基于树结构的分类模型，递归选择最优特征划分数据，形成 if-then 规则。无需标准化、直观可解释，但易过拟合。')
    lines.append('')
    lines.append('### 3.3 随机森林 (Random Forest)')
    lines.append('')
    lines.append('基于 Bagging 的集成方法，构建多棵决策树并投票决策。抗过拟合、对缺失值鲁棒，可输出特征重要性排序。')
    lines.append('')
    lines.append('## 四、模型评估结果分析')
    lines.append('')
    lines.append('### 4.1 评估指标')
    lines.append('')
    lines.append('- **准确率 (Accuracy)**：预测正确的样本占比。')
    lines.append('- **精确率 (Precision)**：预测为正类中实际为正类的比例。')
    lines.append('- **召回率 (Recall)**：实际为正类中被正确预测的比例。')
    lines.append('- **F1 分数**：精确率与召回率的调和平均数。')
    lines.append('')
    lines.append('### 4.2 评估结果')
    lines.append('')
    lines.append('| 模型 | Accuracy | Precision | Recall | F1 |')
    lines.append('|------|----------|-----------|--------|-----|')
    for _, row in results_df.iterrows():
        lines.append(f'| {row["Model"]} | {row["Accuracy"]:.4f} | {row["Precision"]:.4f} | {row["Recall"]:.4f} | {row["F1"]:.4f} |')
    lines.append('')
    lines.append(f'### 4.3 结果分析')
    lines.append('')
    lines.append(f'三个模型中 **{best["Model"]}** 取得了最高的 F1 分数（{best["F1"]:.4f}），表明其在精确率和召回率之间取得了最佳平衡。')
    lines.append('')
    if imp_df is not None:
        lines.append('## 五、特征重要性分析')
        lines.append('')
        lines.append('通过随机森林的 `feature_importances_` 评估各特征对预测的贡献程度。得分越高，预测能力越强。')
        lines.append('')
        lines.append('**Top-5 重要特征：**')
        lines.append('')
        for i, row in imp_df.head(5).iterrows():
            lines.append(f'{i+1}. **{row["feature"]}** — 重要性 {row["importance"]:.4f}')
        lines.append('')
    lines.append('## 六、项目结论')
    lines.append('')
    lines.append(f'本研究基于机器学习构建了学生学业风险预测模型，主要结论如下：')
    lines.append('')
    lines.append(f'1. **模型有效性**：三种分类模型均在测试集上表现良好，其中 **{best["Model"]}** 最优（F1 = {best["F1"]:.4f}），验证了机器学习在学业风险预测中的可行性。')
    lines.append(f'2. **关键影响因素**：特征重要性分析识别出了影响学生学业表现的关键因素，可为学校学业预警和教学干预提供参考。')
    lines.append(f'3. **方法论价值**：本系统建立了完整的数据分析→建模→评估→可视化流程，可复现且易扩展。')
    lines.append('')
    lines.append('---')
    ts = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')
    lines.append(f'*本报告由自动化分析系统生成 — {ts}*')

    report = '\n'.join(lines)
    rpath = os.path.join(OUTPUT_DIR, 'report_material.md')
    with open(rpath, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'课程报告已保存: {rpath}')


# ============================================================
# 主流程
# ============================================================

def main():
    print('=' * 60)
    print('  高校学生学业风险预测系统')
    print('=' * 60)

    csv_files = find_csv_files()
    if not csv_files:
        print(f'\n[提示] 未在 {DATA_DIR}/ 下找到 CSV 文件。')
        print(f'请将 CSV 文件放入 {DATA_DIR}/ 目录后重新运行。')
        print('运行方式:  python train.py')
        sys.exit(0)

    # 多文件选择
    if len(csv_files) > 1:
        print(f'\n发现 {len(csv_files)} 个 CSV 文件:')
        for i, fp in enumerate(csv_files):
            print(f'  [{i+1}] {os.path.basename(fp)}')
        while True:
            raw = input(f'\n请选择 (1-{len(csv_files)}): ').strip()
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(csv_files):
                    break
            except ValueError:
                pass
            print('输入无效，请重新输入。')
        csv_path = csv_files[idx]
    else:
        csv_path = csv_files[0]

    # 读取数据
    print(f'\n读取数据: {os.path.basename(csv_path)}')
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')
    except Exception as e:
        print(f'读取失败: {e}')
        sys.exit(1)

    print(f'数据维度: {df.shape[0]} 行 × {df.shape[1]} 列\n')

    # ---- 任务1 ----
    label_col, feature_cols, drop_cols, _ = task1_analyze(df, csv_path)

    # ---- 任务2 & 3 ----
    results_df, pipelines, preprocessor, numeric_cols, categorical_cols, label_encoder = \
        task2_3_train_and_evaluate(df, label_col, feature_cols, drop_cols)

    # ---- 任务4 ----
    imp_df = task4_feature_importance(preprocessor, pipelines, numeric_cols, categorical_cols)

    # ---- 任务5 ----
    task5_charts(imp_df, results_df, df)

    # ---- 任务6 ----
    task6_report(label_col, feature_cols, drop_cols, results_df, imp_df, df)

    print('\n' + '=' * 60)
    print('  所有任务完成！输出文件在 output/ 目录下。')
    print('=' * 60)


if __name__ == '__main__':
    main()
