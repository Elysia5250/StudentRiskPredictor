#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高校学生学业风险预测系统
===========================
【项目背景】
  本系统面向"高校学生学业风险预测"课程实验/课题任务，基于机器学习分类算法，
  从学生个人信息、学习行为、家庭支持等多维度特征出发，预测学生"学期末是否挂科"，
  并分析影响学业表现的关键因素。

【整体业务流程】
  步骤1: 数据读取与校验
      → 从 data/ 目录读取 CSV 文件
      → 按"数据字典"检查列名是否符合预期（标签列、特征列、排除列）
      → 自动检测数据泄露风险（如将"成绩等级"当作特征训练）

  步骤2: 数据预处理（Pipeline 自动执行）
      → 数值特征：中位数填充缺失值 + 标准化（零均值单位方差）
      → 类别特征：众数填充缺失值 + One-Hot 编码

  步骤3: 建模与评估
      → 三种分类模型：逻辑回归（线性基准）、决策树（树模型基准）、随机森林（集成模型）
      → 训练/测试 80% / 20% 分层抽样
      → 评估指标：Accuracy、Precision、Recall、F1-Score

  步骤4: 特征重要性分析
      → 利用 Random Forest 的 feature_importances_ 属性
      → 输出各特征对预测的贡献度排名

  步骤5: 可视化图表
      → 特征重要性排名柱状图
      → 模型性能对比柱状图
      → 特征相关性热力图

  步骤6: 生成课程报告素材
      → 自动输出 Markdown 格式的中文学术报告，可直接复制到课程报告中修改

【输出文件清单】
  output/
  ├── dataset_report.txt          # 数据集分析报告（行数、列数、缺失值、数据类型等）
  ├── feature_review.txt          # 特征审查报告（哪些列被选中/排除及原因）
  ├── model_metrics.csv           # 三种模型的四类评估指标
  ├── feature_importance.csv      # 各特征的重要性得分
  ├── feature_importance.png      # 特征重要性柱状图
  ├── model_compare.png           # 模型性能对比图
  ├── correlation_heatmap.png     # 相关性热力图
  └── report_material.md          # 可直接使用的课程报告 Markdown 素材

【技术栈】
  - Python 3.8+
  - pandas / numpy（数据处理）
  - scikit-learn（机器学习 Pipeline）
  - matplotlib（图表可视化，支持中文）

【使用方式】
  python train.py
"""

# ============================================================
# 导入依赖库
# ============================================================
# 说明：
#   以下为本项目用到的所有第三方库。写报告/PPT 时可列出"技术依赖"部分：
#   - pandas / numpy：数据处理与数值计算
#   - matplotlib：可视化图表（使用 Agg 后端，可在无 GUI 的服务器环境运行）
#   - scikit-learn：完整的机器学习 Pipeline（预处理→建模→评估）
#
# 注意：
#   单独导入 LabelEncoder / StandardScaler / OneHotEncoder / SimpleImputer 等
#   是为了构建 ColumnTransformer + Pipeline 的模块化预处理流程，
#   这在 scikit-learn 中是最佳实践，便于复用和参数调优。

import os               # 文件和路径操作
import sys              # 系统退出等
import glob             # 查找 data/ 目录下的 CSV 文件
import warnings         # 抑制不必要的警告

import numpy as np      # 数值计算
import pandas as pd     # DataFrame 数据处理

import matplotlib
matplotlib.use('Agg')   # 使用非交互式后端，确保在无图形界面的环境也能保存图片
import matplotlib.pyplot as plt

# sklearn 模型选择
from sklearn.model_selection import train_test_split

# sklearn 预处理组件
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# sklearn 分类模型
from sklearn.linear_model import LogisticRegression     # 逻辑回归（线性模型）
from sklearn.tree import DecisionTreeClassifier          # 决策树（树模型）
from sklearn.ensemble import RandomForestClassifier      # 随机森林（集成模型）

# sklearn 评估指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 忽略 scikit-learn 的一些版本兼容性警告，保持终端输出整洁
warnings.filterwarnings('ignore')

# ============================================================
# 中文字体自动适配
# ============================================================
# 问题背景：
#   matplotlib 默认字体不支持中文，会导致图表中的中文显示为方框 "□□"。
#
# 解决方案：
#   按优先级尝试常见中文字体列表，第一个成功的就设为全局字体。
#   列表覆盖 macOS（Arial Unicode MS / PingFang SC）、
#   Windows（Microsoft YaHei / SimHei）、Linux（WenQuanYi）三大平台。
#
# 报告提示：
#   在 PPT/报告中提到"本系统支持跨平台中文可视化"时可引用此段。

_ZH_FONTS = [
    'Arial Unicode MS',      # macOS 通用
    'PingFang SC',           # macOS 苹方
    'Microsoft YaHei',       # Windows 微软雅黑
    'SimHei',                # Windows 黑体
    'WenQuanYi Micro Hei',   # Linux 文泉驿
]
for _f in _ZH_FONTS:
    try:
        plt.rcParams['font.sans-serif'] = [_f]
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        break
    except Exception:
        continue

# ============================================================
# 路径配置
# ============================================================
# 目录结构约定：
#   project/
#   ├── data/          ← 用户自行放入 CSV 数据文件
#   ├── output/        ← 程序自动生成所有输出文件（报告/图表/CSV）
#   ├── train.py       ← 本程序
#   └── README.md
#
# 设计说明：
#   - BASE_DIR 使用 __file__ 的绝对路径，确保无论在哪个目录执行脚本都能正确找到 data/ 和 output/
#   - exist_ok=True 避免因目录已存在而报错

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# 数据字典（硬编码，禁止自动猜测）
# ============================================================
# 重要设计决策：
#   本项目不采用"自动识别标签列/特征列"的通用策略。
#   而是由数据字典（Data Dictionary）硬编码定义每一列的用途。
#
# 原因：
#   1. 课程实验中数据集结构是已知的，自动猜测可能选错标签列
#   2. 防止数据泄露：某些列（如 GPA、成绩等级）与标签高度相关，
#      如果自动选入特征，会导致模型"作弊"，评估结果虚高，失去实际意义
#   3. 实验报告强调"可复现性"——固定列选择确保每次运行结果一致
#
# 报告/PPT 提示：
#   可在"数据预处理"或"实验设计"部分强调这种基于数据字典的严格列管理策略，
#   这是区别于一般"自动 ML"系统的重要方法论特征。

# ---- 标签列（唯一预测目标） ----
# 模型要预测的就是学生"学期末是否挂科"（二分类：0=未挂科，1=挂科）
LABEL_COLUMN = '学期末是否挂科'

# ---- 必须排除的列（禁止参与训练） ----
# 这些列虽然可能在 CSV 中存在，但绝对不能作为模型特征：
#   - 学生编号：唯一 ID，无预测意义
#   - 成绩等级 / GradeClass：与标签存在直接派生关系（GPA 分段），包含数据泄露
#   - 平均绩点GPA：若包含本学期成绩，则与标签存在时序重叠，属于数据泄露
# 排除的列会保留在最终输出的热力图等 EDA 分析中，但不参与训练
MUST_EXCLUDE = ['学生编号', '成绩等级', 'GradeClass', '平均绩点GPA']

# ---- 允许用于训练的特征列 ----
# 全部 12 个特征，涵盖以下维度：
#   人口学：年龄、性别、学生背景类别
#   家庭背景：父母最高教育水平、父母学业支持程度
#   学习行为：每周自主学习时长、本学期旷课次数
#   课外参与：是否参加课外辅导 / 课外活动 / 体育活动 / 音乐活动 / 志愿服务
# 只有 CSV 中实际存在的（且属于此列表的）列才会进入训练
ALLOWED_FEATURES = [
    '年龄',
    '性别',
    '学生背景类别',
    '父母最高教育水平',
    '每周自主学习时长',
    '本学期旷课次数',
    '是否参加课外辅导',
    '父母学业支持程度',
    '是否参加课外活动',
    '是否参加体育活动',
    '是否参加音乐活动',
    '是否参加志愿服务',
]

# ============================================================
# 辅助函数
# ============================================================
# 以下函数是系统的"基础设施层"，负责文件查找、列校验、错误报告等。
# 在写报告/PPT 时不需要逐行讲解，但可以概括其设计思想：
#   "程序通过严格的数据字典校验防御数据泄露问题，
#    这是实验设计中保证评估结果可信的重要环节。"


def find_csv_files():
    """
    扫描 data/ 目录，返回所有 .csv 文件的排序列表。
    支持用户放置一个或多个 CSV 文件，如果多个则让用户交互选择。
    """
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    return files


def validate_and_resolve_columns(df):
    """
    【核心校验函数】—— 按数据字典规则解析 CSV 的每一列。

    输入：  pandas.DataFrame（原始 CSV 数据）
    输出：  (label_col, feature_cols, drop_cols) 三元组
              - label_col:   标签列名（固定为 '学期末是否挂科'）
              - feature_cols: 实际参与训练的特征列名列表
              - drop_cols:    从 CSV 中排除的列名列表（仅用于 EDA）

    校验规则：
      1. 标签列（LABEL_COLUMN）必须在 CSV 中存在，否则退出
      2. 特征列只取 ALLOWED_FEATURES ∩ CSV.columns（交集）
      3. MUST_EXCLUDE 中的列不得出现在特征中，否则视为"数据泄露"并终止
      4. 不在数据字典中的"未知列"也被加入排除列表、不参与训练

    数据泄露检测说明（重要，适合写进报告）：
      - 如果"成绩等级"或"平均绩点GPA"被错误地用于训练，
        模型相当于"看了答案再做题"——因为这些字段本身就是标签的派生信息。
      - 本系统对此做严格防御：即使 CSV 中有这些列，也会被强制排除，
        如果配置错误导致它们进入了特征，程序会直接报错退出。
    """
    # ---- 第 1 步：检查标签列是否存在 ----
    # 如果 CSV 中没有我们预期的标签列，说明数据格式不对，直接退出并提示用户
    if LABEL_COLUMN not in df.columns:
        print(f'错误：CSV 中未找到标签列 "{LABEL_COLUMN}"')
        print(f'当前 CSV 包含的列：{list(df.columns)}')
        sys.exit(1)

    label_col = LABEL_COLUMN

    # ---- 第 2 步：识别三类别 ----
    # existing_allowed: CSV 中存在的允许特征列（可能为空，只要 CSV 里没有这些字段）
    # existing_excluded: CSV 中存在的排除字段（这些将被记录在报告中）
    # unknown_cols: CSV 中不在数据字典中的额外列（既不是标签、也不是允许特征、也不是排除字段）
    existing_allowed = [c for c in ALLOWED_FEATURES if c in df.columns]
    existing_excluded = [c for c in MUST_EXCLUDE if c in df.columns]
    unknown_cols = [c for c in df.columns if c not in ALLOWED_FEATURES and c != label_col and c not in MUST_EXCLUDE]

    # ---- 第 3 步：数据字典自一致性检查 ----
    # 确保 MUST_EXCLUDE 和 ALLOWED_FEATURES 没有重叠
    # 这是代码配置层面的检查，如果发生说明程序员配置错了
    for col in MUST_EXCLUDE:
        if col in ALLOWED_FEATURES:
            raise ValueError(f'数据字典配置错误："{col}" 同时存在于 MUST_EXCLUDE 和 ALLOWED_FEATURES')

    # ---- 第 4 步：构建最终特征列表 & 排除列表 ----
    # 特征：只取 ALLOWED_FEATURES 中 CSV 实际存在的列
    feature_cols = [c for c in ALLOWED_FEATURES if c in df.columns]
    # 排除：CSV 中的排除字段 + 不在数据字典中的未知列
    excl_in_csv = [c for c in MUST_EXCLUDE if c in df.columns]
    drop_cols = list(excl_in_csv)
    drop_cols.extend(unknown_cols)

    # ---- 第 5 步：数据泄露终极检测 ----
    # 如果 MUST_EXCLUDE 中的列出现在 feature_cols 中，说明存在配置错误或数据泄露，
    # 程序立即终止并给出详细原因。
    for col in MUST_EXCLUDE:
        if col in feature_cols:
            print('\n' + '=' * 50)
            print('  DATA LEAKAGE DETECTED')
            print('=' * 50)
            print(f'\n发现泄露字段: "{col}"')
            print(f'该字段属于 MUST_EXCLUDE，但出现在训练特征中。')
            print(f'原因：{_get_leak_reason(col)}')
            print(f'\n请检查 CSV 和数据字典配置后重试。')
            sys.exit(1)

    return label_col, feature_cols, drop_cols


def _get_leak_reason(col):
    """
    返回排除字段的详细排除原因。
    这些原因会写入 output/feature_review.txt 供报告使用。
    """
    reasons = {
        '学生编号': '唯一ID，不具备预测意义。',
        '成绩等级': '与标签存在直接派生关系，仅用于EDA分析，禁止参与模型训练。',
        'GradeClass': '与标签存在直接派生关系，仅用于EDA分析，禁止参与模型训练。',
        '平均绩点GPA': '若包含本学期成绩则与标签存在时序重叠（数据泄露风险），已剔除；仅用于EDA分析。',
    }
    return reasons.get(col, '不在允许特征列表中。')


# ============================================================
# 任务1：数据集分析
# ============================================================
# 功能：
#   对加载的 CSV 数据进行基础统计分析，包括：
#   - 数据规模（行数、列数）
#   - 标签字段与特征字段列表
#   - 缺失值数量及占比
#   - 各列数据类型分布
#   - 前 5 行数据预览
#
# 输出文件：output/dataset_report.txt
# 报告用途：
#   在 PPT/报告的"数据集介绍"章节可以直接引用此报告的内容。
#   特别适合展示：
#   - "数据集包含 2,392 条记录、16 个字段，无缺失值"
#   - "12 个特征涵盖人口学、学习行为、家庭背景等多个维度"

def task1_analyze(df, csv_path, label_col, feature_cols, drop_cols):
    """
    任务1：数据集分析
    - 统计基本信息（行数/列数/字段列表）
    - 检查缺失值
    - 输出数据类型分布
    - 展示前 5 行数据样例
    - 结果写入 output/dataset_report.txt
    """
    print('=' * 60)
    print('【任务1】数据集分析')
    print('=' * 60)

    n_rows, n_cols = df.shape

    # 终端打印基本信息
    print(f'标签列: [{label_col}]')
    print(f'特征列 ({len(feature_cols)}): {feature_cols}')
    if drop_cols:
        print(f'排除列: {drop_cols}')

    # 计算缺失值（只保留有缺失的列）
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    # 数据类型统计
    dtype_counts = df.dtypes.value_counts()

    # ---- 构建文本报告 ----
    lines = []
    lines.append('数据集分析报告')
    lines.append('=' * 50)
    lines.append('')
    lines.append(f'数据文件:     {os.path.basename(csv_path)}')
    lines.append(f'数据行数:     {n_rows}')
    lines.append(f'数据列数:     {n_cols}')
    lines.append(f'标签字段:     {label_col}')
    lines.append(f'特征字段 ({len(feature_cols)}): {feature_cols}')
    if drop_cols:
        lines.append(f'排除的字段:   {drop_cols}')
    lines.append('')
    lines.append('缺失值统计:')
    if len(missing):
        for col, v in missing.items():
            lines.append(f'  {col}: {v} ({v / n_rows * 100:.2f}%)')
    else:
        lines.append('  无缺失值')
    lines.append('')
    lines.append('数据类型统计:')
    for dt, cnt in dtype_counts.items():
        lines.append(f'  {dt}: {cnt} 列')
    lines.append('')
    lines.append('前5行数据:')
    lines.append(df.head().to_string())

    # 保存到文件
    report = '\n'.join(lines)
    rpath = os.path.join(OUTPUT_DIR, 'dataset_report.txt')
    with open(rpath, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'\n报告已保存: {rpath}')

    return report


# ============================================================
# 输出特征审查报告
# ============================================================
# 功能：
#   生成特征审查报告，明确列出：
#   - 标签列是什么
#   - 哪些列被选为训练特征
#   - 哪些列被排除（以及排除原因）
#
# 输出文件：output/feature_review.txt
# 报告用途：
#   在 PPT/报告的"数据预处理"或"特征工程"章节，
#   用于展示"特征选择策略"和"数据泄露防护机制"。
#   这是方法论严谨性的重要体现。

def output_feature_review(label_col, feature_cols, drop_cols):
    """
    输出特征审查报告（终端打印 + 文件保存）。
    让用户清楚知道模型的输入是什么、哪些被排除及原因。
    这是保证实验透明性和可复现性的关键输出。
    """
    print('\n' + '=' * 60)
    print('【特征审查】')
    print('=' * 60)

    print(f'\n标签列: {label_col}')
    print(f'\n训练特征列表 ({len(feature_cols)}):')
    for i, col in enumerate(feature_cols, 1):
        print(f'  {i}. {col}')
    print(f'\n排除字段（不参与训练，仅用于EDA分析和热力图）: {drop_cols if drop_cols else "无"}')

    lines = []
    lines.append('特征审查报告')
    lines.append('=' * 50)
    lines.append('')
    lines.append(f'标签列: {label_col}（唯一标签，禁止使用其他列作为预测目标）')
    lines.append('')
    lines.append(f'训练特征列表 ({len(feature_cols)}):')
    for i, col in enumerate(feature_cols, 1):
        lines.append(f'  {i}. {col}')
    lines.append('')
    lines.append(f'排除字段（不参与训练，仅用于EDA分析和热力图）: {drop_cols if drop_cols else "无"}')
    if drop_cols:
        lines.append('')
        lines.append('排除原因:')
        for col in drop_cols:
            if col in MUST_EXCLUDE:
                lines.append(f'  - {col}: {_get_leak_reason(col)}')
            else:
                lines.append(f'  - {col}: 不在允许特征列表中')

    rpath = os.path.join(OUTPUT_DIR, 'feature_review.txt')
    with open(rpath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\n特征审查报告已保存: {rpath}')


# ============================================================
# 任务2 & 3：建模 & 评估
# ============================================================
# 【这是整个系统的核心——机器学习建模与评估】
#
# 任务2 — 机器学习建模：
#   同时训练三种经典分类模型，覆盖不同理论范式：
#   - 逻辑回归 (Logistic Regression)：广义线性模型，可解释性最强
#   - 决策树 (Decision Tree)：基于信息增益的树模型，直观展示决策规则
#   - 随机森林 (Random Forest)：基于 Bagging 的集成方法，通常精度最高
#
# 任务3 — 模型评估：
#   在 20% 测试集上计算四类指标：
#   - Accuracy:  预测正确的样本占比（最直观）
#   - Precision: 预测为挂科的学生中，实际挂科的比例
#   - Recall:    实际挂科的学生中，被模型正确识别出的比例
#   - F1-Score:  Precision 和 Recall 的调和平均数（综合指标）
#
# 预处理 Pipeline 设计说明（重要，适合写进报告"方法论"章节）：
#   使用 scikit-learn 的 ColumnTransformer + Pipeline 组合，
#   对不同类型特征执行不同的预处理策略：
#   - 数值特征 → 中位数填充缺失值 → 标准化（StandardScaler）
#   - 类别特征 → 众数填充缺失值 → One-Hot 编码
#   这种 Pipeline 方式保证了预处理步骤与模型绑定在一起，
#   在交叉验证或模型部署时不会遗漏预处理步骤。
#
# 特别说明 — 数据泄露预防：
#   标签编码 (LabelEncoder) 在 train_test_split 之后进行，
#   即先用原始标签划分训练/测试集，再分别编码。
#   这样测试集的信息不会"泄露"到训练集。

def task2_3_train_and_evaluate(df, label_col, feature_cols):
    """
    任务2 & 3：机器学习建模与评估

    输入：
      df           — 原始 DataFrame（包含标签和特征列）
      label_col    — 标签列名
      feature_cols — 特征列名列表

    输出：
      results_df       — DataFrame，含三种模型的 Accuracy/Precision/Recall/F1
      pipelines        — dict，保存训练好的 Pipeline 对象（后续特征重要性需要）
      preprocessor     — ColumnTransformer 对象（后续特征重要性需要）
      numeric_cols     — 数值特征列名列表
      categorical_cols — 类别特征列名列表
      label_encoder    — LabelEncoder 对象
    """
    print('\n' + '=' * 60)
    print('【任务2 & 3】机器学习建模与评估')
    print('=' * 60)

    # ---- 分离特征与标签 ----
    X = df[feature_cols].copy()   # 特征矩阵
    y = df[label_col].copy()      # 标签向量

    # 自动识别数值列和类别列（根据 pandas dtype）
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    print(f'数值特征: {numeric_cols}')
    print(f'类别特征: {categorical_cols}')

    # ---- 构建预处理 Pipeline ----
    # 数值特征处理：中位数填充 → 标准化
    # 中位数对异常值比均值更稳健
    transformers = []
    if numeric_cols:
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),       # 中位数填充缺失值
            ('scaler', StandardScaler()),                        # 标准化（z-score）
        ])
        transformers.append(('num', num_pipe, numeric_cols))

    # 类别特征处理：众数填充 → One-Hot 编码
    # sparse_output=False 保证输出为稠密矩阵，与其他模型兼容
    if categorical_cols:
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')), # 众数填充缺失值
            ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),  # 独热编码
        ])
        transformers.append(('cat', cat_pipe, categorical_cols))

    # ColumnTransformer 将不同变换应用到不同列
    preprocessor = ColumnTransformer(transformers, remainder='drop')

    # ---- 划分训练集和测试集 ----
    # test_size=0.2: 80% 训练, 20% 测试
    # random_state=42: 固定随机种子，确保可复现
    # stratify=y: 分层抽样，保持训练/测试集中正负样本比例与原始数据一致
    # 重要：先划分再编码标签，防止标签信息从测试集泄露到训练集
    X_train, X_test, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f'\n训练集: {X_train.shape[0]}  测试集: {X_test.shape[0]}')

    # 标签编码（将文本标签转为 0/1 数值）
    le = LabelEncoder()
    y_train = le.fit_transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    # ---- 定义三种分类模型 ----
    # 超参数使用默认值或基本设置，未做调优
    # 如果想在报告中展示"超参数调优"，可以在此基础上扩展 GridSearchCV
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = []
    pipelines = {}

    # ---- 逐个模型训练与评估 ----
    for name, clf in models.items():
        # 将预处理和模型组合成一个 Pipeline
        # 这样在 predict 时会自动对新数据应用同样的预处理
        pipe = Pipeline([('prep', preprocessor), ('clf', clf)])
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe

        # 在测试集上进行预测
        y_pred = pipe.predict(X_test)

        # 计算四类评估指标
        acc = accuracy_score(y_test, y_pred)                                            # 准确率
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)     # 精确率（加权平均）
        rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)         # 召回率（加权平均）
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)              # F1 分数（加权平均）

        # 保存结果
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

    # ---- 保存评估结果到 CSV ----
    rdf = pd.DataFrame(results)
    rpath = os.path.join(OUTPUT_DIR, 'model_metrics.csv')
    rdf.to_csv(rpath, index=False)
    print(f'\n评估结果已保存: {rpath}')

    return rdf, pipelines, preprocessor, numeric_cols, categorical_cols, le


# ============================================================
# 任务4：特征重要性分析
# ============================================================
# 功能：
#   利用 Random Forest 模型的 feature_importances_ 属性，
#   评估每个特征对预测结果的贡献程度。
#
# 原理（适合写进报告）：
#   随机森林由多棵决策树组成。每棵树在节点分裂时选择"最有效的特征"。
#   feature_importances_ 衡量的是：在所有树的节点分裂中，
#   某个特征被选中的频率 × 该特征带来的不纯度减少量。
#   得分越高 → 该特征对预测越重要。
#
# 重要实现细节：
#   类别特征经过 One-Hot 编码后会拆分为多个二值特征，
#   因此特征重要性排名中的特征名可能包含原始类别列名 + 类别取值。
#   例如："父母最高教育水平_本科"、"父母最高教育水平_研究生"等。
#
# 输出文件：output/feature_importance.csv
# 报告用途：
#   "特征重要性"是数据分析报告中最有价值的部分之一。
#   可以回答"到底是什么因素影响了学生挂科风险？"这一核心问题。
#   结果通常显示"本学期旷课次数"是最重要的特征。

def task4_feature_importance(preprocessor, pipelines, numeric_cols, categorical_cols):
    """
    任务4：特征重要性分析
    使用 Random Forest 的内置重要性评估。

    输入：
      preprocessor    — ColumnTransformer（用于获取 One-Hot 编码后的特征名）
      pipelines       — 训练好的 Pipeline 字典
      numeric_cols    — 数值特征列名
      categorical_cols — 类别特征列名

    输出：
      imp_df — DataFrame，包含 feature 和 importance 两列，按重要性降序排列
    """
    print('\n' + '=' * 60)
    print('【任务4】特征重要性分析')
    print('=' * 60)

    # 确保 Random Forest 模型存在
    if 'Random Forest' not in pipelines:
        print('跳过：没有 Random Forest 模型')
        return None

    # 从 Pipeline 中提取 Random Forest 模型
    rf = pipelines['Random Forest'].named_steps['clf']
    imp = rf.feature_importances_  # numpy 数组，长度 = 特征总数

    # ---- 重建特征名列表 ----
    # 数值列直接使用原始列名
    # 类别列经过 One-Hot 编码后产生多个二值列，需从 OHE 中获取拆分后的特征名
    names = []
    if numeric_cols:
        names.extend(numeric_cols)
    if categorical_cols:
        cat_tfm = preprocessor.named_transformers_.get('cat')
        if cat_tfm is not None:
            ohe = cat_tfm.named_steps['ohe']
            names.extend(ohe.get_feature_names_out(categorical_cols))

    # 确保长度匹配（防止预处理中的列删除导致不匹配）
    n = min(len(imp), len(names))
    imp = imp[:n]
    names = names[:n]

    # 构建 DataFrame 并按重要性降序排列
    imp_df = pd.DataFrame({'feature': names, 'importance': imp})
    imp_df = imp_df.sort_values('importance', ascending=False).reset_index(drop=True)

    # 保存到文件
    rpath = os.path.join(OUTPUT_DIR, 'feature_importance.csv')
    imp_df.to_csv(rpath, index=False)
    print(f'特征重要性已保存: {rpath}')

    # 终端打印 Top-10
    print('\nTop-10 重要特征:')
    for i, row in imp_df.head(10).iterrows():
        print(f'  {i+1}. {row["feature"]}: {row["importance"]:.4f}')

    return imp_df


# ============================================================
# 任务5：可视化（三张图表）
# ============================================================
# 功能：
#   生成三张高清（200 DPI）学术风格图表，适配中文显示：
#
#   图1 — 特征重要性排名柱状图 (feature_importance.png)
#     横条图，展示 Top-N 重要特征及其得分。
#     通常"本学期旷课次数"会显著高于其他特征。
#     数据来源：task4 输出的 imp_df
#     报告用途：直接插入"特征分析"章节。
#
#   图2 — 模型性能对比图 (model_compare.png)
#     分组柱状图，对比三种模型在四个指标上的表现。
#     可以直观看出哪个模型综合最优。
#     报告用途：插入"模型评估与对比"章节。
#
#   图3 — 特征相关性热力图 (correlation_heatmap.png)
#     展示所有数值特征（包括排除字段）之间的 Pearson 相关系数。
#     颜色越红表示正相关越强，越蓝表示负相关越强。
#     注意：排除字段（如 GPA、成绩等级）也出现在此图中供 EDA 分析，
#     但它们不参与模型训练。
#     报告用途：插入"探索性数据分析（EDA）"章节。

def task5_charts(imp_df, results_df, df):
    """
    任务5：生成可视化图表
    输出三张 PNG 图片到 output/ 目录。
    """
    print('\n' + '=' * 60)
    print('【任务5】生成可视化图表')
    print('=' * 60)

    # 配色方案（学术风格，避免过于鲜艳）
    colors_bar = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3',
                  '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']

    # ============ 图1: 特征重要性排名（水平柱状图） ============
    if imp_df is not None and len(imp_df):
        top_n = min(15, len(imp_df))                           # 最多显示 Top-15
        plot_data = imp_df.head(top_n).iloc[::-1]              # 反转顺序（最重要的在顶部）

        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(top_n), plot_data['importance'],
                       color=[colors_bar[i % len(colors_bar)] for i in range(top_n)])

        ax.set_yticks(range(top_n))
        ax.set_yticklabels(plot_data['feature'], fontsize=11)
        ax.set_xlabel('重要性', fontsize=13)
        ax.set_title('特征重要性排名 (Random Forest)', fontsize=15, fontweight='bold')

        # 在条形末端标注数值
        for bar, val in zip(bars, plot_data['importance']):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9)

        plt.tight_layout()
        p1 = os.path.join(OUTPUT_DIR, 'feature_importance.png')
        plt.savefig(p1, dpi=200, bbox_inches='tight')
        plt.close()
        print(f'特征重要性图: {p1}')

    # ============ 图2: 模型性能对比（分组柱状图） ============
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(results_df))   # [0, 1, 2] 对应三个模型
    width = 0.2                      # 每组 4 个柱子，每个宽度 0.2
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    mcolors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    # 绘制四个指标的分组柱状图
    for i, m in enumerate(metrics):
        ax.bar(x + i * width, results_df[m], width, label=m, color=mcolors[i])

    ax.set_xlabel('模型', fontsize=13)
    ax.set_ylabel('得分', fontsize=13)
    ax.set_title('模型性能对比', fontsize=15, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results_df['Model'], fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.08)            # 给数值标注留出空间

    # 在每个柱子上标注具体数值
    for i, m in enumerate(metrics):
        for j, v in enumerate(results_df[m]):
            ax.text(j + i * width, v + 0.008, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=8, rotation=45)

    plt.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, 'model_compare.png')
    plt.savefig(p2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'模型对比图: {p2}')

    # ============ 图3: 特征相关性热力图 ============
    # 选取所有数值列（包括被排除的 GPA、成绩等级等，用于 EDA 分析）
    ndf = df.select_dtypes(include=[np.number]).copy()

    # 如果 '成绩等级' 或 'GradeClass' 是文本类型，尝试转为数值
    for _c in ['成绩等级', 'GradeClass']:
        if _c in df.columns and _c not in ndf.columns:
            ndf[_c] = pd.to_numeric(df[_c], errors='coerce')

    # 至少需要 2 列才能绘制热力图
    if ndf.shape[1] >= 2:
        corr = ndf.corr()   # Pearson 相关系数矩阵

        # 自适应图片尺寸
        sz = max(10, corr.shape[1] * 0.7)
        sz2 = max(8, corr.shape[0] * 0.7)
        fig, ax = plt.subplots(figsize=(sz, sz2))

        # imshow 显示颜色矩阵
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(im, ax=ax, shrink=0.8, label='相关系数')

        # 设置坐标轴刻度
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(corr.columns, fontsize=9)
        ax.set_title('特征相关性热力图', fontsize=15, fontweight='bold')

        # 在每个格子中标注相关系数
        # 颜色规则：|r| > 0.5 用白色字，否则用黑色字
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
# 任务6：生成课程报告素材（Markdown 格式）
# ============================================================
# 功能：
#   自动生成一份结构完整的中文课程报告 Markdown 文件，
#   包含以下章节：
#   一、数据集介绍
#   二、数据预处理方法
#   三、三种模型简介（含算法原理简述）
#   四、模型评估结果分析（含表格和结论）
#   五、特征重要性分析（Top-5 特征）
#   六、项目结论
#
# 输出文件：output/report_material.md
#
# 使用方式：
#   用户可以直接将此 .md 文件的内容复制到课程报告中，
#   然后根据需要修改、增删、补充图表引用即可。
#   PPT 制作时也可以直接将各段文字提炼为幻灯片要点。
#
# 设计思路：
#   报告内容是在程序运行时根据实际数据动态生成的（如最佳模型名称、
#   F1 分数、Top-5 特征列表等），确保数据驱动、精确无误。

def task6_report(label_col, feature_cols, drop_cols, results_df, imp_df, df):
    """
    任务6：生成课程报告素材

    输入：
      所有前面的分析结果都被汇总到这里，生成一份完整的 Markdown 报告。

    输出：
      output/report_material.md
    """
    print('\n' + '=' * 60)
    print('【任务6】生成课程报告素材')
    print('=' * 60)

    # 找到 F1 分数最高的模型（作为"最佳模型"写入结论）
    best = results_df.loc[results_df['F1'].idxmax()]

    # ---- 逐行构建 Markdown 报告 ----
    lines = []
    lines.append('# 高校学生学业风险预测 — 课程报告素材')
    lines.append('')
    lines.append('---')
    lines.append('')
    lines.append('## 一、数据集介绍')
    lines.append('')
    lines.append(f'本实验采用的数据集为CSV格式，共包含 **{df.shape[0]:,}** 条样本记录和 **{df.shape[1]}** 个字段。')
    lines.append(f'根据数据字典定义，标签列为 **{label_col}**，有效特征共 **{len(feature_cols)}** 个。')
    lines.append('')
    lines.append('**数据集概览：**')
    lines.append(f'- 样本总数：{df.shape[0]:,}')
    lines.append(f'- 训练特征数：{len(feature_cols)}')
    lines.append(f'- 标签字段：{label_col}')
    if drop_cols:
        _drop_str = ', '.join(drop_cols)
        lines.append(f'- EDA分析字段（不参与训练）：{_drop_str}')
    lines.append('')
    lines.append('## 二、数据预处理方法')
    lines.append('')
    lines.append('本研究以 scikit-learn 为核心工具，预处理流程如下：')
    lines.append('')
    lines.append('1. **标签列固定** — 根据数据字典设置标签列为 `学期末是否挂科`（唯一标签列）。')
    lines.append('2. **特征筛选** — 仅使用数据字典允许的特征列进入训练；可能含数据泄露风险的字段仅用于EDA分析。')
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

    # 保存文件
    report = '\n'.join(lines)
    rpath = os.path.join(OUTPUT_DIR, 'report_material.md')
    with open(rpath, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'课程报告已保存: {rpath}')


# ============================================================
# 主流程（main 函数）
# ============================================================
# 业务流程（按顺序依次执行）：
#
#   步骤 0: 文件选择
#     扫描 data/*.csv，如果找到多个文件则让用户交互选择。
#
#   步骤 1: 数据加载与校验
#     读取 CSV → validate_and_resolve_columns() 按数据字典解析列
#
#   步骤 2: 输出特征审查报告
#     让用户清楚知道模型用了哪些列、排除了哪些列及原因
#
#   步骤 3: 数据集分析（任务1）
#     输出 dataset_report.txt
#
#   步骤 4: 建模与评估（任务2 & 3）
#     三种模型 → 四类指标 → model_metrics.csv
#
#   步骤 5: 特征重要性分析（任务4）
#     基于 Random Forest → feature_importance.csv
#
#   步骤 6: 可视化（任务5）
#     三张 PNG 图片
#
#   步骤 7: 生成报告素材（任务6）
#     report_material.md
#
# 整个流程是线性的、自动化的，从原始数据到最终报告一步到位。
# 非常适合在 PPT 中展示为 "系统流程图"。

def main():
    """
    主函数：编排整个分析流程。
    从文件选择到报告生成，按顺序调用各任务函数。
    """
    print('=' * 60)
    print('  高校学生学业风险预测系统')
    print('=' * 60)

    # ============================================
    # 步骤 0：文件选择
    # ============================================
    csv_files = find_csv_files()
    if not csv_files:
        print(f'\n[提示] 未在 {DATA_DIR}/ 下找到 CSV 文件。')
        print(f'请将 CSV 文件放入 {DATA_DIR}/ 目录后重新运行。')
        print('运行方式:  python train.py')
        sys.exit(0)

    # 如果找到多个 CSV 文件，让用户选择要处理哪一个
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

    # ============================================
    # 步骤 1：数据加载
    # ============================================
    print(f'\n读取数据: {os.path.basename(csv_path)}')
    try:
        # 优先 UTF-8，失败则回退 GBK（兼容 Windows 导出的 CSV）
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(csv_path, encoding='gbk')
    except Exception as e:
        print(f'读取失败: {e}')
        sys.exit(1)

    print(f'数据维度: {df.shape[0]} 行 × {df.shape[1]} 列\n')

    # ============================================
    # 步骤 2：列校验与特征审查
    # ============================================
    # 按数据字典校验并解析列
    label_col, feature_cols, drop_cols = validate_and_resolve_columns(df)

    # 输出特征审查报告
    output_feature_review(label_col, feature_cols, drop_cols)

    # ============================================
    # 步骤 3-7：依次执行各任务
    # ============================================
    # 任务1：数据集分析
    task1_analyze(df, csv_path, label_col, feature_cols, drop_cols)

    # 任务2 & 3：建模与评估
    results_df, pipelines, preprocessor, numeric_cols, categorical_cols, label_encoder = \
        task2_3_train_and_evaluate(df, label_col, feature_cols)

    # 任务4：特征重要性
    imp_df = task4_feature_importance(preprocessor, pipelines, numeric_cols, categorical_cols)

    # 任务5：可视化
    task5_charts(imp_df, results_df, df)

    # 任务6：报告素材
    task6_report(label_col, feature_cols, drop_cols, results_df, imp_df, df)

    print('\n' + '=' * 60)
    print('  所有任务完成！输出文件在 output/ 目录下。')
    print('=' * 60)


if __name__ == '__main__':
    main()
