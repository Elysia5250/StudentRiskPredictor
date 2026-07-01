import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ==================== 特征宽表 ====================
# 数据已预处理完毕，直接从宽表读取
WIDE_TABLE = 'OULAD_Final_Feature_Matrix.csv'
FIELD_DOC = 'OULAD_字段说明.csv'

# ==================== 时间窗口说明 ====================
# 宽表中所有以 _120 结尾的特征表示课程开始后前 120 天内的聚合数据。
# 这些特征用于早期预警，避免使用期末信息造成数据泄露。
TIME_WINDOW_DAYS = 120

# ==================== 标签与字段配置 ====================
LABEL_COL = 'academic_risk'

# 禁止作为特征的列（标签 + 原始结果 + ID）
EXCLUDED_COLS = [
    'academic_risk',        # 标签
    'final_result',         # 原始结果，等价于标签
    'id_student',           # 唯一ID
    'is_still_registered',  # 是否未注销 → 已注销=必然退学，标签泄露
    'unregistered_by_120',  # 120天内退课 → 退课=必然挂科，标签泄露
    'date_unregistration',  # 注销日期 → 有值即退学，标签泄露
]

# 类别特征（主键 + 人口学）
CATEGORICAL_COLS = [
    'code_module',
    'code_presentation',
    'gender',
    'region',
    'highest_education',
    'imd_band',
    'age_band',
    'disability',
]

# ==================== 时间窗口相关列 ====================
# 用于图表展示的固定窗口字段
CLICK_TIME_WINDOW_COLS = [
    'click_day_0_30_120',
    'click_day_31_60_120',
    'click_day_61_90_120',
    'click_day_91_120_120',
]

# ==================== 训练参数 ====================
RANDOM_STATE = 42
TEST_SIZE = 0.2

# ==================== 模型名称中文映射 ====================
MODEL_NAME_ZH = {
    'LogisticRegression': '逻辑回归',
    'RandomForest': '随机森林',
    'LightGBM': 'LightGBM',
    'XGBoost': 'XGBoost',
}

# ==================== 可选模型列表 ====================
AVAILABLE_MODELS = [
    'LogisticRegression',
    'RandomForest',
    'LightGBM',
    'XGBoost',
]
