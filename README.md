# 学生早期挂科预警系统

基于 **OULAD（Open University Learning Analytics Dataset）** 特征宽表，利用学期前期数据（前120天）预测学生最终学业风险（挂科/退学）。

**核心设计**：严格的时间截断机制防止数据泄露，所有特征仅使用课程开始后前120天的信息。

---

## 项目结构

```
project/
├── data/                          # 数据文件
│   └── OULAD_Final_Feature_Matrix.csv  # 特征宽表（约3.2万行 × 63列）
├── outputs/                       # 输出目录
│   ├── metrics.csv                # 各模型评估指标
│   ├── prediction.csv             # 测试集预测结果
│   ├── feature_importance_*.csv   # 特征重要性
│   ├── models/                    # 训练好的模型
│   │   ├── best_model.pkl
│   │   └── feature_columns.pkl
│   └── figures/                   # 所有图表
│       ├── roc_comparison.png     # ROC 曲线（所有模型）
│       ├── pr_comparison.png      # PR 曲线（所有模型）
│       ├── confusion_matrix.png   # 最佳模型混淆矩阵 + 指标
│       ├── feature_importance.png # Top20 特征重要性
│       ├── model_metrics_comparison.png  # 模型性能对比
│       ├── shap_summary.png       # SHAP 蜜蜂图
│       ├── shap_bar.png           # SHAP 特征重要性
│       ├── shap_waterfall.png     # SHAP 单样本瀑布图
│       ├── shap_dependence.png    # SHAP 依赖图
│       ├── label_distribution.png # 标签分布
│       ├── click_time_window_distribution.png  # 时间窗口点击对比
│       ├── activity_type_click_bar.png         # 活动类型柱状图
│       └── activity_type_radar.png             # 活动类型雷达图
├── config.py                     # 全局配置（路径、字段定义、训练参数）
├── preprocess.py                  # 数据加载与字段类型识别
├── feature_engineering.py         # 特征工程说明文档
├── models.py                      # 模型定义、预处理流水线、训练
├── evaluate.py                    # 评估指标、图表生成
├── interpretability.py            # SHAP 可解释性分析
├── train.py                       # 一键训练主入口
├── predict.py                     # 命令行批量预测
├── web/                           # Web 应用（Flask）
│   ├── app.py
│   ├── static/style.css
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── train.html
│   │   └── predict.html
│   └── uploads/
├── OULAD_字段说明.csv              # 字段中英文对照
├── requirements.txt
└── README.md
```

---

## 使用方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 准备数据

将 `OULAD_Final_Feature_Matrix.csv` 放入 `data/` 目录。

### 3. 训练模型

```bash
python train.py
```

一键完成：数据加载 → 字段识别 → 模型训练 → 评估 → 图表生成 → SHAP 分析。

### 4. 批量预测（命令行）

```bash
python predict.py data/新数据.csv
```

输出追加 `prediction`（0/1）和 `risk_probability` 两列。

### 5. Web 应用（仪表盘）

```bash
cd web
python app.py
# 浏览器访问 http://localhost:5001
```

三个页面：
- **仪表盘**：数据集概览 + 上次训练结果
- **模型训练**：一键训练，实时展示指标、特征重要性、混淆矩阵、ROC 曲线
- **风险预测**：上传 CSV → 预览 → 预测 → 下载结果

---

## 模型结果

| 模型 | 准确率 | 精确率 | 召回率 | F1 分数 | ROC AUC |
|------|:-----:|:-----:|:-----:|:-------:|:-------:|
| 逻辑回归 | 86.81% | 91.19% | 83.03% | 0.8692 | 0.9414 |
| 随机森林 | 87.33% | 89.90% | **85.62%** | 0.8771 | 0.9444 |
| LightGBM | 87.64% | 90.55% | 85.50% | 0.8796 | 0.9444 |
| XGBoost | 87.65% | 90.68% | 85.39% | 0.8795 | 0.9449 |
| **Ensemble（集成）** | **87.73%** | **90.70%** | **85.53%** | **0.8804** | **0.9450** |

> 注：所有指标基于最优截断阈值 0.477（通过验证集搜索得到）。5折交叉验证 AUC 标准差仅 ±0.004，结果稳定。

**最佳模型**：Ensemble（LightGBM + XGBoost 概率平均），AUC 0.9450，F1 0.8804。

---

## V2.0 高级优化详解

从基线版本到最终版本的性能提升，来源于以下 5 项关键优化。每一项都在 `models.py` 中有独立函数实现。

### 1. 特征筛选（Feature Selection）

**背景**：原始宽表包含 57 个特征。部分特征（如特定的活动类型点击量）与学业风险的关联很弱，反而引入噪声。

**方法**：先用 LightGBM 在训练集上训练一次，获取每个特征的重要性得分。按重要性降序排列后，保留累积重要性占前 80% 的特征，剔除尾部低贡献特征。

**效果**：57 个特征 → 28 个特征。模型更轻量、泛化性更好，且 AUC 不降反升（噪声被移除）。

### 2. 5 折交叉验证（Cross-Validation）

**背景**：单次 80/20 划分的结果可能受随机拆分影响，一次好不代表真的好。

**方法**：将训练集等分为 5 份，轮流用 4 份训练、1 份验证，重复 5 次。最终取 5 次 AUC 的均值 ± 标准差。

**效果**：各模型 AUC 标准差仅 ±0.004，说明结果非常稳定、不受划分方式影响。

### 3. 概率校准（Probability Calibration）

**背景**：树模型（LightGBM / XGBoost / RandomForest）输出的概率往往有系统性偏差——例如预测 0.7 的样本，真实正例比例可能只有 0.6。

**方法**：使用 `CalibratedClassifierCV(method='sigmoid')` 对树模型的概率输出进行 Platt Scaling，将原始得分映射到更接近真实概率的区间。

**效果**：校准后的概率更可信，在做阈值决策时更有依据。

### 4. 阈值优化（Threshold Tuning）

**背景**：默认的概率截断点 0.5 对预警系统不一定最优。漏报一个高危学生的代价远大于误报。

**方法**：在验证集上从 0.05 到 0.95 以 0.01 为步长搜索，以 **F1 分数**为目标找到最优截断点。

**效果**：最优阈值从默认 0.5 调整至 **0.477**。虽然精度略有下降，但召回率提升显著——模型能够捕捉到更多真正的高危学生。

### 5. 模型集成（Ensemble）

**背景**：不同模型的优势和盲区不同。LightGBM 和 XGBoost 虽然同属 GBDT 家族，但在特征利用上有细微差异。

**方法**：将 LightGBM 和 XGBoost 在测试集上的预测概率做简单算术平均，得到 Ensemble 模型的最终预测。

**效果**：Ensemble 的 AUC **0.9450** 和 F1 **0.8804** 均超过任意单一模型，且预测更鲁棒。

---

## 技术特性

### 数据防泄露
- 所有特征仅使用课程前 **120天** 数据（字段后缀 `_120`）
- 严格排除标签泄露特征：`is_still_registered`、`unregistered_by_120`、`date_unregistration`
- `final_result` 禁止入模，`id_student` 禁止入模

### 模型可解释性
- **SHAP** 自动分析最佳树模型（优先 LightGBM > XGBoost > RandomForest）
- 生成 Summary、Bar、Waterfall、Dependence 四种图表
- 缺失 shap 库时自动提示安装，不中断训练

### 预处理流水线
- 类别特征：缺失填 `"Unknown"` → OneHotEncoder
- 数值特征：中位数填充 → (可选) StandardScaler
- 提交日期类缺失填 `-1`（表示"从未提交"，保持信息完整性）

### 类别权重
- 逻辑回归 / 随机森林：`class_weight='balanced'`
- LightGBM / XGBoost：动态 `scale_pos_weight` 根据训练集正负比自动计算

### 高级优化
- **特征筛选**：LightGBM 训练后按累积重要性保留前 80% 特征，剔除噪声（57 → 28 个特征）
- **5折交叉验证**：替代单次划分，报告 mean ± std，评估模型稳定性
- **概率校准**：`CalibratedClassifierCV(sigmoid)` 校准树模型概率输出
- **阈值优化**：在验证集上搜索最优截断点（当前最优 0.477），提升召回率
- **模型集成**：LightGBM + XGBoost 概率平均，作为 Ensemble 模型参与评估

---

## 依赖

- Python >= 3.10
- pandas, numpy, scikit-learn, matplotlib, joblib
- lightgbm（可选，自动跳过）
- xgboost（可选，自动跳过）
- shap（可选，自动提示安装）
- flask（Web 应用）

---

## 输出清单

运行 `python train.py` 后生成：

**CSV 文件**（`outputs/`）
- `metrics.csv` — 各模型五类指标（基于最优阈值）
- `cv_results.csv` — 5折交叉验证结果
- `threshold_search.csv` — 阈值搜索过程
- `prediction.csv` — 测试集预测详情
- `feature_importance_*.csv` — 各树模型特征重要性

**图表文件**（`outputs/figures/`，全部 DPI=300，中文标签）
- 评估类：ROC 曲线、PR 曲线、混淆矩阵、模型对比、特征重要性
- 探索类：标签分布、时间窗口点击、活动类型柱状图/雷达图
- 可解释类：SHAP Summary、SHAP Bar、SHAP Waterfall、SHAP Dependence
