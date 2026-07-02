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
| 逻辑回归 | 86.95% | 92.02% | 82.42% | 0.8696 | 0.9426 |
| 随机森林 | 87.42% | 91.99% | 83.44% | 0.8751 | 0.9441 |
| LightGBM | 87.71% | 92.23% | 83.79% | 0.8781 | 0.9457 |
| **XGBoost** | **87.51%** | **92.41%** | **83.18%** | **0.8755** | **0.9461** |

**最佳模型**：XGBoost（最高 AUC 0.9461），LightGBM 紧随其后。

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
- `metrics.csv` — 各模型五类指标
- `prediction.csv` — 测试集预测详情
- `feature_importance_*.csv` — 各树模型特征重要性

**图表文件**（`outputs/figures/`，全部 DPI=300，中文标签）
- 评估类：ROC 曲线、PR 曲线、混淆矩阵、模型对比、特征重要性
- 探索类：标签分布、时间窗口点击、活动类型柱状图/雷达图
- 可解释类：SHAP Summary、SHAP Bar、SHAP Waterfall、SHAP Dependence
