# 高校学生学业风险预测系统

基于机器学习算法预测学生挂科风险，分析影响学业的主要因素。

## 项目结构

```
project/
├── data/                    # 存放 CSV 数据文件（用户自行放入）
├── output/                  # 自动生成的输出文件
│   ├── dataset_report.txt          # 数据集分析报告
│   ├── feature_review.txt          # 特征审查报告
│   ├── model_metrics.csv           # 模型评估指标
│   ├── feature_importance.csv      # 特征重要性排序
│   ├── feature_importance.png      # 特征重要性柱状图
│   ├── model_compare.png           # 模型性能对比图
│   ├── correlation_heatmap.png     # 相关性热力图
│   ├── report_material.md          # 课程报告素材
│   └── models/                     # 训练好的模型文件
│       ├── best_model.pkl
│       ├── label_encoder.pkl
│       └── feature_columns.pkl
├── train.py                 # 模型训练（一键运行）
├── predict.py               # 批量预测（命令行）
├── web/                     # Web 应用（Flask）
│   ├── app.py
│   ├── static/style.css
│   ├── templates/
│   │   ├── base.html
│   │   ├── index.html
│   │   ├── train.html
│   │   └── predict.html
│   └── uploads/
├── requirements.txt
└── README.md
```

## 使用方式

### 1. 训练模型

```bash
pip install -r requirements.txt
python3 train.py
```

训练完成后，模型自动保存在 `output/models/`，评估结果和图表保存在 `output/`。

### 2. 批量预测（命令行）

```bash
python3 predict.py data/新数据.csv
```

输出 `data/新数据_predicted.csv`，在原数据上追加 `预测结果` 和 `挂科概率` 两列。

### 3. Web 应用（仪表盘）

```bash
cd web
python app.py
# 浏览器访问 http://localhost:5001
```

三个页面：
- **仪表盘**：数据集概览 + 上次训练结果
- **模型训练**：一键训练，同页展示指标表格、特征重要性、图表
- **风险预测**：上传 CSV → 预览 → 预测 → 下载结果

## 功能说明

- **数据字典校验**：严格管理标签列/特征列/排除列，防止数据泄露
- **自动数据预处理**：缺失值填充、One-Hot 编码、标准化
- **三种分类模型**：逻辑回归、决策树、随机森林
- **完整评估体系**：Accuracy、Precision、Recall、F1
- **模型持久化**：训练完成后自动保存最佳模型，供预测调用
- **特征重要性分析**：基于 Random Forest
- **可视化输出**：高清 PNG 图表（适配中文）
- **课程报告素材**：Markdown 格式学术报告
- **Web 仪表盘**：可视化操作界面，无需命令行

## 依赖

- Python >= 3.8
- pandas, numpy, scikit-learn, matplotlib, flask
