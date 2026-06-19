# 高校学生学业风险预测系统

基于机器学习算法预测学生挂科风险，分析影响学业的主要因素。

## 项目结构

```
project/
├── data/           # 存放 CSV 数据文件（用户自行放入）
├── output/         # 自动生成的输出文件
│   ├── dataset_report.txt      # 数据集分析报告
│   ├── model_metrics.csv       # 模型评估指标
│   ├── feature_importance.csv  # 特征重要性排序
│   ├── feature_importance.png  # 特征重要性柱状图
│   ├── model_compare.png       # 模型性能对比图
│   ├── correlation_heatmap.png # 相关性热力图
│   └── report_material.md      # 课程报告素材
├── train.py        # 主程序（一键运行）
├── requirements.txt
└── README.md
```

## 使用方式

1. 将 CSV 文件放入 `data/` 目录
2. 安装依赖：`pip install -r requirements.txt`
3. 运行程序：`python train.py`

## 功能说明

- **自动识别标签列**：支持多种列名（Grade、Result、挂科 等）
- **自动数据预处理**：缺失值填充、类别编码、标准化
- **三种分类模型**：逻辑回归、决策树、随机森林
- **完整评估体系**：Accuracy、Precision、Recall、F1
- **特征重要性分析**：基于 Random Forest
- **可视化输出**：高清 PNG 图表（适配中文）
- **课程报告素材**：Markdown 格式学术报告

## 依赖

- Python >= 3.8
- pandas, numpy, scikit-learn, matplotlib
# StudentRiskPredictor
# StudentRiskPredictor
# StudentRiskPredictor
