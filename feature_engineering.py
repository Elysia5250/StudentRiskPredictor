"""
特征工程说明
============

当前项目直接使用预处理好的特征宽表 OULAD_Final_Feature_Matrix.csv，
不再从七张原始 OULAD 表在线合并。

时间窗口说明
------------
宽表中所有以 _120 结尾的特征（如 click_day_0_30_120、click_activity_forum_120 等）
均表示课程开始后前 120 天内的聚合数据。

这些特征已经过严格的时间截断处理，用于早期预警场景，
避免使用学期末信息造成数据泄露（Data Leakage）。

特征构成
--------
1. 人口学特征：gender, region, highest_education, imd_band, age_band, disability
2. 注册信息：date_registration, has_unregistered, unregistered_before_cutoff
3. 作业特征：num_assessments, avg_score, std_score, missed_submissions 等
4. VLE 点击行为：按时间窗口拆分的点击量、按活动类型拆分的点击量
5. 课程信息：module_presentation_length

标签
----
academic_risk: 0 = 学业正常 (Pass/Distinction), 1 = 学业危机 (Fail/Withdrawn)

扩展特征
--------
如需添加新特征，在宽表生成阶段（独立脚本）增加列即可，
本模块不再承担在线特征聚合职责。
"""

import pandas as pd
import numpy as np


def get_click_activity_cols(df):
    """返回表中所有以 click_activity_ 开头的列名"""
    return [c for c in df.columns if c.startswith('click_activity_')]
