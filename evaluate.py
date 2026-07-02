import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    auc, ConfusionMatrixDisplay
)
from config import BASE_DIR, OUTPUT_DIR, FIGURES_DIR, CLICK_TIME_WINDOW_COLS, MODEL_NAME_ZH

# ==================== 统一图表风格 ====================
CHART_DPI = 300
CHART_FONTSIZE_TITLE = 14
CHART_FONTSIZE_LABEL = 12
CHART_FONTSIZE_TICK = 9
CHART_FONTSIZE_LEGEND = 10
CHART_COLORS = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']

# ==================== 加载字段说明 ====================
_FIELD_DOC_PATH = os.path.join(BASE_DIR, 'OULAD_字段说明.csv')
_COLUMN_ZH = {}
if os.path.exists(_FIELD_DOC_PATH):
    doc_df = pd.read_csv(_FIELD_DOC_PATH)
    for _, row in doc_df.iterrows():
        _COLUMN_ZH[str(row.iloc[0]).strip()] = str(row.iloc[1]).strip()


def col_zh(name, fallback=True):
    """字段名 → 中文说明，查不到时返回原字段名或精简名"""
    if name in _COLUMN_ZH:
        return _COLUMN_ZH[name]
    if fallback:
        return name
    short = name.replace('_120', '').replace('click_activity_', '').replace('click_day_', '')
    return short


def short_time_window(name):
    """click_day_0_30_120 → 第0-30天"""
    return name.replace('click_day_', '第').replace('_120', '天')


def short_activity(name):
    """click_activity_forumng_120 → forumng（英文字段名）"""
    return name.replace('click_activity_', '').replace('_120', '')


def short_activity_zh(name):
    """
    click_activity_forumng_120 → 论坛（中文名）
    优先使用字段说明中的中文名，并去除公共前缀。
    """
    full = col_zh(name, fallback=False)
    if full and '：' in full:
        return full.split('：')[-1]
    return short_activity(name)

# ==================== 中文字体配置 ====================
_ZH_FONTS = [
    'Microsoft YaHei',
    'SimHei',
    'DengXian',
    'Noto Sans SC',
    'PingFang SC',
    'Arial Unicode MS',
]
for _f in _ZH_FONTS:
    try:
        plt.rcParams['font.sans-serif'] = [_f]
        plt.rcParams['axes.unicode_minus'] = False
        break
    except Exception:
        continue

# 清除字体缓存确保新配置生效
import matplotlib.font_manager as fm
fm._load_fontmanager(try_read_cache=False)

METRIC_ZH = {
    'Accuracy': '准确率',
    'Precision': '精确率',
    'Recall': '召回率',
    'F1': 'F1分数',
    'ROC_AUC': 'ROC_AUC',
}


# ==================== 1. 标签分布图 ====================
def plot_label_distribution(y):
    fig, ax = plt.subplots(figsize=(6, 5))
    counts = y.value_counts().sort_index()
    labels = ['学业正常 (0)', '学业危机 (1)']
    colors = ['#55A868', '#C44E52']
    bars = ax.bar(labels, counts.values, color=colors, width=0.5, edgecolor='white')
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f'{v}\n({v / len(y) * 100:.1f}%)', ha='center', va='bottom', fontsize=11)
    ax.set_ylabel('学生人数', fontsize=12)
    ax.set_title('标签分布 (academic_risk)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, counts.max() * 1.2)
    ax.grid(axis='y', alpha=0.3)
    path = os.path.join(FIGURES_DIR, 'label_distribution.png')
    plt.tight_layout()
    plt.savefig(path, dpi=CHART_DPI, bbox_inches='tight')
    plt.close()
    print(f'  标签分布图 -> {path}')


# ==================== 2. 点击时间窗口分布图 ====================
def plot_click_time_window(df):
    """对 academic_risk=0/1 分别求 click_day_* 列的均值，画分组柱状图"""
    cols = [c for c in CLICK_TIME_WINDOW_COLS if c in df.columns]
    if not cols or 'academic_risk' not in df.columns:
        print('  跳过点击时间窗口图: 缺少必要字段')
        return
    grouped = df.groupby('academic_risk')[cols].mean()
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(cols))
    width = 0.3
    colors = ['#55A868', '#C44E52']
    labels_map = {0: '学业正常', 1: '学业危机'}
    for risk in [0, 1]:
        if risk in grouped.index:
            vals = grouped.loc[risk].values
            ax.bar(x + risk * width, vals, width, label=labels_map[risk],
                   color=colors[risk])
    short_names = [short_time_window(c) for c in cols]
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(short_names, fontsize=10)
    ax.set_xlabel('时间窗口', fontsize=12)
    ax.set_ylabel('平均点击量', fontsize=12)
    ax.set_title('不同时间窗口点击量对比（前120天）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    path = os.path.join(FIGURES_DIR, 'click_time_window_distribution.png')
    plt.tight_layout()
    plt.savefig(path, dpi=CHART_DPI, bbox_inches='tight')
    plt.close()
    print(f'  点击时间窗口图 -> {path}')


# ==================== 3. 活动类型点击柱状图 ====================
def plot_activity_type_bar(df):
    """取 click_activity_* 列，按 academic_risk 分组画均值柱状图"""
    cols = [c for c in df.columns if c.startswith('click_activity_')]
    if not cols or 'academic_risk' not in df.columns:
        print('  跳过活动类型柱状图: 缺少必要字段')
        return
    grouped = df.groupby('academic_risk')[cols].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cols))
    width = 0.3
    colors = ['#55A868', '#C44E52']
    labels_map = {0: '学业正常', 1: '学业危机'}
    for risk in [0, 1]:
        if risk in grouped.index:
            vals = grouped.loc[risk].values
            ax.bar(x + risk * width, vals, width, label=labels_map[risk],
                   color=colors[risk])
    short_names = [short_activity_zh(c) for c in cols]
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(short_names, fontsize=8, rotation=45, ha='right')
    ax.set_xlabel('活动类型', fontsize=12)
    ax.set_ylabel('平均点击量', fontsize=12)
    ax.set_title('不同活动类型点击量对比（前120天）', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    path = os.path.join(FIGURES_DIR, 'activity_type_click_bar.png')
    plt.tight_layout()
    plt.savefig(path, dpi=CHART_DPI, bbox_inches='tight')
    plt.close()
    print(f'  活动类型柱状图 -> {path}')


# ==================== 4. 活动类型雷达图 ====================
def plot_activity_type_radar(df):
    cols = [c for c in df.columns if c.startswith('click_activity_')]
    if not cols or 'academic_risk' not in df.columns:
        print('  跳过活动类型雷达图: 缺少必要字段')
        return
    grouped = df.groupby('academic_risk')[cols].mean()
    categories = [short_activity_zh(c) for c in cols]
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = ['#55A868', '#C44E52']
    labels_map = {0: '学业正常', 1: '学业危机'}
    for risk in [0, 1]:
        if risk in grouped.index:
            vals = grouped.loc[risk].values.tolist()
            vals += vals[:1]
            ax.plot(angles, vals, 'o-', linewidth=2, label=labels_map[risk],
                    color=colors[risk])
            ax.fill(angles, vals, alpha=0.1, color=colors[risk])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_title('活动类型点击量雷达图（前120天）', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=11)
    path = os.path.join(FIGURES_DIR, 'activity_type_radar.png')
    plt.tight_layout()
    plt.savefig(path, dpi=CHART_DPI, bbox_inches='tight')
    plt.close()
    print(f'  活动类型雷达图 -> {path}')


# ==================== 5. ROC 曲线 ====================
def plot_roc_curves(models_results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
    for idx, (name, res) in enumerate(models_results.items()):
        y_proba = res['y_proba']
        pos_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        fpr, tpr, _ = roc_curve(y_test, pos_proba)
        roc_auc = auc(fpr, tpr)
        display_name = MODEL_NAME_ZH.get(name, name)
        ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2,
                label=f'{display_name} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正率', fontsize=12)
    ax.set_ylabel('真正率', fontsize=12)
    ax.set_title('ROC 曲线', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    path = os.path.join(FIGURES_DIR, 'roc_curve.png')
    plt.tight_layout()
    plt.savefig(path, dpi=CHART_DPI, bbox_inches='tight')
    plt.close()
    print(f'  ROC 曲线 -> {path}')


# ==================== 6. PR 曲线 ====================
def plot_pr_curves(models_results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
    for idx, (name, res) in enumerate(models_results.items()):
        y_proba = res['y_proba']
        pos_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        precision, recall, _ = precision_recall_curve(y_test, pos_proba)
        pr_auc = auc(recall, precision)
        display_name = MODEL_NAME_ZH.get(name, name)
        ax.plot(recall, precision, color=colors[idx % len(colors)], lw=2,
                label=f'{display_name} (AUC = {pr_auc:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('召回率', fontsize=12)
    ax.set_ylabel('精确率', fontsize=12)
    ax.set_title('精确率-召回率曲线', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(alpha=0.3)
    path = os.path.join(FIGURES_DIR, 'pr_curve.png')
    plt.tight_layout()
    plt.savefig(path, dpi=CHART_DPI, bbox_inches='tight')
    plt.close()
    print(f'  PR 曲线 -> {path}')


# ==================== 7. 混淆矩阵热力图 ====================
def plot_confusion_matrix(models_results, y_test):
    n = len(models_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
    for idx, (name, res) in enumerate(models_results.items()):
        cm = confusion_matrix(y_test, res['y_pred'])
        display_name = MODEL_NAME_ZH.get(name, name)
        ax = axes[idx]
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f'{display_name}', fontsize=13, fontweight='bold')
        tick_marks = [0, 1]
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['正常', '危机'], fontsize=10)
        ax.set_yticklabels(['正常', '危机'], fontsize=10)
        ax.set_xlabel('预测标签', fontsize=11)
        ax.set_ylabel('真实标签', fontsize=11)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        fontsize=14, color='white' if cm[i, j] > cm.max() / 2 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)
    path = os.path.join(FIGURES_DIR, 'confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(path, dpi=CHART_DPI, bbox_inches='tight')
    plt.close()
    print(f'  混淆矩阵 -> {path}')


# ==================== 8. 特征重要性（仅树模型） ====================
def plot_feature_importance(models_results, feature_cols, top_n=20):
    for name, res in models_results.items():
        model = res['model']
        if not hasattr(model, 'feature_importances_'):
            continue
        importances = model.feature_importances_
        n = min(len(importances), len(feature_cols))
        indices = np.argsort(importances[:n])[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.35)))
        y_pos = range(len(indices))
        ax.barh(y_pos, importances[indices][::-1], color='#4C72B0')
        ax.set_yticks(range(len(indices)))
        labels = [col_zh(feature_cols[i]) for i in indices][::-1]
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel('重要性', fontsize=12)
        display_name = MODEL_NAME_ZH.get(name, name)
        ax.set_title(f'特征重要性 - {display_name}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        path = os.path.join(FIGURES_DIR, f'feature_importance_{name}.png')
        plt.tight_layout()
        plt.savefig(path, dpi=CHART_DPI, bbox_inches='tight')
        plt.close()
        print(f'  特征重要性 ({name}) -> {path}')
        imp_df = pd.DataFrame({
            'feature': [col_zh(feature_cols[i]) for i in indices],
            'importance': importances[indices],
        })
        imp_path = os.path.join(OUTPUT_DIR, f'feature_importance_{name}.csv')
        imp_df.to_csv(imp_path, index=False, encoding='utf-8-sig')
        print(f'    CSV -> {imp_path}')


# ==================== 9. 模型指标对比图 ====================
def plot_model_comparison(metrics_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(metrics_df))
    width = 0.18
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC_AUC']
    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3']
    for i, m in enumerate(metrics):
        if m in metrics_df.columns:
            ax.bar(x + i * width, metrics_df[m], width,
                   label=METRIC_ZH.get(m, m), color=colors[i % len(colors)])
    model_names = [MODEL_NAME_ZH.get(n, n) for n in metrics_df['Model']]
    ax.set_xlabel('模型', fontsize=12)
    ax.set_ylabel('得分', fontsize=12)
    ax.set_title('模型性能对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(model_names, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.grid(alpha=0.3, axis='y')
    for i, m in enumerate(metrics):
        if m in metrics_df.columns:
            for j, v in enumerate(metrics_df[m]):
                ax.text(j + i * width, v + 0.01, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=7, rotation=45)
    path = os.path.join(FIGURES_DIR, 'model_metrics_comparison.png')
    plt.tight_layout()
    plt.savefig(path, dpi=CHART_DPI, bbox_inches='tight')
    plt.close()
    print(f'  模型对比图 -> {path}')


# ==================== 指标计算 ====================
def evaluate_all(models_results, y_test):
    rows = []
    for name, res in models_results.items():
        y_pred = res['y_pred']
        y_proba = res['y_proba']
        pos_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, pos_proba)
        rows.append({
            'Model': name,
            'Accuracy': round(acc, 4),
            'Precision': round(prec, 4),
            'Recall': round(rec, 4),
            'F1': round(f1, 4),
            'ROC_AUC': round(roc_auc, 4),
        })
        display_name = MODEL_NAME_ZH.get(name, name)
        print(f'\n  {display_name}')
        print(f'    准确率 (Accuracy):   {acc:.4f}')
        print(f'    精确率 (Precision):  {prec:.4f}')
        print(f'    召回率 (Recall):     {rec:.4f}')
        print(f'    F1 分数:            {f1:.4f}')
        print(f'    ROC AUC:            {roc_auc:.4f}')
    metrics_df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, 'metrics.csv')
    metrics_df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'\n  指标已保存 -> {path}')
    return metrics_df


# ==================== 保存预测结果 ====================
def save_predictions(models_results, X_test, y_test):
    rows = []
    for name, res in models_results.items():
        y_proba = res['y_proba']
        pos_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        rows.append(pd.DataFrame({
            'Model': name,
            'y_true': y_test.values,
            'y_pred': res['y_pred'],
            'y_proba': pos_proba,
        }))
    pred_df = pd.concat(rows, ignore_index=True)
    path = os.path.join(OUTPUT_DIR, 'prediction.csv')
    pred_df.to_csv(path, index=False, encoding='utf-8-sig')
    print(f'  预测结果 -> {path}')


# ==================== 10. 最佳模型混淆矩阵（含指标） ====================
def plot_confusion_matrix_best(models_results, y_test, X_test):
    """绘制最佳模型的混淆矩阵，同时显示 Accuracy/Precision/Recall/F1"""
    best_name, best_res = _find_best_model(models_results, y_test)
    if best_name is None:
        print('  跳过最佳模型混淆矩阵')
        return

    display_name = MODEL_NAME_ZH.get(best_name, best_name)
    cm = confusion_matrix(y_test, best_res['y_pred'])
    y_proba = best_res['y_proba']
    pos_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
    acc = accuracy_score(y_test, best_res['y_pred'])
    prec = precision_score(y_test, best_res['y_pred'], zero_division=0)
    rec = recall_score(y_test, best_res['y_pred'], zero_division=0)
    f1 = f1_score(y_test, best_res['y_pred'], zero_division=0)
    roc_auc = roc_auc_score(y_test, pos_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['正常', '危机'], fontsize=CHART_FONTSIZE_TICK + 1)
    ax.set_yticklabels(['正常', '危机'], fontsize=CHART_FONTSIZE_TICK + 1)
    ax.set_xlabel('预测标签', fontsize=CHART_FONTSIZE_LABEL)
    ax.set_ylabel('真实标签', fontsize=CHART_FONTSIZE_LABEL)
    ax.set_title(f'混淆矩阵 - {display_name}（最佳模型）', fontsize=CHART_FONTSIZE_TITLE, fontweight='bold')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    fontsize=16, color='white' if cm[i, j] > cm.max() / 2 else 'black')

    # 右侧显示指标信息
    metrics_text = (
        f'准确率 (Accuracy):  {acc:.4f}\n'
        f'精确率 (Precision): {prec:.4f}\n'
        f'召回率 (Recall):    {rec:.4f}\n'
        f'F1 分数:           {f1:.4f}\n'
        f'ROC AUC:           {roc_auc:.4f}'
    )
    ax.text(1.35, 0.5, metrics_text, transform=ax.transAxes,
            fontsize=11, va='center', ha='left',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.subplots_adjust(right=0.7)
    path = os.path.join(FIGURES_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=CHART_DPI, bbox_inches='tight')
    plt.close()
    print(f'  最佳模型混淆矩阵 -> {path}')


def _find_best_model(models_results, y_test):
    """根据 ROC AUC 返回最佳模型"""
    best_name, best_auc = None, -1
    for name, res in models_results.items():
        y_proba = res['y_proba']
        pos_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        try:
            auc_val = roc_auc_score(y_test, pos_proba)
            if auc_val > best_auc:
                best_auc, best_name = auc_val, name
        except Exception:
            continue
    if best_name:
        return best_name, models_results[best_name]
    return None, None


# ==================== 11. 合并特征重要性（最佳树模型 Top20） ====================
def plot_feature_importance_combined(models_results, feature_cols, top_n=20):
    """最佳树模型的 Top20 特征重要性，输出到 feature_importance.png"""
    for name, res in models_results.items():
        model = res['model']
        if not hasattr(model, 'feature_importances_'):
            continue
        display_name = MODEL_NAME_ZH.get(name, name)
        importances = model.feature_importances_
        n = min(len(importances), len(feature_cols))
        indices = np.argsort(importances[:n])[::-1][:top_n]
        fig, ax = plt.subplots(figsize=(11, max(6, top_n * 0.4)))
        y_pos = range(len(indices))
        ax.barh(y_pos, importances[indices][::-1], color='#4C72B0', height=0.7)
        ax.set_yticks(range(len(indices)))
        labels = [col_zh(feature_cols[i]) for i in indices][::-1]
        ax.set_yticklabels(labels, fontsize=CHART_FONTSIZE_TICK)
        ax.set_xlabel('重要性', fontsize=CHART_FONTSIZE_LABEL)
        ax.set_title(f'前{top_n}个重要特征 - {display_name}', fontsize=CHART_FONTSIZE_TITLE, fontweight='bold')
        ax.grid(alpha=0.3, axis='x')
        for i, (bar, val) in enumerate(zip(ax.patches, importances[indices][::-1])):
            ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=8)
        path = os.path.join(FIGURES_DIR, 'feature_importance.png')
        plt.tight_layout()
        plt.savefig(path, dpi=CHART_DPI, bbox_inches='tight')
        plt.close()
        print(f'  合并特征重要性 ({name}) -> {path}')
        break  # 只用第一个树模型


# ==================== 12. ROC / PR 对比图（所有模型在同一张图） ====================
def plot_roc_comparison(models_results, y_test):
    """所有模型 ROC 曲线对比图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (name, res) in enumerate(models_results.items()):
        y_proba = res['y_proba']
        pos_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        fpr, tpr, _ = roc_curve(y_test, pos_proba)
        roc_auc = auc(fpr, tpr)
        display_name = MODEL_NAME_ZH.get(name, name)
        ax.plot(fpr, tpr, color=CHART_COLORS[idx % len(CHART_COLORS)], lw=2.5,
                label=f'{display_name} (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正率', fontsize=CHART_FONTSIZE_LABEL)
    ax.set_ylabel('真正率', fontsize=CHART_FONTSIZE_LABEL)
    ax.set_title('ROC 曲线对比', fontsize=CHART_FONTSIZE_TITLE, fontweight='bold')
    ax.legend(loc='lower right', fontsize=CHART_FONTSIZE_LEGEND)
    ax.grid(alpha=0.3)
    path = os.path.join(FIGURES_DIR, 'roc_comparison.png')
    plt.tight_layout()
    plt.savefig(path, dpi=CHART_DPI, bbox_inches='tight')
    plt.close()
    print(f'  ROC 对比图 -> {path}')


def plot_pr_comparison(models_results, y_test):
    """所有模型 PR 曲线对比图"""
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, (name, res) in enumerate(models_results.items()):
        y_proba = res['y_proba']
        pos_proba = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba[:, 0]
        precision, recall, _ = precision_recall_curve(y_test, pos_proba)
        pr_auc = auc(recall, precision)
        display_name = MODEL_NAME_ZH.get(name, name)
        ax.plot(recall, precision, color=CHART_COLORS[idx % len(CHART_COLORS)], lw=2.5,
                label=f'{display_name} (AUC = {pr_auc:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('召回率', fontsize=CHART_FONTSIZE_LABEL)
    ax.set_ylabel('精确率', fontsize=CHART_FONTSIZE_LABEL)
    ax.set_title('精确率-召回率曲线对比', fontsize=CHART_FONTSIZE_TITLE, fontweight='bold')
    ax.legend(loc='lower left', fontsize=CHART_FONTSIZE_LEGEND)
    ax.grid(alpha=0.3)
    path = os.path.join(FIGURES_DIR, 'pr_comparison.png')
    plt.tight_layout()
    plt.savefig(path, dpi=CHART_DPI, bbox_inches='tight')
    plt.close()
    print(f'  PR 对比图 -> {path}')


# ==================== 评估入口 ====================
def run_evaluation(models_results, X_test, y_test, feature_cols, df_full=None):
    print('=' * 60)
    print('  评估阶段')
    print('=' * 60)

    metrics_df = evaluate_all(models_results, y_test)

    # EDA 图（使用全量数据）
    if df_full is not None:
        print('\n  生成探索性图表...')
        plot_label_distribution(df_full['academic_risk'] if 'academic_risk' in df_full.columns else y_test)
        plot_click_time_window(df_full)
        plot_activity_type_bar(df_full)
        plot_activity_type_radar(df_full)

    # 模型评估图
    print('\n  生成评估图表...')
    plot_roc_curves(models_results, y_test)                     # roc_curve.png
    plot_pr_curves(models_results, y_test)                      # pr_curve.png
    plot_confusion_matrix_best(models_results, y_test, X_test)  # confusion_matrix.png (最佳模型)
    plot_feature_importance(models_results, feature_cols)       # 各树模型特征重要性
    plot_model_comparison(metrics_df)                           # model_metrics_comparison.png

    print('\n  生成论文级图表...')
    plot_roc_comparison(models_results, y_test)                 # roc_comparison.png
    plot_pr_comparison(models_results, y_test)                  # pr_comparison.png
    plot_feature_importance_combined(models_results, feature_cols)  # feature_importance.png

    save_predictions(models_results, X_test, y_test)

    return metrics_df
