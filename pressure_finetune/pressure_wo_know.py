#%%
import torch
import torch.nn as nn
import os
import torch.optim as optim
import sys
import csv
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast
import time
sys.path.append('../model_construction')

    
        # for batch_id, batch in enumerate(test_data):
        #     x = batch['features'].to(device)
        #     y = batch['label'].to(device)
        #     y_out = combined_model(x)
        #     predicted_indices = torch.argmax(y_out, dim=1)
        #     true_indices = torch.argmax(y, dim=1)
        #     correct_predictions = (predicted_indices == true_indices)
        #     num_correct = num_correct + correct_predictions.sum().item()
        #     total_num = total_num + y.shape[0]
        # # print(total_num)
        # print(num_correct)
        # torch.cuda.empty_cache()
#%%
# 读取保存的预测结果和真实标签
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize

# 读取with knowledge的数据
all_predictions_with = np.load('../tmp7-5/results/pressure_test_predictions.npy')
all_true_labels_with = np.load('../tmp7-5/results/pressure_test_true_labels.npy')

# 读取without knowledge的数据
all_predictions_without = np.load('../tmp7-5/results/pressure_wo_know_test_predictions.npy')
all_true_labels_without = np.load('../tmp7-5/results/pressure_wo_know_test_true_labels.npy')

def calculate_roc_metrics(all_predictions, all_true_labels):
    """计算ROC曲线和AUC值的函数"""
    # 将三维数据在第一个维度上展开
    y_score = all_predictions.reshape(-1, all_predictions.shape[-1])
    y_test = all_true_labels.reshape(-1, all_true_labels.shape[-1])
    
    n_classes = y_score.shape[1]
    
    # 计算每一类的ROC
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算micro-average ROC曲线
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # 计算macro-average ROC曲线
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    return fpr, tpr, roc_auc, n_classes

# 计算两组数据的ROC指标
fpr_with, tpr_with, roc_auc_with, n_classes = calculate_roc_metrics(all_predictions_with, all_true_labels_with)
fpr_without, tpr_without, roc_auc_without, _ = calculate_roc_metrics(all_predictions_without, all_true_labels_without)
#%%
# 绘制对比ROC曲线
lw = 2
plt.figure(figsize=(12, 10))

# 定义颜色和线条样式
colors = ['aqua', 'darkorange', 'cornflowerblue', 'red']
line_styles_with = ['-', '-', '-', '-']  # with knowledge用实线
line_styles_without = ['--', '--', '--', '--']  # without knowledge用虚线

# 绘制每个类别的ROC曲线
for i in range(n_classes):
    # with knowledge (实线)
    plt.plot(fpr_with[i], tpr_with[i], color=colors[i], lw=lw,
             linestyle=line_styles_with[i],
             label=f'Class {i+1} with knowledge (AUC = {roc_auc_with[i]:.3f})')
    
    # without knowledge (虚线)
    plt.plot(fpr_without[i], tpr_without[i], color=colors[i], lw=lw,
             linestyle=line_styles_without[i],
             label=f'Class {i+1} without knowledge (AUC = {roc_auc_without[i]:.3f})')

# 绘制micro-average ROC曲线
plt.plot(fpr_with["micro"], tpr_with["micro"],
         label=f'micro-average with knowledge (AUC = {roc_auc_with["micro"]:.3f})',
         color='deeppink', linestyle='-', linewidth=4)
plt.plot(fpr_without["micro"], tpr_without["micro"],
         label=f'micro-average without knowledge (AUC = {roc_auc_without["micro"]:.3f})',
         color='deeppink', linestyle='--', linewidth=4)

# 绘制macro-average ROC曲线
plt.plot(fpr_with["macro"], tpr_with["macro"],
         label=f'macro-average with knowledge (AUC = {roc_auc_with["macro"]:.3f})',
         color='navy', linestyle='-', linewidth=4)
plt.plot(fpr_without["macro"], tpr_without["macro"],
         label=f'macro-average without knowledge (AUC = {roc_auc_without["macro"]:.3f})',
         color='navy', linestyle='--', linewidth=4)

# 绘制对角线
plt.plot([0, 1], [0, 1], 'k--', lw=lw, alpha=0.5)

# 设置图表属性
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.02])
plt.xlabel('False positive rate', fontsize=33)
plt.ylabel('True positive rate', fontsize=33)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(loc="lower right", fontsize=18)

# 设置图表边框为黑色
ax = plt.gca()
ax.spines['top'].set_color('black')
ax.spines['bottom'].set_color('black')
ax.spines['left'].set_color('black')
ax.spines['right'].set_color('black')
ax.spines['top'].set_linewidth(3)
ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)

plt.grid(True, alpha=0.3)
plt.savefig('roc_comparison_8-7.svg', format='svg', dpi=300, bbox_inches='tight')

plt.show()

# 打印对比结果
print("=== AUC值对比 ===")
print("With Knowledge:")
for i in range(n_classes):
    print(f"  Class {i+1}: {roc_auc_with[i]:.3f}")
print(f"  Micro-average: {roc_auc_with['micro']:.3f}")
print(f"  Macro-average: {roc_auc_with['macro']:.3f}")

print("\nWithout Knowledge:")
for i in range(n_classes):
    print(f"  Class {i+1}: {roc_auc_without[i]:.3f}")
print(f"  Micro-average: {roc_auc_without['micro']:.3f}")
print(f"  Macro-average: {roc_auc_without['macro']:.3f}")

# %%
