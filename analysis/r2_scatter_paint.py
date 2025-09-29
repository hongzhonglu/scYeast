
# %%
import pandas as pd

val_data=pd.read_csv("/home/lulab/scyeast/rpf_kfold_results/round_0/fold_2/predictions.csv")

# %%
import pandas as pd

val_data=pd.read_csv("/home/lulab/scyeast/turnover_kfold_results/round_4/fold_0/predictions.csv")

# %%
import pandas as pd

val_data=pd.read_csv("../data/ml_model_results5/fold_7_predictions.csv")

# %%
real=val_data["true"]
predict = val_data["pred"]

# %%
from sklearn.metrics import r2_score

# 示例数据（假设 y_true 是真实值，y_pred 是预测值）
y_true = real
y_pred = predict

# 计算R²
r_squared = r2_score(y_true, y_pred)
print(f"R² (sklearn): {r_squared:.4f}")

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

plt.rcParams.update({
    "font.family": "Arial", 
    "axes.labelsize": 24,  # 增大坐标轴标签大小
    "xtick.labelsize": 18,  # 增大刻度标签大小
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "axes.titlesize": 20,
    "axes.linewidth": 1.5,
    "grid.alpha": 0.8,
    "grid.linestyle": '--',
    "legend.frameon": False
})

# 计算数据范围，减少空白
data_min = min(min(y_true), min(y_pred))
data_max = max(max(y_true), max(y_pred))
data_range = data_max - data_min
margin = data_range * 0.05  # 5%的边距

plt.figure(figsize=(12, 8))

# 创建密度散点图
# 计算点的密度
xy = np.vstack([y_true, y_pred])
z = gaussian_kde(xy)(xy)

# 绘制密度散点图
scatter = plt.scatter(y_true, y_pred, c=z, cmap='viridis', alpha=0.7, s=30)

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('Density', fontsize=18)

# 绘制y=x参考线
plt.plot([data_min - margin, data_max + margin], [data_min - margin, data_max + margin], 'r--', lw=2, label='y = x')

# 设置优化的坐标轴范围
plt.xlim(data_min - margin, data_max + margin)
plt.ylim(data_min - margin, data_max + margin)

# 添加R²标识
plt.text(
    x = data_min + margin,  # 调整位置
    y = data_max - margin,  # 调整位置
    s = f'$R^2 = {r_squared:.4f}$',
    fontsize = 20,  # 增大字体
    bbox = dict(
        facecolor = 'white', 
        alpha = 0.9,
        edgecolor = 'gray',
        boxstyle = 'round,pad=0.5'
    )
)

# 添加标签和标题
plt.xlabel('Ribosome occupancy(log2-normalized)', fontsize=24)
plt.ylabel('Prediction', fontsize=24)
plt.grid(alpha=0.3)
plt.savefig(
    '../turnover/rpf_scatter.svg',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

plt.rcParams.update({
    "font.family": "Arial", 
    "axes.labelsize": 24,  # 增大坐标轴标签大小
    "xtick.labelsize": 18,  # 增大刻度标签大小
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "axes.titlesize": 20,
    "axes.linewidth": 1.5,
    "grid.alpha": 0.8,
    "grid.linestyle": '--',
    "legend.frameon": False
})

# 计算数据范围，减少空白
data_min = min(min(y_true), min(y_pred))
data_max = max(max(y_true), max(y_pred))
data_range = data_max - data_min
margin = data_range * 0.05  # 5%的边距

plt.figure(figsize=(12, 8))

# 创建密度散点图
# 计算点的密度
xy = np.vstack([y_true, y_pred])
z = gaussian_kde(xy)(xy)

# 绘制密度散点图
scatter = plt.scatter(y_true, y_pred, c=z, cmap='viridis', alpha=0.7, s=30)

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('Density', fontsize=18)

# 绘制y=x参考线
plt.plot([data_min - margin, data_max + margin], [data_min - margin, data_max + margin], 'r--', lw=2, label='y = x')

# 设置优化的坐标轴范围
plt.xlim(data_min - margin, data_max + margin)
plt.ylim(data_min - margin, data_max + margin)

# 添加R²标识
plt.text(
    x = data_min + margin,  # 调整位置
    y = data_max - margin,  # 调整位置
    s = f'$R^2 = {r_squared:.4f}$',
    fontsize = 20,  # 增大字体
    bbox = dict(
        facecolor = 'white', 
        alpha = 0.9,
        edgecolor = 'gray',
        boxstyle = 'round,pad=0.5'
    )
)

# 添加标签和标题
plt.xlabel('Turnover(log2-normalized)', fontsize=24)
plt.ylabel('Prediction', fontsize=24)
plt.grid(alpha=0.3)
plt.savefig(
    '../turnover/turnover_scatter.svg',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.show()

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

plt.rcParams.update({
    "font.family": "Arial", 
    "axes.labelsize": 24,  # 增大坐标轴标签大小
    "xtick.labelsize": 18,  # 增大刻度标签大小
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "axes.titlesize": 20,
    "axes.linewidth": 1.5,
    "grid.alpha": 0.8,
    "grid.linestyle": '--',
    "legend.frameon": False
})

# 计算数据范围，减少空白
data_min = min(min(y_true), min(y_pred))
data_max = max(max(y_true), max(y_pred))
data_range = data_max - data_min
margin = data_range * 0.05  # 5%的边距

plt.figure(figsize=(12, 8))

# 创建密度散点图
# 计算点的密度
xy = np.vstack([y_true, y_pred])
z = gaussian_kde(xy)(xy)

# 绘制密度散点图
scatter = plt.scatter(y_true, y_pred, c=z, cmap='viridis', alpha=0.7, s=30)

# 添加颜色条
cbar = plt.colorbar(scatter)
cbar.set_label('Density', fontsize=18)

# 绘制y=x参考线
plt.plot([data_min - margin, data_max + margin], [data_min - margin, data_max + margin], 'r--', lw=2, label='y = x')

# 设置优化的坐标轴范围
plt.xlim(data_min - margin, data_max + margin)
plt.ylim(data_min - margin, data_max + margin)

# 添加R²标识
plt.text(
    x = data_min + margin,  # 调整位置
    y = data_max - margin,  # 调整位置
    s = f'$R^2 = {r_squared:.4f}$',
    fontsize = 20,  # 增大字体
    bbox = dict(
        facecolor = 'white', 
        alpha = 0.9,
        edgecolor = 'gray',
        boxstyle = 'round,pad=0.5'
    )
)

# 添加标签和标题
plt.xlabel('Growth rate', fontsize=24)
plt.ylabel('Prediction', fontsize=24)
plt.grid(alpha=0.3)
plt.savefig(
    '../turnover/growth_scatter.svg',
    dpi=300,
    bbox_inches='tight',
    facecolor='white'
)
plt.show()
# %%
