# %%
with open("../data/output_log_v4.txt","r") as f:
    
    content = f.read().replace('/n', '\n')
lines = [line.strip() for line in content.split('\n') if line.strip()]

# %%


# %%
import pandas as pd
import matplotlib.pyplot as plt


# %% [markdown]
# 

# %%


# %%
lines[11000]

# %%
# 解析每行数据
data = []
for line in lines:
    entry = {}
    parts = line.split('|')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # 提取键值对（兼容大小写和空格）
        last_space = part.rfind(' ')
        if last_space == -1:
            continue
        key = part[:last_space].strip().lower().replace(' ', '_')  # 标准化键名
        value = part[last_space+1:].strip()
        
        # 转换数据类型
        if key == 'epoch':
            entry[key] = int(value)
        elif key == 'batch':
            entry[key] = int(value)
        elif key == 'time':
            entry[key] = float(value.replace('s', ''))
        elif key in ['total_loss', 'mask_zero_loss', 'mask_nonzero_loss', 'zero_loss', 'nonzero_loss']:
            entry[key] = float(value)
    data.append(entry)

# %%
data[10700]

# %%
df = pd.DataFrame(data)
df['global_step'] = df.index  # 生成全局步数
df = pd.DataFrame(data)
df['global_step'] = df.index  # 生成全局步数

# 可视化
plt.figure(figsize=(15, 12))
loss_columns = ['total_loss', 'mask_zero_loss', 'mask_nonzero_loss', 'zero_loss', 'nonzero_loss']

# 绘制每个loss的子图
for i, col in enumerate(loss_columns, 1):
    plt.subplot(len(loss_columns), 1, i)
    plt.plot(df['global_step'], df[col], label=col.replace('_', ' ').title())
    
    # 标记epoch边界
    epoch_changes = df[df['epoch'].diff() > 0].index
    for change in epoch_changes:
        plt.axvline(change, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    
    plt.xlabel('Global Step (Batch Index)')
    plt.ylabel('Loss Value')
    plt.title(f'{col.replace("_", " ").title()} over Training Steps')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

plt.tight_layout()
plt.show()


# %%
df = pd.DataFrame(data)
df['global_step'] = df.index  # 生成全局步数
df = pd.DataFrame(data)
df['global_step'] = df.index  # 生成全局步数

# 可视化
plt.figure(figsize=(15, 12))
loss_columns = ['total_loss', 'mask_zero_loss', 'mask_nonzero_loss', 'zero_loss', 'nonzero_loss']

# 绘制每个loss的子图
for i, col in enumerate(loss_columns, 1):
    plt.subplot(len(loss_columns), 1, i)
    plt.plot(df['global_step'], df[col], label=col.replace('_', ' ').title())
    
    # 标记epoch边界
    epoch_changes = df[df['epoch'].diff() > 0].index
    for change in epoch_changes:
        plt.axvline(change, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    
    plt.xlabel('Global Step (Batch Index)')
    plt.ylabel('Loss Value')
    plt.title(f'{col.replace("_", " ").title()} over Training Steps')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

plt.tight_layout()
plt.show()


# %%
# 修改后的数据解析代码
data = []
for line in lines:
    entry = {'is_validation': False}  # 默认标记为训练集
    parts = [p.strip() for p in line.split('|')]
    
    # 检查是否包含验证集标记
    if parts[0].lower() == 'validation':
        entry['is_validation'] = True
        parts = parts[1:]  # 移除validation标记
    
    for part in parts:
        if not part:
            continue
            
        # 提取键值对（兼容大小写和空格）
        last_space = part.rfind(' ')
        if last_space == -1:
            continue
            
        key = part[:last_space].strip().lower().replace(' ', '_')
        value = part[last_space+1:].strip()
        
        # 转换数据类型
        if key == 'epoch':
            entry[key] = int(value)
        elif key == 'batch':
            entry[key] = int(value)
        elif key == 'time':
            entry[key] = float(value.replace('s', ''))
        elif key in ['total_loss', 'mask_zero_loss', 
                    'mask_nonzero_loss', 'zero_loss', 'nonzero_loss']:
            entry[key] = float(value)
            
    data.append(entry)

# %%


# 修改后的可视化代码
df = pd.DataFrame(data)
df['global_step'] = df.index  # 生成全局步数

# 分离训练集和验证集数据
train_df = df[df['is_validation'] == False]
val_df = df[df['is_validation'] == True]

plt.figure(figsize=(15, 20))
loss_columns = ['total_loss', 'mask_zero_loss', 
               'mask_nonzero_loss', 'zero_loss', 'nonzero_loss']

# 绘制每个loss的双曲线对比
for i, col in enumerate(loss_columns, 1):
    plt.subplot(len(loss_columns), 1, i)
    
    # 绘制训练曲线
    plt.plot(train_df['global_step'], train_df[col], 
            label='Train', color='blue', alpha=0.8, linewidth=1)
    
    # 绘制验证曲线（如果存在数据点）
    if not val_df.empty:
        plt.plot(val_df['global_step'], val_df[col], 
                label='Validation', color='orange', alpha=0.8, linewidth=1, linestyle='--')
    
    # 标记epoch边界（使用原始数据中的epoch变化）
    epoch_changes = df[df['epoch'].diff() > 0].index
    for change in epoch_changes:
        plt.axvline(change, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    
    plt.xlabel('Global Step')
    plt.ylabel('Loss Value')
    plt.title(f'{col.replace("_", " ").title()} Comparison')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # 自动调整y轴范围
    y_min = min(train_df[col].min(), val_df[col].min() if not val_df.empty else np.inf)
    y_max = max(train_df[col].max(), val_df[col].max() if not val_df.empty else -np.inf)
    plt.ylim(y_min - 0.1*(y_max-y_min), y_max + 0.1*(y_max-y_min))

plt.tight_layout()
plt.savefig(
    'v4_loss.pdf',  # 保存路径（可修改为如 './results/loss.png'）
    dpi=300,                        # 分辨率（印刷推荐300dpi，屏幕显示72dpi即可）
    bbox_inches='tight',            # 自动裁剪白边
    facecolor='white'               # 背景设为白色（默认是透明）
)
plt.show()

# %%
with open("./data/output_pro.txt","r") as f:
    
    content = f.read().replace('/n', '\n')
lines = [line.strip() for line in content.split('\n') if line.strip()]

# %%
# 修改后的数据解析代码
data = []
for line in lines:
    entry = {'is_validation': False}  # 默认标记为训练集
    parts = [p.strip() for p in line.split('|')]
    
    # 检查是否包含验证集标记
    if parts[0].lower() == 'validation':
        entry['is_validation'] = True
        parts = parts[1:]  # 移除validation标记
    
    for part in parts:
        if not part:
            continue
            
        # 提取键值对（兼容大小写和空格）
        last_space = part.rfind(' ')
        if last_space == -1:
            continue
            
        key = part[:last_space].strip().lower().replace(' ', '_')
        value = part[last_space+1:].strip()
        
        # 转换数据类型
        if key == 'epoch':
            entry[key] = int(value)
        elif key == 'batch':
            entry[key] = int(value)
        elif key == 'time':
            entry[key] = float(value.replace('s', ''))
        elif key in ['total_loss', 'mask_zero_loss', 
                    'mask_nonzero_loss', 'zero_loss', 'nonzero_loss']:
            entry[key] = float(value)
            
    data.append(entry)

# %%
data_a=data

#
with open("./data/output_pro_nopretrain.txt", "r") as f:
    content_b = f.read().replace('/n', '\n')
lines_b = [line.strip() for line in content_b.split('\n') if line.strip()]

# 解析data_b
data_b = []
for line in lines_b:
    entry = {'is_validation': False}  # 默认标记为训练集
    parts = [p.strip() for p in line.split('|')]
    
    # 检查是否包含验证集标记
    if parts[0].lower() == 'validation':
        entry['is_validation'] = True
        parts = parts[1:]  # 移除validation标记
    
    for part in parts:
        if not part:
            continue
            
        # 提取键值对（兼容大小写和空格）
        last_space = part.rfind(' ')
        if last_space == -1:
            continue
            
        key = part[:last_space].strip().lower().replace(' ', '_')
        value = part[last_space+1:].strip()
        
        # 转换数据类型
        if key == 'epoch':
            entry[key] = int(value)
        elif key == 'batch':
            entry[key] = int(value)
        elif key == 'time':
            entry[key] = float(value.replace('s', ''))
        elif key in ['total_loss', 'mask_zero_loss', 
                    'mask_nonzero_loss', 'zero_loss', 'nonzero_loss']:
            entry[key] = float(value)
            
    data_b.append(entry)

# %%
import pandas as pd
data_a = pd.DataFrame(data_a)
data_b = pd.DataFrame(data_b)
data_a['global_step'] = data_a.index  
data_b['global_step'] = data_b.index

# %%
# 假设已有两个模型的data_a和data_b
# 预处理数据，分离训练集和验证集
train_df_a = data_a[data_a['is_validation'] == False]
val_df_a = data_a[data_a['is_validation'] == True]
train_df_b = data_b[data_b['is_validation'] == False]
val_df_b = data_b[data_b['is_validation'] == True]

plt.figure(figsize=(15, 10))
loss_columns = ['mask_nonzero_loss']

for i, col in enumerate(loss_columns, 1):
    plt.subplot(len(loss_columns), 1, i)
    
    # 绘制模型A的曲线
    plt.plot(train_df_a['global_step'], train_df_a[col], 
            label='Proteomic model Train', color='blue', alpha=0.8, linewidth=1, linestyle='-')
    if not val_df_a.empty:
        plt.plot(val_df_a['global_step'], val_df_a[col], 
                label='Proteomic model Val', color='green', alpha=0.8, linewidth=1, linestyle='--')
    
    # 绘制模型B的曲线
    plt.plot(train_df_b['global_step'], train_df_b[col], 
            label='No pretrain model Train', color='red', alpha=0.8, linewidth=1, linestyle='-')
    if not val_df_b.empty:
        plt.plot(val_df_b['global_step'], val_df_b[col], 
                label='No pretrain model Val', color='yellow', alpha=0.8, linewidth=1, linestyle='--')
    
    # 调整坐标轴范围
    y_min = min(train_df_a[col].min(), val_df_a[col].min() if not val_df_a.empty else float('inf'),
                train_df_b[col].min(), val_df_b[col].min() if not val_df_b.empty else float('inf'))
    y_max = max(train_df_a[col].max(), val_df_a[col].max() if not val_df_a.empty else float('-inf'),
                train_df_b[col].max(), val_df_b[col].max() if not val_df_b.empty else float('-inf'))
    plt.ylim(y_min - 0.1*(y_max-y_min), y_max + 0.1*(y_max-y_min))
    
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.title(f'{col.replace("_", " ").title()}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

plt.tight_layout()
# plt.savefig(
#     'pro_compare_loss3.pdf',  # 保存路径（可修改为如 './results/loss.png'）
#     dpi=300,                        # 分辨率（印刷推荐300dpi，屏幕显示72dpi即可）
#     bbox_inches='tight',            # 自动裁剪白边
#     facecolor='white'               # 背景设为白色（默认是透明）
#plt.savefig("./compare_loss3.svg",format="svg",bbox_inches="tight")
# )
#plt.savefig('comparison_loss_single.pdf', dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# %%
# 将用于绘制compare_loss3的数据单独保存为csv文件
compare_loss3_data = {
    'train_df_a': train_df_a[['global_step', 'mask_nonzero_loss']],
    'val_df_a': val_df_a[['global_step', 'mask_nonzero_loss']],
    'train_df_b': train_df_b[['global_step', 'mask_nonzero_loss']],
    'val_df_b': val_df_b[['global_step', 'mask_nonzero_loss']]
}

compare_loss3_data['train_df_a'].to_csv('compare_loss3_train_a.csv', index=False)
compare_loss3_data['val_df_a'].to_csv('compare_loss3_val_a.csv', index=False)
compare_loss3_data['train_df_b'].to_csv('compare_loss3_train_b.csv', index=False)
compare_loss3_data['val_df_b'].to_csv('compare_loss3_val_b.csv', index=False)

# 以下是依据上述保存的数据绘图的代码
import pandas as pd
import matplotlib.pyplot as plt

train_df_a = pd.read_csv('compare_loss3_train_a.csv')
val_df_a = pd.read_csv('compare_loss3_val_a.csv')
train_df_b = pd.read_csv('compare_loss3_train_b.csv')
val_df_b = pd.read_csv('compare_loss3_val_b.csv')

plt.figure(figsize=(15, 10))
loss_columns = ['mask_nonzero_loss']

for i, col in enumerate(loss_columns, 1):
    plt.subplot(len(loss_columns), 1, i)
    
    plt.plot(train_df_a['global_step'], train_df_a[col], 
            label='Proteomic model Train', color='blue', alpha=0.8, linewidth=1, linestyle='-')
    if not val_df_a.empty:
        plt.plot(val_df_a['global_step'], val_df_a[col], 
                label='Proteomic model Val', color='green', alpha=0.8, linewidth=1, linestyle='--')
    
    plt.plot(train_df_b['global_step'], train_df_b[col], 
            label='No pretrain model Train', color='red', alpha=0.8, linewidth=1, linestyle='-')
    if not val_df_b.empty:
        plt.plot(val_df_b['global_step'], val_df_b[col], 
                label='No pretrain model Val', color='yellow', alpha=0.8, linewidth=1, linestyle='--')
    
    y_min = min(train_df_a[col].min(), val_df_a[col].min() if not val_df_a.empty else float('inf'),
                train_df_b[col].min(), val_df_b[col].min() if not val_df_b.empty else float('inf'))
    y_max = max(train_df_a[col].max(), val_df_a[col].max() if not val_df_a.empty else float('-inf'),
                train_df_b[col].max(), val_df_b[col].max() if not val_df_b.empty else float('-inf'))
    plt.ylim(y_min - 0.1*(y_max-y_min), y_max + 0.1*(y_max-y_min))
    
    plt.xlabel('Global Step')
    plt.ylabel('Loss')
    plt.title(f'{col.replace("_", " ").title()}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

plt.tight_layout()
#plt.savefig("./compare_loss3_from_csv.svg",format="svg",bbox_inches="tight")
plt.show()

# %%
