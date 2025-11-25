import pandas as pd
import ast
import os

os.chdir('..')
# 读取 CSV 文件
file_path = 'pressure_sensitivity_new_true_all.csv'
df = pd.read_csv(file_path)

# 将 Impact 列转换为列表
df['Impact'] = df['Impact'].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else [])

# 标签列表
labels = ['equal', 'low', 'AAS', 'GS']

# 创建一个新列用于存放 max impact value
df['max impact value'] = None

# 根据 Max Impact 找到对应的影响值
for index, row in df.iterrows():
    max_impact_label = row['Max impact']  # 这里假设 Max Impact 存储的是标签
    if max_impact_label in labels and row['Impact']:
        max_impact_index = labels.index(max_impact_label)  # 找到标签的索引
        df.at[index, 'max impact value'] = row['Impact'][max_impact_index]
df['max impact value'].fillna('No Value', inplace=True)
output_file_path = 'processed_data_with_true_max_impact_pressure_new.csv'
df.to_csv(output_file_path, index=False)
