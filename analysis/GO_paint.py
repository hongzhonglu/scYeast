# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# %%


# %%


# %%
import pickle
with open("../tmp3-3/filter_pronames.pkl","rb") as f:
    proname=pickle.load(f)

# %%
import numpy as np
import pandas as pd

# %%
proname

# %%
# 3. 找到最佳模型
import pickle

best_model_path= '/home/lulab/scyeast/data/ml_model_results5/extra_tree_fold_7.pkl'

# 4. 加载最佳模型
with open(best_model_path, "rb") as f:
    best_model = pickle.load(f)

# 5. 提取特征重要性
feature_importances = best_model.feature_importances_

# 6. 获取前30个重要特征的索引
top30_indices = np.argsort(feature_importances)[:][::-1]

# 7. 分析特征组成
results = []
for idx in top30_indices:
    # 判断特征类型
    if idx < 1795:
        feature_type = "max"
        protein_name = proname[idx]
    else:
        feature_type = "min" 
        protein_name = proname[idx - 1795]
    
    results.append({
        "protein": protein_name,
        "type": feature_type,
        "importance": feature_importances[idx],
        "original_index": idx
    })

# 8. 生成结果DataFrame并保存
result_df = pd.DataFrame(results)
print("\nTop 30 important features:")
print(result_df)

# 9. 保存分析结果
result_df.to_csv("growth_top_mportant_proteins2.csv", index=False)

# 可选：统计蛋白出现频次
protein_counts = result_df.groupby('protein').size()
print("\nProtein frequency in top30:")
print(protein_counts.sort_values(ascending=False))

# %%
with open("../turnover/ko_samples.pkl","rb") as f:
    knock_samples=pickle.load(f)

# %%
len(knock_samples)

# %%
# 3. 找到最佳模型
import pickle
#proname=knock_samples
best_model_path= '/home/lulab/scyeast/turnover_kfold_results/round_4/fold_0/model.pkl'

# 4. 加载最佳模型
with open(best_model_path, "rb") as f:
    best_model = pickle.load(f)

# 5. 提取特征重要性
feature_importances = best_model.feature_importances_
print(len(feature_importances))
# 6. 获取前30个重要特征的索引
top30_indices = np.argsort(feature_importances)[:][::-1]

# 7. 分析特征组成
results = []
for idx in top30_indices:
    # 判断特征类型
    if idx < 5476:
        feature_type = "max"
        protein_name = knock_samples[idx]
    else:
        feature_type = "min" 
        protein_name = knock_samples[idx - 5476]
    
    results.append({
        "protein": protein_name,
        "type": feature_type,
        "importance": feature_importances[idx],
        "original_index": idx
    })

# 8. 生成结果DataFrame并保存
result_df = pd.DataFrame(results)
print("\nTop 30 important features:")
print(result_df)

# 9. 保存分析结果
result_df.to_csv("turnover_top_mportant_proteins2.csv", index=False)

# 可选：统计蛋白出现频次
protein_counts = result_df.groupby('protein').size()
print("\nProtein frequency in top30:")
print(protein_counts.sort_values(ascending=False))

# %%
# 3. 找到最佳模型
import pickle
proname=knock_samples
best_model_path= '/home/lulab/scyeast/rpf_kfold_results/round_0/fold_2/model.pkl'

# 4. 加载最佳模型
with open(best_model_path, "rb") as f:
    best_model = pickle.load(f)

# 5. 提取特征重要性
feature_importances = best_model.feature_importances_

# 6. 获取前30个重要特征的索引
top30_indices = np.argsort(feature_importances)[:][::-1]

# 7. 分析特征组成
results = []
for idx in top30_indices:
    # 判断特征类型
    if idx < 5476:
        feature_type = "max"
        protein_name = knock_samples[idx]
    else:
        feature_type = "min" 
        protein_name = knock_samples[idx - 5476]
    
    results.append({
        "protein": protein_name,
        "type": feature_type,
        "importance": feature_importances[idx],
        "original_index": idx
    })

# 8. 生成结果DataFrame并保存
result_df = pd.DataFrame(results)
print("\nTop 30 important features:")
print(result_df)

# 9. 保存分析结果
result_df.to_csv("rpf_top_mportant_proteins2.csv", index=False)

# 可选：统计蛋白出现频次
protein_counts = result_df.groupby('protein').size()
print("\nProtein frequency in top30:")
print(protein_counts.sort_values(ascending=False))

# %%
len(feature_importances)/2

# %%
len(list(set(result_df['protein'])))

# %%
knock_samples

# %%
proname

# %%
with open("../turnover/gene2entry_dict.pkl","rb") as f:
    gene2entry=pickle.load(f)
