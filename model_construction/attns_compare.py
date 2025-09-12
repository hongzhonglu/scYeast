import pickle
import os
import torch
import pandas as pd

os.chdir('../data')
df_true = pd.read_csv('relation_matrix.txt', sep='\s+', index_col=0).astype(int)
with open('final_dynamic_attn_200_10.pkl', 'rb') as file:
    data = pickle.load(file)
final_attns = data.cpu().detach()
max = torch.max(final_attns)
min = torch.min(final_attns)
normalized = final_attns - min
final_attns = normalized / (max - min)
data_predict = (final_attns >= 0.86).float()
data_np = data_predict.numpy()
df_predict = pd.DataFrame(data_np).astype(int)
gene_names = df_true.columns.tolist()
df_predict.columns = gene_names
df_predict.index = gene_names
common_ones = (df_predict == 1) & (df_true == 1)
true_ones = (df_true == 1).sum().sum()
predict_ones = (df_predict == 1).sum().sum()
count_common_ones = common_ones.sum().sum()
print(data)