import numpy as np
import pandas as pd
import torch
import os
import pickle
import time
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('./model_construction')
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from embedding_pro  import Gene2VecPositionalEmbedding, ExpressionTokenEmbedding, random_mask,AutoDiscretizationEmbedding2
from DSgraph_pro import DynamicEncoder, DynamicEncoderLayer, StaticEncoder, StaticEncoderLayer
from self_attention_pro import FullAttention, AttentionLayer, TriangularCausalMask
from gated_fusion import GatedFusion
from load_data import read_scaled_data_pro
import pickle
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import r2_score
from torch.utils.data import Dataset, DataLoader
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
def read_age_fine_tune_data():
    aligned_protein, growth=load_aligned_data()
    return aligned_protein.numpy(), growth.numpy()
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'features': self.features[idx], 'label': self.labels[idx]}
def load_aligned_data():
    with open("data/scaled_data_1_pro.pkl", "rb") as f:
        full_protein = torch.tensor(pickle.load(f))
    with open("data/growth_tensor.pkl", "rb") as f:
        growth = torch.tensor(pickle.load(f))
    with open("data/sample_index.pkl", "rb") as f:
        indices = pickle.load(f)
    aligned_protein = full_protein[indices]
    YPD = growth[:,-1]
    return aligned_protein,YPD

with open("data/filter_proindex.pkl","rb") as f:
    proindex=torch.tensor(pickle.load(f))
with open("data/rbf_data_filter.pkl","rb") as f:
    rbf_data_filter=torch.tensor(pickle.load(f))
with open("data/pro_interaction.pkl", 'rb') as f:
    all_embeddings = pickle.load(f)  # 形状 (5476, 1795, 200)
with open("data/rbf_data_index.pkl","rb") as f:
    rbf_data_index=torch.tensor(pickle.load(f))
with open("data/rbf_1795_index.pkl","rb") as f:
    rbf_1795_index=torch.tensor(pickle.load(f))

    
#all_embeddings_filter=all_embeddings[:,rbf_1795_index:]

# def evaluate_with_features(features, labels, output_dir='ml_raw_results_5'):
#     os.makedirs(output_dir, exist_ok=True)
#     kf = KFold(n_splits=10, shuffle=True)
#     results = []
    
#     n_samples = features.shape[0]
#     # total_pred = np.full(n_samples, np.nan)
#     # total_true = np.full(n_samples, np.nan)
    
#     for fold_i, (train_idx, test_idx) in enumerate(kf.split(features), 1):
#         X_train, X_test = features[train_idx], features[test_idx]
#         y_train, y_test = labels[train_idx], labels[test_idx]
        
#         model = ExtraTreesRegressor(n_estimators=300)
#         model.fit(X_train, y_train)
        
#         # Save model
#         model_path = os.path.join(output_dir, f'extra_tree_fold_{fold_i}.pkl')
#         with open(model_path, 'wb') as f:
#             pickle.dump(model, f)
        
#         # Predict and evaluate
#         pred = model.predict(X_test)
#         mse = mean_squared_error(y_test, pred)
#         r2 = r2_score(y_test, pred)
#         print(mse,r2)
#         results.append({'mse': mse, 'r2': r2, 'fold': fold_i})
        
#         # Save fold predictions
#         fold_df = pd.DataFrame({
#             'index': test_idx,
#             'true': y_test,
#             'pred': pred
#         })
#         fold_df.to_csv(os.path.join(output_dir, f'fold_{fold_i}_predictions.csv'), index=False)
        
#         # Update total arrays
    
#     # Save all predictions
    
#     avg_r2 = np.mean([res['r2'] for res in results])
#     print(f"Average R²: {avg_r2:.4f}")
#     return results

# if __name__ == '__main__':
#     # 配置设置
#     os.chdir('./data')
#     device = torch.device("cuda:0")
#     torch.set_default_dtype(torch.float32)
    
#     # # # # 1. 数据加载
#     features, labels = read_age_fine_tune_data()  # 保持原有数据加载方法
    
#     #2. 模型准备
#     # net= DSgraphNet().to(device).float()
#     # pretrained_dict = torch.load(f'final_checkpoint_spearman0.5_w_knowledge_v4pro.pth')['net']
#     # model_dict = net.state_dict()
#     # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     # model_dict.update(pretrained_dict) 
#     # net.load_state_dict(model_dict, strict=False)
#     # #pretrained_dict = torch.load(f'final_checkpoprint(mse,r2)')
#     # dataset=CustomDataset(features,labels)
#     # dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
#     # feature_matrix = extract_features(net, dataloader, device)
    
#     # # 4. 特征存储
#     # with open("pretrained_features_200_max.pkl", "wb") as f:
#     #     pickle.dump({'features': feature_matrix, 'labels': labels}, f)
    
#     # # 5. 机器学习模型评估
#     with open("pretrained_features_max.pkl", "rb") as f:
#         data1 = pickle.load(f)
    
#     # with open("pretrained_features.pkl", "rb") as f:
#     #     data2 = pickle.load(f)
    
#     # with open("pretrained_features_200_max.pkl", "rb") as f:
#     #     data3 = pickle.load(f)
    
#     #embedding_feature=np.concatenate((data1['features'],data2['features']),axis=1)
#     # #embedding_feature = data1['features']
#     embedding_feature= features
#     for i in range(1,11):
#         results = evaluate_with_features(embedding_feature, data1['labels'],output_dir=f'rf_raw_results{i}')
    
#         # 可选：保存评估结果
#         pd.DataFrame(results).to_csv(f"rf_model_performance_raw_{i}_10.csv", index=False)
    

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.model_selection import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 数据预处理函数
def preprocess_labels(y):
    """对标签进行 log2 转换和标准化"""
    y_log = np.log2(y + 1e-6)  # 处理零值
    scaler = StandardScaler()
    y_scaled = scaler.fit_transform(y_log.reshape(-1, 1)).flatten()
    return y_scaled, scaler

# 特征提取函数（保持不变）
def extract_pooling_features(embeddings):
    transposed = np.transpose(embeddings, (1, 0, 2))
    max_pool = np.max(transposed, axis=2)
    min_pool = np.min(transposed, axis=2)
    #mean_pool = np.mean(transposed, axis=2)
    return np.concatenate([max_pool, min_pool], axis=1)

# 修改后的评估函数：多轮次K折
def evaluate_protein_regression(features, labels, n_rounds=5, n_splits=5, output_dir='rpf_kfold_results'):
    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    
    for round_idx in range(n_rounds):
        round_dir = os.path.join(output_dir, f'round_{round_idx}')
        os.makedirs(round_dir, exist_ok=True)
        
        # 初始化本轮结果
        round_results = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=round_idx)  # 每轮不同随机种子
        
        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(features)):
            # 数据划分
            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]
            
            # 训练模型
            model = ExtraTreesRegressor(n_estimators=100)
            model.fit(X_train, y_train)
            
            # 保存模型
            fold_model_dir = os.path.join(round_dir, f'fold_{fold_idx}')
            os.makedirs(fold_model_dir, exist_ok=True)
            with open(os.path.join(fold_model_dir, 'model.pkl'), 'wb') as f:
                pickle.dump(model, f)
            
            # 预测评估
            pred = model.predict(X_test)
            mse = mean_squared_error(y_test, pred)
            r2 = r2_score(y_test, pred)
            
            # 保存预测结果
            fold_df = pd.DataFrame({
                'true': y_test,
                'pred': pred,
                'test_indices': test_idx
            })
            fold_df.to_csv(os.path.join(fold_model_dir, 'predictions.csv'), index=False)
            
            # 记录结果
            round_results.append({
                'round': round_idx,
                'fold': fold_idx,
                'mse': mse,
                'r2': r2
            })
            print(f"Round {round_idx} Fold {fold_idx}: MSE={mse:.4f}, R²={r2:.4f}")
        
        # 保存本轮结果
        pd.DataFrame(round_results).to_csv(os.path.join(round_dir, 'round_results.csv'))
        all_results.extend(round_results)
    
    # 汇总所有结果
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(os.path.join(output_dir, 'all_rounds_summary.csv'))
    
    # 计算全局平均
    avg_r2 = summary_df['r2'].mean()
    print(f"Global Average R² across {n_rounds} rounds: {avg_r2:.4f}")
    return summary_df

if __name__ == '__main__':
    # 加载数据（示例路径需替换）
    # with open("/path/to/all_embeddings.pkl", 'rb') as f:
    #     all_embeddings = pickle.load(f)
    # with open("/path/to/rbf_data_filter.pkl", 'rb') as f:
    #     rbf_data = pickle.load(f)
    
    # 预处理标签
    y_processed, _ = preprocess_labels(rbf_data_filter)
    
    # 提取特征
    features = extract_pooling_features(all_embeddings[:, rbf_1795_index, :])
    
    # 运行多轮次K折验证
    results = evaluate_protein_regression(
        features, 
        y_processed,
        n_rounds=5,   # 运行5轮
        n_splits=5,    # 每轮5折
        output_dir='rpf_kfold_results'
    )

    #pd.DataFrame(results).to_csv(f"rpf_max_min_{i}.csv", index=False)