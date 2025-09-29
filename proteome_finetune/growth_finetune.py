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







def pad_and_mask(batch, index, total_genes=5812):
    """
    batch: [B, 1780] 蛋白质组数据
    index: [1780] 有效基因位置索引
    return:
        padded_data: [B, 5812]
        mask: [B, 5812] (True表示填充位置)
    """
    device = batch.device
    B = batch.size(0)
    padded_data = torch.full((B, total_genes),
                              fill_value=-2, 
                              dtype=batch.dtype,
                              device=device)
    padded_data[:, index] = batch

    mask = torch.ones((B, total_genes),
                      dtype=torch.bool,
                      device=device)
    mask[:, index] = False

    return padded_data,mask
# 修改后的DSgraphNet类添加特征提取方法
class DSgraphNet(nn.Module):
    def __init__(self, output_attention=True, d_model=200, d_ff=800, max_value=800,
                 factor=5, n_heads=1, activation='ReLu', dropout=0.1):
        super(DSgraphNet, self).__init__()
        self.output_attention = output_attention

        # Embedding
        self.pos_embedding = Gene2VecPositionalEmbedding()
        self.token_embedding = AutoDiscretizationEmbedding2(dim=d_model)

        # Dynamic
        self.Dynamicencoder = DynamicEncoder(
            [
                DynamicEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model=d_model, n_heads=n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Static
        self.Staticencoder = StaticEncoder(
            [
                StaticEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model=d_model, n_heads=n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )

        # Gate
        self.gate = GatedFusion(d_model)
        
    def forward_features(self, x):
        """特征提取专用前向传播"""
        x_nonmasked = x.clone().float()
        masked_x,mask_seqs=x,None
        padded_x,promask = pad_and_mask(masked_x,proindex)
        promask = promask.unsqueeze(1).unsqueeze(1)
        # Embedding
        x1 = self.pos_embedding(padded_x)
        x2,x2_mask_idx,x2_zero_idx,x2_pad_idx = self.token_embedding(padded_x)
        #x2_masked, mask_seqs = random_mask(x2)
        x = x1 + x2

        # parallel
        Dynamic_out, Dynamic_attns = self.Dynamicencoder(x,promask)
        Static_out, Static_attns = self.Staticencoder(x,promask)

        # Gate
        y = self.gate(Dynamic_out, Static_out)

        y = y[:,proindex,:]
        
        # 特征聚合（平均池化）
        #return torch.mean(y, dim=2)
        return torch.max(y, dim=1).values

# 特征提取函数
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    i=0
    with torch.no_grad():
        for batch in dataloader:
            print(i)
            x = batch['features'].to(device).float()  # 确保数据类型正确
            batch_features = model.forward_features(x)
            features.append(batch_features.cpu().numpy())
            i+=1
    return np.concatenate(features, axis=0)

# 修改后的机器学习评估函数
# def evaluate_with_features(features, labels):
#     kf = KFold(n_splits=10, shuffle=True)
#     results = []
    
#     for train_idx, test_idx in kf.split(features):
#         # 数据划分
#         X_train, X_test = features[train_idx], features[test_idx]
#         y_train, y_test = labels[train_idx], labels[test_idx]
        
#         # 模型训练
#         model = ExtraTreesRegressor(n_estimators=300)
#         model.fit(X_train, y_train)
        
#         # 预测评估
#         pred = model.predict(X_test)
#         mse = mean_squared_error(y_test, pred)
#         r2 = r2_score(y_test, pred)
        
#         results.append({'mse': mse, 'r2': r2})
#         print(mse,r2)
    
#     # 打印平均结果
#     avg_r2 = np.mean([res['r2'] for res in results])
#     print(f"Average R²: {avg_r2:.4f}")
#     return results

def evaluate_with_features(features, labels, output_dir='ml_raw_results_5'):
    os.makedirs(output_dir, exist_ok=True)
    kf = KFold(n_splits=10, shuffle=True)
    results = []
    
    n_samples = features.shape[0]
    # total_pred = np.full(n_samples, np.nan)
    # total_true = np.full(n_samples, np.nan)
    
    for fold_i, (train_idx, test_idx) in enumerate(kf.split(features), 1):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        
        model = ExtraTreesRegressor(n_estimators=300)
        model.fit(X_train, y_train)
        
        # Save model
        model_path = os.path.join(output_dir, f'extra_tree_fold_{fold_i}.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Predict and evaluate
        pred = model.predict(X_test)
        mse = mean_squared_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        print(mse,r2)
        results.append({'mse': mse, 'r2': r2, 'fold': fold_i})
        
        # Save fold predictions
        fold_df = pd.DataFrame({
            'index': test_idx,
            'true': y_test,
            'pred': pred
        })
        fold_df.to_csv(os.path.join(output_dir, f'fold_{fold_i}_predictions.csv'), index=False)
        
        # Update total arrays
    
    # Save all predictions
    
    avg_r2 = np.mean([res['r2'] for res in results])
    print(f"Average R²: {avg_r2:.4f}")
    return results

if __name__ == '__main__':
    # 配置设置
    os.chdir('./data')
    device = torch.device("cuda:0")
    torch.set_default_dtype(torch.float32)
    
    # # # # 1. 数据加载
    features, labels = read_age_fine_tune_data()  # 保持原有数据加载方法
    
    #2. 模型准备
    # net= DSgraphNet().to(device).float()
    # pretrained_dict = torch.load(f'final_checkpoint_spearman0.5_w_knowledge_v4pro.pth')['net']
    # model_dict = net.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict) 
    # net.load_state_dict(model_dict, strict=False)
    # #pretrained_dict = torch.load(f'final_checkpoprint(mse,r2)')
    # dataset=CustomDataset(features,labels)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    
    # feature_matrix = extract_features(net, dataloader, device)
    
    # # 4. 特征存储
    # with open("pretrained_features_200_max.pkl", "wb") as f:
    #     pickle.dump({'features': feature_matrix, 'labels': labels}, f)
    
    # # 5. 机器学习模型评估
    with open("data/pretrained_features_max.pkl", "rb") as f:
        data1 = pickle.load(f)
    
    with open("data/pretrained_features_min.pkl", "rb") as f:
        data2 = pickle.load(f)
    
    # with open("pretrained_features_200_max.pkl", "rb") as f:
    #     data3 = pickle.load(f)
    
    embedding_feature=np.concatenate((data1['features'],data2['features']),axis=1)
    # #embedding_feature = data1['features']
    # embedding_feature= features
    for i in range(2,21):
        results = evaluate_with_features(embedding_feature, data1['labels'],output_dir=f'ml_model_results{i}')
    
        # 可选：保存评估结果
        pd.DataFrame(results).to_csv(f"ml_model_performance_max_min_{i}_10.csv", index=False)