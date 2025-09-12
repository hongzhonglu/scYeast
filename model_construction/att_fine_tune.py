import torch
import torch.nn as nn
import os
import torch.optim as optim
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import ast
import warnings

from DSgraph_main import DSgraphNet
from load_data import read_relation_data
from get_att_io import CustomDataset


def draw_dynamic_attn_graph(attn_matrix):
    """
    绘制单个样本的自注意力热图并保存
    :param attn_matrix: (5812, 5812) 的 PyTorch 张量
    """
    attn_matrix = attn_matrix.cpu().detach()

    # 归一化
    max_val = torch.max(attn_matrix)
    min_val = torch.min(attn_matrix)
    normalized = (attn_matrix - min_val) / (max_val - min_val)

    # 设定阈值：过滤较低权重
    filtered_matrix = normalized.clone()
    filtered_matrix[filtered_matrix < 0.85] = 0

    # 绘制热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(filtered_matrix.numpy()), cmap="coolwarm", square=True)
    plt.title("Dynamic Attention Heatmap")
    plt.show()
    return filtered_matrix


class AdditionalNetwork(nn.Module):
    def __init__(self):
        super(AdditionalNetwork, self).__init__()
        self.conv1d1 = nn.Conv1d(in_channels=200, out_channels=1, kernel_size=1)
        self.conv1d2 = nn.Conv1d(in_channels=5812, out_channels=512, kernel_size=1)
        self.conv1d3 = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d1(x)
        x = nn.ReLU()(x)
        x = x.transpose(1, 2)
        x = self.conv1d2(x)
        x = nn.ReLU()(x)
        x = self.conv1d3(x)
        x = nn.Sigmoid()(x)
        return x.squeeze(-1)


class CombinedNet(nn.Module):
    def __init__(self, net1, net2):
        super(CombinedNet, self).__init__()
        self.net1 = net1
        self.net2 = net2
        for param in self.net1.parameters():
            param.requires_grad = False

    def forward(self, x, relation):
        with torch.no_grad():
            _, net_output, attn_matrix = self.net1(x, relation)  # 获取注意力矩阵
        y_out = self.net2(net_output)
        return y_out, attn_matrix


def read_age_fine_tune_data(batch_size):
    """
    读取微调数据集
    """
    age_dataset = pd.read_csv('alignment_enz_data_with_labels.csv')
    features = age_dataset.iloc[:, :-1]
    labels = age_dataset.iloc[:, -1]

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2)

    train_dataset = CustomDataset(features_train, labels_train)
    test_dataset = CustomDataset(features_test, labels_test)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 1000)
    np.set_printoptions(linewidth=400, threshold=200)

    os.chdir('../data')
    train_data, test_data = read_age_fine_tune_data(batch_size=5)
    torch.set_default_dtype(torch.float32)
    device = 'cuda'

    # 加载预训练模型
    net = DSgraphNet().to(device).to(torch.float32)
    addition_net = AdditionalNetwork().to(device).to(torch.float32)
    pre_model = torch.load('final_checkpoint_200_10.pth')
    net.load_state_dict(pre_model['net'])

    # relation_np = read_relation_data()
    # relation_tensor = torch.tensor(relation_np, dtype=torch.float32)
    # relation = relation_tensor.to(device)

    # 组合模型
    combined_model = CombinedNet(net, addition_net).to(device)

    # 训练设置
    loss_func = nn.MSELoss(reduction='mean')
    optimizer = optim.AdamW(combined_model.parameters(), lr=0.001, weight_decay=0.01)
    max_epoch = 8
    train_loss_history = []
    test_loss_history = []

    for epoch in range(max_epoch):
        train_loss = 0.0
        test_loss = 0.0
        batch_count = 0

        for batch_id, batch in enumerate(train_data):
            optimizer.zero_grad()
            x = batch['features'].to(device)
            y = batch['label'].to(device)

            y_out, _ = combined_model(x)
            loss = loss_func(y_out, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batch_count += 1
            print(f'Epoch {epoch + 1}/{max_epoch}, Batch {batch_id + 1}, Train Loss={loss.item():.6f}')

        avg_train_loss = train_loss / batch_count
        train_loss_history.append(avg_train_loss)

        combined_model.eval()
        batch_count = 0
        with torch.no_grad():
            for batch_id, batch in enumerate(test_data):
                x = batch['features'].to(device)
                y = batch['label'].to(device)

                y_out, _ = combined_model(x)
                loss = loss_func(y_out, y)

                test_loss += loss.item()
                batch_count += 1

        # **计算平均测试损失**
        avg_test_loss = test_loss / batch_count
        test_loss_history.append(avg_test_loss)

        print(f'epoch={epoch + 1}/{max_epoch}, batch={batch_id + 1}, loss={loss.item():.6f}')
    print("训练完成！")

    # **提取所有微调后样本的自注意力矩阵，不求均值**
    all_attn_matrices = []  # 用于存储所有样本 attention 矩阵，每个 shape 为 (5812, 5812)


    with torch.no_grad():
        for batch_id, batch in enumerate(test_data):
            x = batch['features'].to(device)
            # 获取模型输出和 attention，保持原始 shape
            _, attn_matrix = combined_model(x)
            if isinstance(attn_matrix, list):
                attn_matrix = attn_matrix[0]  # 保证是 Tensor

            # 不进行 squeeze，以便与预训练阶段一致，attn_matrix 形状：(batch_size, 1, 5812, 5812)
            attn_matrix = attn_matrix.to(torch.float32)
            batch_size, one, num_genes, _ = attn_matrix.shape

            for i in range(batch_size):
                temp_attn = torch.zeros((num_genes, num_genes)).to(device).to(torch.float32)
                # 对于该样本的每一行（即每个基因）进行归一化处理
                for j in range(num_genes):
                    a = attn_matrix[i][0][j]  # a 的形状为 (5812,)
                    max_val = torch.max(a)
                    min_val = torch.min(a)
                    if max_val == min_val:
                        coff = 1
                        temp_attn[j] = (a - min_val + 0.5) * coff
                    else:
                        coff = 1 / (max_val - min_val)
                        temp_attn[j] = (a - min_val) * coff
                all_attn_matrices.append(temp_attn.detach().cpu())

    # 将所有样本的 attention 矩阵保存为一个列表
    # 若需要统一保存为 Tensor，可用 torch.stack，但注意 batch数可能不固定
    # 这里我们直接保存列表
    with open('final_attention_matrices.pkl', 'wb') as f:
        pickle.dump(all_attn_matrices, f)
    print("所有微调注意力矩阵已保存！")

    # **绘制每个样本的注意力热图**
    for idx, attn_mat in enumerate(all_attn_matrices):
        draw_dynamic_attn_graph(attn_mat)

    plt.figure(figsize=(8, 6))
    plt.plot(train_loss_history, label='Train Loss', color='blue')
    plt.plot(test_loss_history, label='Test Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs. Testing Loss Curve')
    plt.legend()
    plt.grid()
    plt.show()
