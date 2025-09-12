import numpy as np
import pandas as pd
import torch
import os
import sys
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import torch.optim as optim

sys.path.append('../model_construction')
from read_checkpoint import DSgraphNetWO_Output, DSgraphNet_WO_knowlegde
from torch_geometric.nn import GCNConv
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt

# 可调整的超参数
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0.01
BATCH_SIZE = 2
NUM_EPOCHS = 15
DROPOUT_RATE = 0.3
PAITIENCE = 10


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


class TrainingLogger:
    def __init__(self, output_dir='training_logs'):
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.r2_val_scores = []
        self.r2_test_scores = []

    def log_epoch(self, train_loss, val_loss, test_loss, r2_val_score, r2_test_score):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.test_losses.append(test_loss)
        self.r2_val_scores.append(r2_val_score)
        self.r2_test_scores.append(r2_test_score)

    def save_logs(self):
        logs_df = pd.DataFrame({
            'Train Loss': self.train_losses,
            'Validation Loss': self.val_losses,
            'Test Loss': self.test_losses,
            'R2 val Score': self.r2_val_scores,
            'R2 test Score': self.r2_test_scores
        })
        logs_df.to_csv(os.path.join(self.output_dir, 'promodel_logs_teva_YLL052C_1.csv'), index=False)


class CustomDataset(Dataset):
    def __init__(self, file_path):
        df = pd.read_csv(file_path, sep=',', index_col=0)
        data = df.values.astype(np.float32)
        self.original_indices = df.index.tolist()  # 保存原始索引

        self.x1 = torch.tensor(data[:, :5812])
        self.x2 = torch.tensor(data[:, 5812:11624])
        self.y = torch.tensor(data[:, -1])

    def __len__(self):
        return self.x1.shape[0]

    def __getitem__(self, idx):
        return self.x1[idx], self.x2[idx], self.y[idx], self.original_indices[idx]


def get_train_val_test_loaders(file_path, batch_size=BATCH_SIZE, val_size=0.5, test_size=0.2, shuffle=True):
    dataset = CustomDataset(file_path)
    indices = list(range(len(dataset)))
    all_data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # 首先划分出测试集
    train_idx, test_indices = train_test_split(indices, test_size=test_size, shuffle=shuffle)

    # 从训练集中划分出验证集
    test_idx, val_idx = train_test_split(test_indices, test_size=val_size, shuffle=shuffle)

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, test_idx, all_data_loader


def get_train_test_loaders(file_path, batch_size=BATCH_SIZE, test_size=0.2, shuffle=True):
    dataset = CustomDataset(file_path)
    indices = list(range(len(dataset)))
    train_idx, test_idx = train_test_split(indices, test_size=test_size)
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def read_go_matrix():
    ppi_matrix = pd.read_csv('gene_association_matrix_with_counts.csv', index_col=0)
    ppi_matrix = ppi_matrix.apply(pd.to_numeric, errors='coerce')
    ppi_matrix = ppi_matrix.fillna(0)
    ppi_matrix = ppi_matrix.values
    edges = np.array(np.nonzero(ppi_matrix))
    edge_index = torch.tensor(edges, dtype=torch.long)
    edge_weight = ppi_matrix[edges[0], edges[1]]
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    return edge_index, edge_weight


class ImprovedLoss(nn.Module):
    def __init__(self):
        super(ImprovedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.SmoothL1Loss()

    def forward(self, pred, target, model, l2_lambda=0.001):
        # 确保预测和目标张量维度一致
        mse = self.mse_loss(pred, target)
        huber = self.huber_loss(pred, target)

        # L2正则化
        l2_norm = sum(p.pow(2.0).sum() for p in model.parameters()
                      if p.requires_grad and len(p.shape) > 1)

        return 0.7 * mse + 0.3 * huber + l2_lambda * l2_norm


class GNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, edge_index, edge_weight):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, 100)
        self.conv2 = GCNConv(100, output_dim)
        self.edge_index = edge_index
        self.edge_weight = edge_weight

    def forward(self, x):
        x = self.conv1(x, self.edge_index, self.edge_weight)
        x = nn.LeakyReLU(0.01)(x)
        x = self.conv2(x, self.edge_index, self.edge_weight)
        return x


class CombinedNet(nn.Module):
    def __init__(self, net1, net2):
        super(CombinedNet, self).__init__()
        self.net1 = net1
        self.net2 = net2

        # 保持原始网络的线性层
        self.linear1 = nn.Linear(200, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, 1)
        self.linear4 = nn.Linear(5812, 1024)
        self.linear5 = nn.Linear(1024, 512)
        self.linear6 = nn.Linear(512, 128)
        self.linear7 = nn.Linear(128, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(128)

        self.dropout = nn.Dropout(DROPOUT_RATE)

        # 参数初始化
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # 对新增的线性层应用初始化
        self.linear1.apply(init_weights)
        self.linear2.apply(init_weights)
        self.linear3.apply(init_weights)
        self.linear4.apply(init_weights)
        self.linear5.apply(init_weights)
        self.linear6.apply(init_weights)
        self.linear7.apply(init_weights)

        # 冻结预训练网络参数
        for param in self.net1.parameters():
            param.requires_grad = False

    def forward(self, x1, x2):
        with torch.no_grad():
            net1_out = self.net1(x1)    # x1:2,5812; net1_out:2,5812,200
        x2 = x2.unsqueeze(-1)           # x2:2,5812,1
        x2_emb = self.net2(x2)          # x2:2,5812,200
        combined_x = net1_out + x2_emb  # combined_x:2,5812,200

        x = self.linear1(combined_x)    # x:2,5812,512
        x = nn.LeakyReLU(0.01)(x)
        x = self.dropout(x)

        x = self.linear2(x)             # x:2,5812,128
        x = nn.LeakyReLU(0.01)(x)
        x = self.dropout(x)

        x = self.linear3(x)             # x:2,5812,1
        x = nn.LeakyReLU(0.01)(x)
        x = x.transpose(1, 2)           # x:2,1,5812

        x = self.linear4(x)             # x:2,1,1024
        x = nn.LeakyReLU(0.01)(x)
        x = self.dropout(x)

        x = self.linear5(x)             # x:2,1,512
        x = nn.LeakyReLU(0.01)(x)
        x = self.dropout(x)

        x = self.linear6(x)             # x:2,1,128
        x = nn.LeakyReLU(0.01)(x)
        x = self.dropout(x)

        x = self.linear7(x)             # x:2,1,1
        x = x.transpose(1, 2)           # x:2,1,1
        x = x.squeeze(-1)               # x:2,1
        x = x.squeeze(-1)               # x:2
        return x  # 返回 1 维输出


def comprehensive_model_evaluation(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x1, x2, y in test_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            output = model(x1, x2)
            all_preds.append(output.cpu())
            all_targets.append(y.cpu())

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # 详细评估指标
    # r2_scores = [r2_score(all_targets[:, i], all_preds[:, i]) for i in range(all_targets.shape[1])]
    # mse_scores = [mean_squared_error(all_targets[:, i], all_preds[:, i]) for i in range(all_targets.shape[1])]
    r2_score_single = r2_score(all_targets, all_preds)
    mse_score_single = mean_squared_error(all_targets, all_preds)

    # print("详细评估报告:")
    # for i, (r2, mse) in enumerate(zip(r2_score_single, mse_score_single)):
    #     print(f"输出 {i + 1}: R² = {r2:.6f}, MSE = {mse:.6f}")
    print("详细评估报告:")
    print(f"R² = {r2_score_single:.6f}, MSE = {mse_score_single:.6f}")

    return [r2_score_single]


def main():
    device = torch.device('cuda')
    os.chdir('C:/Users/Wenb1n/PycharmProjects/scYeast/DSGraph/data')

    # 创建保存测试数据的目录
    test_data_dir = 'data_YLL052C'
    os.makedirs(test_data_dir, exist_ok=True)

    logger = TrainingLogger()

    # 加载预训练模型
    net = DSgraphNetWO_Output().to(device).to(torch.float32)
    modified_state_dict = net.state_dict()
    checkpoint = torch.load('final_checkpoint_spearman0.5_w_knowledge_huber_v4.pth')
    checkpoint_state_dict = checkpoint['net']
    for name, param in checkpoint_state_dict.items():
        if name in modified_state_dict:
            modified_state_dict[name].copy_(param)
    net.load_state_dict(modified_state_dict)
    net.eval()

    # 加载GO
    edge_index, edge_weight = read_go_matrix()
    edge_index = edge_index.to(device)
    edge_weight = edge_weight.to(device)
    go_gnn = GNNModel(1, 200, edge_index, edge_weight).to(device).to(torch.float32)

    model = CombinedNet(net, go_gnn).to(device).to(torch.float32)
    train_loader, val_loader, test_loader, test_indices, all_data_loader = get_train_val_test_loaders('pert_train_data/pert_training_data_YLL052C.csv')

    # 优化器和损失函数
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )

    # 损失函数
    criterion = nn.MSELoss(reduction='mean')

    # 训练循环
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_train_loss = 0.0

        for x1, x2, y, _ in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            output = model(x1, x2)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        total_test_loss = 0.0
        all_val_preds = []
        all_val_targets = []
        all_test_preds = []
        all_test_targets = []
        all_data_preds = []
        all_data_targets = []

        # 每个epoch的数据列表
        epoch__original_indices = []
        epoch__predictions = []
        epoch__targets = []
        # 每个epoch的测试数据列表
        epoch_test_original_indices = []
        epoch_test_predictions = []
        epoch_test_targets = []

        with torch.no_grad():
            for x1, x2, y, _ in val_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)

                output = model(x1, x2)
                val_loss = criterion(output, y)
                total_val_loss += val_loss.item()
                all_val_preds.append(output.cpu())
                all_val_targets.append(y.cpu())

        with torch.no_grad():
            for x1, x2, y, orig_idx in test_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)

                output = model(x1, x2)
                test_loss = criterion(output, y)
                total_test_loss += test_loss.item()
                all_test_preds.append(output.cpu())
                all_test_targets.append(y.cpu())

                # 收集当前epoch的测试数据
                epoch_test_original_indices.extend(orig_idx)
                epoch_test_predictions.append(output.cpu().numpy())
                epoch_test_targets.append(y.cpu().numpy())

        with torch.no_grad():
            for x1, x2, y, orig_idx in all_data_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)
                output = model(x1, x2)
                all_data_preds.append(output.cpu())
                all_data_targets.append(y.cpu())

                # 收集当前epoch的数据
                epoch__original_indices.extend(orig_idx)
                epoch__predictions.append(output.cpu().numpy())
                epoch__targets.append(y.cpu().numpy())

        all_val_preds = torch.cat(all_val_preds, dim=0).numpy()
        all_val_targets = torch.cat(all_val_targets, dim=0).numpy()
        all_test_preds = torch.cat(all_test_preds, dim=0).numpy()
        all_test_targets = torch.cat(all_test_targets, dim=0).numpy()

        r2_val_scores = r2_score(all_val_targets, all_val_preds)
        r2_test_scores = r2_score(all_test_targets, all_test_preds)

        # 记录日志
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_test_loss = total_test_loss / len(test_loader)
        # print(len(train_loader), len(val_loader), len(test_loader))
        logger.log_epoch(avg_train_loss, avg_val_loss, avg_test_loss, r2_val_scores, r2_test_scores)

        print(f'[Train] Epoch [{epoch + 1}/{NUM_EPOCHS}], '
              f'Train Loss: {avg_train_loss:.6f}, '
              f'Validation Loss: {avg_val_loss:.6f}, '
              f'Test Loss: {avg_test_loss:.6f}, '
              f'R² val Score: {r2_val_scores:.6f}',
              f'R² test Score: {r2_test_scores:.6f}')

        # 保存模型
        each_val_loss = avg_val_loss
        each_model_state = model.state_dict().copy()
        each_epoch = epoch

        torch.save({
            'model_state_dict': each_model_state,
            'epoch': each_epoch,
            'val_loss': each_val_loss
        }, f'checkpoint_teva_YLL052C_epoch{each_epoch+1}.pth')

        # 保存当前epoch的数据
        if epoch__original_indices:
            # 将列表转换为numpy数组
            epoch__predictions = np.concatenate(epoch__predictions)
            epoch__targets = np.concatenate(epoch__targets)

            # 创建结果DataFrame并保存为CSV
            results_df = pd.DataFrame({
                'Original_Index': epoch__original_indices,
                'Predicted_Value': epoch__predictions,
                'True_Label': epoch__targets
            })
            results_df.to_csv(f'{test_data_dir}/all_results_epoch_{epoch + 1}.csv', index=False)
            print(f"已保存第 {epoch + 1} 轮的测试数据")

        # 保存当前epoch的测试数据
        if epoch_test_original_indices:
            # 将列表转换为numpy数组
            epoch_test_predictions = np.concatenate(epoch_test_predictions)
            epoch_test_targets = np.concatenate(epoch_test_targets)

            # 创建结果DataFrame并保存为CSV
            results_df = pd.DataFrame({
                'Original_Index': epoch_test_original_indices,
                'Predicted_Value': epoch_test_predictions,
                'True_Label': epoch_test_targets
            })
            results_df.to_csv(f'{test_data_dir}/test_results_epoch_{epoch + 1}.csv', index=False)
            print(f"已保存第 {epoch + 1} 轮的测试数据")

    # 绘制训练曲线并保存日志
    logger.save_logs()

    return model

if __name__ == '__main__':
    model = main()
    print('1')
