import torch
import torch.nn as nn
import os
import torch.optim as optim
import sys
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import csv
from scipy.stats import pearsonr
from read_checkpoint import DSgraphNetWO_Output, DSgraphNet_WO_knowlegde
from sklearn.metrics import r2_score


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.values, dtype=torch.float32)
        self.labels = torch.tensor(labels.tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'features': self.features[idx], 'label': self.labels[idx]}
        return sample


class AdditionalNetwork(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super(AdditionalNetwork, self).__init__()
        self.conv1d1 = nn.Conv1d(in_channels=200, out_channels=1, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.conv1d2 = nn.Conv1d(in_channels=5812, out_channels=512, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.conv1d3 = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = x.transpose(1, 2)
        x = self.conv1d2(x)
        x = self.gelu(x)
        x = self.dropout2(x)
        x = self.conv1d3(x)
        return x.squeeze(-1)


class CombinedNet(nn.Module):
    def __init__(self, net1, net2):
        super(CombinedNet, self).__init__()
        self.net1 = net1
        self.net2 = net2
        for param in self.net1.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.net1(x)
        x = self.net2(x)
        return x.squeeze()


def read_grow_fine_tune_data(batch_size, data_file, test_size=0.2, val_size=0.5):
    age_dataset = pd.read_csv(data_file)
    features = age_dataset.iloc[:, :-1]
    labels = age_dataset.iloc[:, -1]
    if test_size > 0 and test_size < 1:
        features_train, features_temp, labels_train, labels_temp = train_test_split(features, labels, test_size=test_size)
        features_val, features_test, labels_val, labels_test = train_test_split(features_temp, labels_temp, test_size=val_size)
    elif test_size == 1:
        features_train, features_test = None, features
        labels_train, labels_test = None, labels
        features_val, labels_val = None, None
    else:
        features_train, features_test = features, None
        labels_train, labels_test = labels, None
        features_val, labels_val = None, None
    if features_train is not None and labels_train is not None:
        train_dataset = CustomDataset(features_train, labels_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    else:
        train_dataloader = None
    if features_val is not None and labels_val is not None:
        val_dataset = CustomDataset(features_val, labels_val)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        val_dataloader = None
    if features_test is not None and labels_test is not None:
        test_dataset = CustomDataset(features_test, labels_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        test_dataloader = None
    return train_dataloader, val_dataloader, test_dataloader


def load_split_data(split_id, batch_size=5, splits_dir='data_splits'):
    """
    加载指定的数据集划分
    """
    # 加载训练集
    train_data = pd.read_csv(f'{splits_dir}/train_split_{split_id}.csv')
    train_features = train_data.iloc[:, :-1]
    train_labels = train_data.iloc[:, -1]
    train_dataset = CustomDataset(train_features, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 加载验证集
    val_data = pd.read_csv(f'{splits_dir}/val_split_{split_id}.csv')
    val_features = val_data.iloc[:, :-1]
    val_labels = val_data.iloc[:, -1]
    val_dataset = CustomDataset(val_features, val_labels)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 加载测试集
    test_data = pd.read_csv(f'{splits_dir}/test_split_{split_id}.csv')
    test_features = test_data.iloc[:, :-1]
    test_labels = test_data.iloc[:, -1]
    test_dataset = CustomDataset(test_features, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == '__main__':
    # os.chdir('../data')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 初始化保存训练过程的CSV文件
    training_process_csv = 'w_training_process_data.csv'
    final_results_csv = 'w_final_results.csv'
    test_predictions_csv = 'w_test_predictions.csv'

    # 创建训练过程CSV文件头
    with open(training_process_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Experiment_ID', 'Epoch', 'Train_Loss', 'Val_Loss', 'Val_R2'])

    # 创建最终结果CSV文件头
    with open(final_results_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Experiment_ID', 'Best_Epoch', 'Test_R2', 'Test_Loss'])

    # 创建测试预测结果CSV文件头
    with open(test_predictions_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Experiment_ID', 'Sample_ID', 'True_Label', 'Predicted_Label'])

    device = 'cuda'
    splits_dir = 'data_splits'

    for i in range(1):
        # train_data, val_data, test_data = read_grow_fine_tune_data(5, 'processed_yeast9_pnas_data_ori.csv', 0.2, 0.5)
        train_data, val_data, test_data = load_split_data(i, batch_size=5, splits_dir=splits_dir)
        torch.set_default_dtype(torch.float32)
        # net = DSgraphNetWO_Output().to(device).to(torch.float32)
        net = DSgraphNet_WO_knowlegde().to(device).to(torch.float32)
        modified_state_dict = net.state_dict()
        checkpoint = torch.load('final_checkpoint_spearman0.5_wo_knowledge_huber_v4.pth')
        checkpoint_state_dict = checkpoint['net']
        for name, param in checkpoint_state_dict.items():
            if name in modified_state_dict:
                modified_state_dict[name].copy_(param)
        net.load_state_dict(modified_state_dict)
        net.eval()

        addition_net = AdditionalNetwork().to(device).to(torch.float32)
        combined_model = CombinedNet(net, addition_net)
        loss_func = nn.MSELoss(reduction='mean')
        optimizer = optim.AdamW(combined_model.parameters(), lr=0.0005, weight_decay=0.01)

        max_epoch = 24
        max_r2 = -np.inf
        best_model = None
        best_exp_index = -1
        for epoch in range(max_epoch):
            combined_model.train()
            train_loss = 0
            train_num = 0
            for batch_id, batch in enumerate(train_data):
                optimizer.zero_grad()
                x = batch['features'].to(device)
                y = batch['label'].to(device)
                y_out = combined_model(x)
                loss = loss_func(y_out, y)
                loss.backward()
                optimizer.step()
                # print('epoch=%d, batch_id=%d, loss=%f' % (epoch, batch_id, loss))
                train_loss += loss.item() * y.size(0)
                train_num += y.size(0)

            train_loss /= train_num

            combined_model.eval()
            val_loss = 0
            val_num = 0
            all_val_y = []
            all_val_y_out = []
            for batch_id, batch in enumerate(val_data):
                x = batch['features'].to(device)
                y = batch['label'].to(device)
                y_out = combined_model(x)
                loss = loss_func(y_out, y)
                val_loss += loss.item() * y.size(0)
                val_num += y.size(0)
                all_val_y.append(y.cpu().detach().numpy())
                all_val_y_out.append(y_out.cpu().detach().numpy())

            val_loss /= val_num
            all_val_y = np.concatenate(all_val_y)
            all_val_y_out = np.concatenate(all_val_y_out)
            r_squared = r2_score(all_val_y, all_val_y_out)

            print(
                f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val R^2: {r_squared:.4f}")

            # 保存训练过程数据
            with open(training_process_csv, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([i + 1, epoch + 1, train_loss, val_loss, r_squared])

            # 保存最佳模型
            if r_squared >= max_r2:
                max_r2 = r_squared
                best_model = combined_model.state_dict()
                best_exp_index = epoch

            # 保存最佳模型
        if best_model is not None:
            torch.save(best_model, f'grow_finetune_best_model_w_{i}.pth')

            if r_squared >= max_r2:
                max_r2 = r_squared
                best_model = combined_model.state_dict()
                best_exp_index = epoch

        combined_model.load_state_dict(best_model)
        combined_model.eval()
        test_loss = 0
        test_num = 0
        all_test_y = []
        all_test_y_out = []

        for batch_id, batch in enumerate(test_data):
            x = batch['features'].to(device)
            y = batch['label'].to(device)
            y_out = combined_model(x)
            loss = loss_func(y_out, y)
            test_loss += loss.item() * y.size(0)
            test_num += y.size(0)
            all_test_y.append(y.cpu().detach().numpy())
            all_test_y_out.append(y_out.cpu().detach().numpy())

        test_loss /= test_num
        all_test_y = np.concatenate(all_test_y)
        all_test_y_out = np.concatenate(all_test_y_out)
        test_r2 = r2_score(all_test_y, all_test_y_out)

        print(f"Test R^2: {test_r2:.4f}, Test Loss: {test_loss:.4f}")

        # 保存最终结果
        with open(final_results_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i + 1, best_exp_index + 1, test_r2, test_loss])

        # 保存测试集预测结果
        with open(test_predictions_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            for sample_id, (true_label, pred_label) in enumerate(zip(all_test_y, all_test_y_out)):
                writer.writerow([i + 1, sample_id, true_label, pred_label])

    torch.cuda.empty_cache()
    print("\n所有实验完成！")
    print(f"训练过程数据保存在: {training_process_csv}")
    print(f"最终结果保存在: {final_results_csv}")
    print(f"测试预测结果保存在: {test_predictions_csv}")

