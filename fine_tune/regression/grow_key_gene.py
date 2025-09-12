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


if __name__ == '__main__':
    os.chdir('../data')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = 'cuda'
    with open('relation_matrix.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == 'Q0045':
                name_list = row
                break
    train_data, val_data, test_data = read_grow_fine_tune_data(1, 'processed_yeast9_pnas_data_ori.csv', 0, 0)
    torch.set_default_dtype(torch.float32)
    net = DSgraphNetWO_Output().to(device).to(torch.float32)
    addition_net = AdditionalNetwork().to(device).to(torch.float32)
    combined_model = CombinedNet(net, addition_net)
    combined_model.load_state_dict(torch.load('grow_finetune_best_model.pth'))
    combined_model.eval()
    importances = []
    for batch_id, batch in enumerate(train_data):
        x = batch['features'].to(device)
        x.requires_grad_(True)
        y_out = combined_model(x)
        y_out.sum().backward()
        grad = x.grad.mean(dim=0).detach().cpu().numpy()
        importances.append(grad)
    mean_importance = np.mean(importances, axis=0)
    importance_df = pd.DataFrame({
        'Feature': name_list,
        'Importance': mean_importance
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    importance_df.to_csv('feature_importance.csv', index=False)
    torch.cuda.empty_cache()