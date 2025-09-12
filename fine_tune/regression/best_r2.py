import torch
import torch.nn as nn
import os
import torch.optim as optim
import sys
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.stats import pearsonr
sys.path.append('../fine_tune')
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


if __name__ == '__main__':
    os.chdir('../data')
    device = 'cuda'
    train_data, val_data, test_data = read_grow_fine_tune_data(1, 'processed_yeast9_pnas_data_ori.csv', 0, 0)
    torch.set_default_dtype(torch.float32)
    net = DSgraphNetWO_Output().to(device).to(torch.float32)
    addition_net = AdditionalNetwork().to(device).to(torch.float32)
    combined_model = CombinedNet(net, addition_net)
    pre_model = torch.load('grow_finetune_best_model.pth')
    combined_model.load_state_dict(pre_model)
    combined_model.eval()
    true_y = []
    predicted_y = []
    for batch_id, batch in enumerate(train_data):
        x = batch['features'].to(device)
        y = batch['label'].to(device)
        true_y.append(y.cpu().detach().numpy()[0])
        y_out = combined_model(x)
        predicted_y.append(y_out.cpu().detach().numpy())
        print(batch_id)
    r_squared = r2_score(true_y, predicted_y)
    plt.figure(figsize=(10, 8))
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 18})
    plt.scatter(true_y, predicted_y, color='orange', label='Predictions', alpha=0.6)
    plt.plot([min(true_y), max(true_y)], [min(true_y), max(true_y)], 'k--', label='Perfect Prediction')
    plt.text(0.5, 0.9, f'R² = {r_squared:.4f}', horizontalalignment='center', verticalalignment='center',
             transform=plt.gca().transAxes, fontsize=30, color='black')
    plt.xlabel('True Values', fontsize=16)
    plt.ylabel('Predicted Values', fontsize=16)
    plt.grid(False)
    plt.legend(fontsize=14)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    os.chdir('../figure')
    plt.savefig('grow_r2.svg', bbox_inches='tight')
    plt.show()
    print(1)