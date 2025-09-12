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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_full_dataset():
    full_data = pd.read_csv('processed_yeast9_pnas_data_ori.csv')
    features = full_data.iloc[:, :-1]
    labels = full_data.iloc[:, -1]
    gene_names = features.columns.tolist()
    return features, labels, gene_names


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


def load_model():
    net1 = DSgraphNetWO_Output().to(device).to(torch.float32)
    net1.eval()
    net2 = AdditionalNetwork().to(device).to(torch.float32)
    model = CombinedNet(net1, net2)
    model.load_state_dict(torch.load('grow_finetune_best_model.pth', map_location=device))
    model.eval()
    return model


def gene_knockout_analysis(model, features, gene_names, save_path='gene_knockout_impact.csv'):
    features_tensor = torch.tensor(features.values, dtype=torch.float32)
    original_outputs = []
    for i in range(len(features_tensor)):
        with torch.no_grad():
            out = model(features_tensor[i:i + 1].to(device)).detach().cpu().item()
            original_outputs.append(out)
    original_outputs = np.array(original_outputs)
    results = []
    for i, gene in enumerate(gene_names):
        mean_delta = 0
        for j in range(len(features_tensor)):
            modified = features_tensor[j].clone()
            modified[i] = 0.0
            with torch.no_grad():
                y_pred = model(modified.unsqueeze(0).to(device)).cpu().item()
            delta = y_pred - original_outputs[j]
            mean_delta += delta
        mean_delta /= len(features_tensor)
        results.append((gene, mean_delta))
        print(f"Gene {gene}: Mean ΔGrowth = {mean_delta:.12f}")
    pd.DataFrame(results, columns=['Gene', 'MeanGrowthChange']).to_csv(save_path, index=False)
    print(f"\n敲除结果已保存至: {save_path}")


if __name__ == '__main__':
    os.chdir('../data')
    model = load_model()
    features, labels, gene_names = load_full_dataset()
    gene_knockout_analysis(model, features, gene_names)
