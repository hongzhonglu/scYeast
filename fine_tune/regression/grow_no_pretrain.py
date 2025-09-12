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
sys.path.append('../model_construction')
# from get_age_io import CustomDataset
from sklearn.metrics import r2_score


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.values, dtype=torch.int)
        self.labels = torch.tensor(labels.tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'features': self.features[idx], 'label': self.labels[idx]}
        return sample


class CustomTestDataset(Dataset):
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
        self.conv1d1 = nn.Conv1d(in_channels=5812, out_channels=512, kernel_size=1)
        self.dropout1 = nn.Dropout(p=dropout_prob)
        self.conv1d2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1)
        self.dropout2 = nn.Dropout(p=dropout_prob)
        self.conv1d3 = nn.Conv1d(in_channels=256, out_channels=1, kernel_size=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = x.unsqueeze(0)
        x = x.transpose(1, 2)
        x = self.conv1d1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.conv1d2(x)
        x = self.gelu(x)
        x = self.dropout2(x)
        x = self.conv1d3(x)
        return x.squeeze(0).squeeze(0)


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
    # os.chdir('../data')
    csv_file = 'grow_fine_tune_without_pretrain.csv'
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['NO', 'R2', 'PCC', 'test mean error'])

    device = 'cuda'
    for i in range(20):
        train_data, val_data, test_data = read_grow_fine_tune_data(5, 'processed_yeast9_pnas_data_ori.csv', 0.2, 0.5)
        torch.set_default_dtype(torch.float32)
        addition_net = AdditionalNetwork().to(device).to(torch.float32)
        loss_func = nn.MSELoss(reduction='mean')
        optimizer = optim.AdamW(addition_net.parameters(), lr=0.001, weight_decay=0.01)
        max_epoch = 24
        max_r2 = -np.inf
        best_model = None
        best_exp_index = -1
        for epoch in range(max_epoch):
            addition_net.train()
            for batch_id, batch in enumerate(train_data):
                optimizer.zero_grad()
                x = batch['features'].to(device)
                y = batch['label'].to(device)
                y_out = addition_net(x)
                loss = loss_func(y_out, y)
                loss.backward()
                optimizer.step()
                # print('epoch=%d, batch_id=%d, loss=%f' % (epoch, batch_id, loss))
            addition_net.eval()
            val_loss = 0
            val_num = 0
            all_val_y = []
            all_val_y_out = []
            for batch_id, batch in enumerate(val_data):
                x = batch['features'].to(device)
                y = batch['label'].to(device)
                y_out = addition_net(x)
                loss = loss_func(y_out, y)
                val_loss += loss.item() * y.size(0)
                val_num += y.size(0)
                all_val_y.append(y.cpu().detach().numpy())
                all_val_y_out.append(y_out.cpu().detach().numpy())
            val_loss /= val_num
            all_val_y = np.concatenate(all_val_y)
            all_val_y_out = np.concatenate(all_val_y_out)
            r_squared = r2_score(all_val_y, all_val_y_out)
            pcc, _ = pearsonr(all_val_y, all_val_y_out)
            print(f"Validation R^2: {r_squared}, PCC: {pcc}, Validation Loss: {val_loss}")
            if r_squared >= max_r2:
                max_r2 = r_squared
                best_model = addition_net.state_dict()
                best_exp_index = epoch
        if best_model is not None:
            torch.save(best_model, 'grow_finetune_best_model' + str(i) + '.pth')
        addition_net.load_state_dict(best_model)
        addition_net.eval()
        test_loss = 0
        test_num = 0
        all_test_y = []
        all_test_y_out = []
        for batch_id, batch in enumerate(test_data):
            x = batch['features'].to(device)
            y = batch['label'].to(device)
            y_out = addition_net(x)
            loss = loss_func(y_out, y)
            test_loss += loss.item() * y.size(0)
            test_num += y.size(0)
            all_test_y.append(y.cpu().detach().numpy())
            all_test_y_out.append(y_out.cpu().detach().numpy())
        test_loss /= test_num
        all_test_y = np.concatenate(all_test_y)
        all_test_y_out = np.concatenate(all_test_y_out)
        r_squared = r2_score(all_test_y, all_test_y_out)
        pcc, _ = pearsonr(all_test_y, all_test_y_out)
        print(f"Test R^2: {r_squared}, PCC: {pcc}, Test Loss: {test_loss}")
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([r_squared, pcc, test_loss])
