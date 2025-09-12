import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import ast


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features.values, dtype=torch.int)
        self.labels = torch.tensor(labels.tolist(), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'features': self.features[idx], 'label': self.labels[idx]}
        return sample


if __name__ == '__main__':
    os.chdir('../data')
    data = []
    with open('processed_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == 'Q0045':
                name_list = row
            break
    age_dataset = pd.read_csv('divide2_age_data.csv')
    positions = []
    for i in range(len(age_dataset.keys())):
        if age_dataset.keys()[i] == 'Gene':
            for j in range(len(name_list)):
                if name_list[j] in age_dataset[age_dataset.keys()[i]].values:
                    position = age_dataset[age_dataset['Gene'] == name_list[j]].index[0]
                    positions.append(position)
                else:
                    positions.append(None)
        else:
            alignment_data = []
            for j in range(len(positions)):
                if positions[j] is not None:
                    alignment_data.append(age_dataset[age_dataset.keys()[i]][positions[j]])
                else:
                    alignment_data.append(0)
            if age_dataset.keys()[i][0] == 'A':
                alignment_data.append([1, 0, 0])
            elif age_dataset.keys()[i][0] == 'B':
                alignment_data.append([0, 1, 0])
            elif age_dataset.keys()[i][0] == 'C':
                alignment_data.append([0, 0, 1])
            data.append(alignment_data)
    df = pd.DataFrame(data)
    df.to_csv('alignment_age_data_with_labels.csv', index=False)
    df = pd.read_csv('alignment_age_data_with_labels.csv')
    features = df.iloc[:, :-1]
    labels = df.iloc[:, -1]
    labels = np.array([ast.literal_eval(item) for item in labels])
    dataset = CustomDataset(features, labels)
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    for batch_id, batch in enumerate(dataloader):
        print(batch['features'].shape)
        print(batch['label'].shape)
        print(batch_id)
        break
    print(1)