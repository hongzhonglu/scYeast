import os
import pickle
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MyDataLoader(Dataset):

    def __init__(self, length, x):
        self.len = length
        self.x = x

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.len


def save_processed_data():
    data = []
    with open('processed_data.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[0] == 'Q0045':
                name_list = row
            else:
                data.append([round(float(x)*100) for x in row])
    with open("scaled_data.pkl", "wb") as f:
        pickle.dump(data, f)
    return 0


def read_scaled_data(batch_size):
    with open("scaled_data.pkl", "rb") as f:
        data = pickle.load(f)
    data = torch.tensor(data)
    x_train, x_test = train_test_split(data, test_size=0.2)
    del data, f
    train_loader = DataLoader(dataset=MyDataLoader(len(x_train), x_train),
                              batch_size=batch_size, shuffle=True)
    del x_train
    test_loader = DataLoader(dataset=MyDataLoader(len(x_test), x_test),
                             batch_size=batch_size, shuffle=True)
    del x_test
    return train_loader, test_loader


def read_relation_data():
    relation = []
    with open('relation_matrix.txt', 'r') as f:
        for line in f:
            row = line.strip().split()
            if row[1] == 'Q0080':
                continue
            else:
                row.pop(0)
                row_float = [float(x) for x in row]
                relation.append(row_float)
    relation = np.array(relation)
    return relation


if __name__ == '__main__':
    os.chdir('../data')
    # read_scaled_data(10000)
