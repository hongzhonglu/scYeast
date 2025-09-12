import torch
import torch.nn as nn
import os
import torch.optim as optim
import sys
import csv
import time
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from load_data import read_scaled_data
from DSgraph_main import DSgraphNetWithoutMask


if __name__ == '__main__':
    os.chdir('../data')
    csv_file = 'DSGraph_valid.csv'
    torch.set_default_dtype(torch.float32)
    device = 'cuda'
    net = DSgraphNetWithoutMask().to(device).to(torch.float32)
    pre_model = torch.load('final_checkpoint_200_10.pth')
    net.load_state_dict(pre_model['net'])
    net.eval()
    _, test_loader = read_scaled_data(10)
    loss_func = nn.MSELoss(reduction='mean')
    for batch_idx, (x) in enumerate(test_loader):
        start_time = time.time()
        x = x.to(device)
        target_out, net_out, attn = net(x)
        loss = loss_func(net_out, target_out)
        end_time = time.time()
        print('Tested    batch_id=%d    using time=%f    loss=%f'
              % (batch_idx, end_time - start_time, loss.item()))
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([batch_idx, loss.item()])

