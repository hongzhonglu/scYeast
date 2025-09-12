import pickle
import os
from matplotlib import pyplot as plt
import numpy as np
import warnings
import torch
import seaborn as sns
import pandas as pd
import time


def draw_dynamic_attn_graph():
    with open('final_dynamic_attn_100.pkl', 'rb') as f:
        dynamic_attns = pickle.load(f)

    final_attns = dynamic_attns.cpu().detach()
    plot = sns.heatmap(pd.DataFrame(final_attns))
    plt.show()
    max = torch.max(final_attns)
    min = torch.min(final_attns)
    normalized = final_attns - min
    final_attns_n = normalized / (max - min)
    greater_than_threshold = torch.gt(final_attns_n, 0.8)
    count = greater_than_threshold.sum()
    print(count.item())
    final_attns_n[final_attns_n <= 0.85] = 0
    data = pd.DataFrame(final_attns_n.detach())
    print(data.shape)
    data_s = data.iloc[0: 100, 0: 100]
    plot1 = sns.heatmap(data)
    plt.show()
    plot2 = sns.heatmap(data_s)
    plt.show()
    print(data)
    print(data_s)
    return 0


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 1000)
    np.set_printoptions(linewidth=400, threshold=200)  # np.inf表示正无穷
    os.chdir('../data')
    draw_dynamic_attn_graph()
    print('1')
