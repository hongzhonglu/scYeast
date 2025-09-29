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
    with open("../old_data/scaled_data_1.pkl", "rb") as f:
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

def read_scaled_data_pro(batch_size):
    with open("../tmp3-24/scaled_data_1_pro.pkl", "rb") as f:
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
    read_scaled_data_pro(10000)
    read_relation_data()

# ### **代码解析**
#
# ---
#
# #### **1. 功能概述**
# 该代码用于 **数据处理与加载**，主要功能包括：
# - 将预处理后的文本数据转换为序列化的缩放数据文件。
# - 加载缩放数据并划分为训练集和测试集，生成 PyTorch 数据加载器。
# - 读取基因关系矩阵文件，构建关系矩阵供模型使用。
#
# ---
#
# #### **2. 核心类与函数解析**
#
# ##### **2.1 `MyDataLoader` 类**
# - **功能**：自定义 PyTorch 数据集类，用于封装数据并提供标准接口。
# - **方法**：
#   - `__init__`: 初始化数据集长度和数据张量。
#   - `__getitem__`: 返回指定索引的数据样本。
#   - `__len__`: 返回数据集总长度。
#
# ##### **2.2 `save_processed_data()`**
# - **功能**：
#   从 `processed_data.txt` 读取原始数据，进行缩放和格式转换后保存为 `.pkl` 文件。
# - **实现细节**：
#   - **数据读取**：跳过首行（假设 `Q0045` 为列名），将每行数据转换为浮点数并缩放（`×100` 后四舍五入）。
#   - **数据保存**：使用 `pickle` 将处理后的数据保存到 `scaled_data.pkl`。
# - **关键假设**：
#   - `processed_data.txt` 的格式为：首行为列名（`Q0045` 开头），后续行为数值数据。
#
# ##### **2.3 `read_scaled_data(batch_size)`**
# - **功能**：
#   加载序列化的缩放数据，划分为训练集和测试集，并生成 PyTorch 数据加载器。
# - **流程**：
#   1. **加载数据**：从 `scaled_data.pkl` 读取数据，转为 PyTorch 张量。
#   2. **数据划分**：按 8:2 比例划分训练集和测试集。
#   3. **生成 DataLoader**：封装为 `MyDataLoader` 对象，设置批次大小和是否打乱顺序。
# - **内存管理**：使用 `del` 显式释放不再使用的变量（如 `data`, `x_train`, `x_test`）。
#
# ##### **2.4 `read_relation_data()`**
# - **功能**：
#   从 `relation_matrix.txt` 读取基因关系矩阵，转换为 NumPy 数组。
# - **实现细节**：
#   - **跳过首行**：假设首行为列名（`Q0080` 开头）。
#   - **数据处理**：移除每行首列（基因名称），将剩余值转为浮点数列表。
#   - **返回结果**：NumPy 数组形式的关系矩阵（形状 `[num_genes, num_genes]`）。
#
# ---
#
# #### **3. 关键设计细节**
# - **数据缩放**：
#   - 原始数据通过 `×100` 和四舍五入进行缩放，可能用于整数化处理（如分类任务）。
# - **数据划分**：
#   - 使用 `train_test_split` 默认随机划分，测试集占比 20%。
# - **关系矩阵格式**：
#   - 假设 `relation_matrix.txt` 的格式为每行首列为基因名称，后续为关联标记（`0.0` 或 `1.0`）。
#
# ---
#
# #### **4. 代码架构图**
# ```plaintext
# data_processing.py
# ├─ MyDataLoader(Dataset)
# │  ├─ __init__: 初始化数据集
# │  ├─ __getitem__: 获取样本
# │  └─ __len__: 返回数据集长度
# │
# ├─ save_processed_data()
# │  └─ 读取 processed_data.txt → 缩放 → 保存为 scaled_data.pkl
# │
# ├─ read_scaled_data(batch_size)
# │  └─ 加载 scaled_data.pkl → 划分训练/测试集 → 生成 DataLoader
# │
# ├─ read_relation_data()
# │  └─ 读取 relation_matrix.txt → 构建关系矩阵（NumPy）
# │
# └─ 主程序
#    └─ 切换目录并调用函数（需取消注释执行）
# ```
#
# ---
#
# #### **5. 应用场景**
# - **模型训练**：通过 `DataLoader` 提供批量化数据输入，适用于 PyTorch 模型训练。
# - **关系网络建模**：`read_relation_data` 生成的关系矩阵可用于图神经网络（GNN）的先验知识。
# - **数据标准化**：缩放和序列化处理便于数据的快速加载和复用。
#
# ---
#
# #### **6. 输出文件示例**
# **`scaled_data.pkl` 内容**：
# - 二维列表，每行对应一个样本的缩放后特征（整数格式）。
#
# **`relation_matrix.txt` 格式**：
# ```plaintext
# GeneA   0.0     1.0     1.0
# GeneB   1.0     0.0     0.0
# GeneC   1.0     0.0     0.0
# ```