import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from load_data import read_relation_data


class StaticEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(StaticEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.relation = read_relation_data()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model, dtype=torch.float32)
        self.norm2 = nn.LayerNorm(d_model, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, dtype=torch.float32)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, dtype=torch.float32)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask):
        Q = x
        K = x
        V = x
        A = torch.tensor(self.relation, dtype=torch.float32).reshape(1, 5812, 5812).to('cuda')
        attn_mask = attn_mask[0]
        if attn_mask != None:
            A.masked_fill_(attn_mask, 0.0)
        x_temp = torch.einsum("bik,bij->bij", A, V).to(torch.float32)

        new_x = x_temp
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = self.norm2(x + y)
        attn = None

        return y, attn


class StaticEncoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(StaticEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DynamicEncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(DynamicEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.d_model = d_model
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, dtype=torch.float32)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, dtype=torch.float32)
        self.norm1 = nn.LayerNorm(d_model, dtype=torch.float32)
        self.norm2 = nn.LayerNorm(d_model, dtype=torch.float32)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # Q = x.permute(0, 2, 1)
        # K = x.permute(0, 2, 1)
        # V = x.permute(0, 2, 1)
        Q = x
        K = x
        V = x

        new_x, attn = self.attention(
            Q, K, V,
            attn_mask=attn_mask
        )
        # new_x = new_x.permute(0, 2, 1)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = self.norm2(x + y)

        return y, attn


class DynamicEncoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(DynamicEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns



# 以下是代码的详细解析：
#
# ---
#
# ### **1. 代码结构与依赖**
# - **依赖库**：
#   - `numpy`, `pandas`：基础数据处理。
#   - `torch`：PyTorch 深度学习框架。
#   - `torch.nn`：神经网络模块（如卷积层、归一化层）。
#   - `load_data.read_relation_data`：自定义函数，用于加载预定义的关系矩阵。
#
# ---
#
# ### **2. 核心类解析**
# #### **2.1 `StaticEncoderLayer`（静态编码器层）**
# - **功能**：处理固定关系矩阵的编码层。
# - **初始化参数**：
#   - `attention`：未直接使用（可能是占位符）。
#   - `d_model`：输入特征的维度。
#   - `d_ff`：前馈网络隐藏层维度（默认 `4 * d_model`）。
#   - `relation`：通过 `read_relation_data()` 加载的预定义关系矩阵（形状 `(5812, 5812)`）。
# - **前向传播逻辑**：
#   1. **关系矩阵融合**：
#      - 使用 `torch.einsum("bik,bij->bij", A, V)` 将关系矩阵 `A` 与输入 `V` 融合。
#      - 公式等价于：`A @ V`，其中 `A` 是批次化的关系矩阵（扩展为 `(1, 5812, 5812)`）。
#   2. **残差连接与归一化**：
#      - 通过 `x + self.dropout(new_x)` 实现残差连接。
#      - 使用 `LayerNorm` 进行归一化。
#   3. **前馈网络（FFN）**：
#      - 包含两个 `Conv1d` 层，通过激活函数（ReLU/GELU）实现非线性变换。
#
# #### **2.2 `StaticEncoder`（静态编码器堆叠）**
# - **功能**：将多个 `StaticEncoderLayer` 堆叠，构建深层编码器。
# - **前向传播**：
#   - 依次通过各编码器层，可选是否添加卷积层（`conv_layers`）。
#   - 最终输出归一化（若 `norm_layer` 存在）。
#
# #### **2.3 `DynamicEncoderLayer`（动态编码器层）**
# - **功能**：通过注意力机制动态学习关系的编码层。
# - **与前静态层的区别**：
#   - 未使用预定义关系矩阵，而是通过 `self.attention` 动态计算注意力权重。
#   - 注意力机制接受 `Q, K, V` 作为输入（默认 `Q=K=V=x`）。
# - **前向传播逻辑**：
#   1. **动态注意力计算**：
#      - 调用 `self.attention(Q, K, V)` 生成新的表示 `new_x` 和注意力权重 `attn`。
#   2. **残差连接与归一化**：与静态层类似。
#
# #### **2.4 `DynamicEncoder`（动态编码器堆叠）**
# - **功能**：堆叠多个 `DynamicEncoderLayer`，结构与 `StaticEncoder` 类似。
#
# ---
#
# ### **3. 关键设计细节**
# #### **3.1 静态与动态编码器的对比**
# | **特性**               | **StaticEncoder**                     | **DynamicEncoder**                   |
# |-------------------------|---------------------------------------|---------------------------------------|
# | 关系矩阵来源           | 预定义关系矩阵 (`read_relation_data`) | 通过注意力机制动态计算               |
# | 适用场景               | 已知固定关系（如基因相互作用网络）    | 需自适应学习关系（如文本序列建模）   |
# | 实现复杂度             | 低（直接使用预定义矩阵）              | 高（需计算注意力权重）               |
#
# #### **3.2 爱因斯坦求和（`einsum`）的作用**
# 在 `StaticEncoderLayer` 中：
# ```python
# x_temp = torch.einsum("bik,bij->bij", A, V)
# ```
# - **等价操作**：对批次中的每个样本，将关系矩阵 `A` 与值矩阵 `V` 相乘。
# - **维度解释**：
#   - `A`：形状为 `(1, 5812, 5812)`，表示批次化的关系矩阵。
#   - `V`：形状为 `(B, L, D)`（`B` 是批次大小，`L` 是序列长度，`D` 是特征维度）。
#   - 输出 `x_temp`：形状与 `V` 相同，表示通过关系矩阵加权后的新表示。
#
# ---
#
# ### **4. 潜在问题与改进**
# #### **4.1 静态编码器的局限性**
# - **关系矩阵固定**：若关系矩阵未能覆盖所有潜在关系，可能导致模型灵活性不足。
# - **改进建议**：结合静态与动态编码（如混合注意力机制）。
#
# #### **4.2 动态编码器的计算开销**
# - **注意力计算复杂度**：若序列长度 `L` 较大，动态注意力计算复杂度为 `O(L^2)`。
# - **改进建议**：使用稀疏注意力或分块计算优化。
#
# #### **4.3 数据维度对齐**
# - **硬编码关系矩阵**：`A` 的形状固定为 `(5812, 5812)`，需确保输入数据的基因/节点数量匹配。
# - **改进建议**：动态调整关系矩阵尺寸或添加适配层。
#
# ---
#
# ### **5. 代码功能总结**
# - **静态编码器**：利用预定义关系矩阵（如基因共现网络）增强特征表示，适用于关系已知的任务。
# - **动态编码器**：通过注意力机制自适应学习输入数据的关系，适用于关系未知或复杂的场景。
# - **应用场景**：基因表达预测、图节点分类、序列建模等需要关系建模的任务。