from math import sqrt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape

        scale = self.scale or 1. / sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys).to(torch.float32)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        if attn_mask != None:
            scores.masked_fill_(attn_mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        V = torch.einsum("bhls,bshd->blhd", A, values).to(torch.float32)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads, dtype=torch.float32)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads, dtype=torch.float32)
        self.value_projection = nn.Linear(d_model, d_values * n_heads, dtype=torch.float32)
        self.out_projection = nn.Linear(d_values * n_heads, d_model, dtype=torch.float32)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


# ### **代码解析**
#
# ---
#
# #### **1. 功能概述**
# 该代码实现了 **Transformer 多头注意力机制**，包含因果掩码（Causal Mask）功能，适用于自回归建模任务（如时间序列预测、文本生成）。核心组件包括：
# - **因果掩码生成**：防止模型访问未来信息。
# - **完整注意力计算**：支持掩码、缩放点积注意力及多头处理。
# - **注意力层封装**：将输入投影到多头子空间并整合结果。
#
# ---
#
# #### **2. 核心类与函数解析**
#
# ##### **2.1 `TriangularCausalMask`（因果掩码类）**
# - **功能**：生成上三角布尔掩码矩阵，用于遮挡未来位置的信息。
# - **参数**：
#   - `B`：批次大小。
#   - `L`：序列长度。
#   - `device`：设备（如 CPU/GPU）。
# - **关键方法**：
#   ```python
#   self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)
#   ```
#   - 生成形状为 `[B, 1, L, L]` 的上三角矩阵，对角线及以上为 `True`（表示需掩码的位置）。
#   - 示例（`L=3`）：
#     ```plaintext
#     [[[[False,  True,  True],
#        [False, False,  True],
#        [False, False, False]]]]
#     ```
#
# ##### **2.2 `FullAttention`（完整注意力类）**
# - **功能**：实现缩放点积注意力计算，支持因果掩码。
# - **参数**：
#   - `mask_flag`：是否启用因果掩码。
#   - `scale`：缩放因子（默认 `1/sqrt(dim)`）。
#   - `output_attention`：是否返回注意力权重矩阵。
# - **前向传播流程**：
#   1. **计算注意力分数**：
#      ```python
#      scores = torch.einsum("blhe,bshe->bhls", queries, keys)
#      ```
#      - 输入 `queries` 和 `keys` 形状为 `[B, L, H, E]` 和 `[B, S, H, E]`，输出 `scores` 形状为 `[B, H, L, S]`。
#   2. **应用因果掩码**：
#      - 若 `mask_flag=True`，将未来位置的分数设为 `-inf`，使 softmax 后权重趋近于零。
#   3. **计算注意力权重**：
#      ```python
#      A = torch.softmax(scale * scores, dim=-1)
#      ```
#   4. **加权聚合值向量**：
#      ```python
#      V = torch.einsum("bhls,bshd->blhd", A, values)
#      ```
#      - 输出 `V` 形状为 `[B, L, H, D]`，合并多头后为 `[B, L, H*D]`。
#
# ##### **2.3 `AttentionLayer`（注意力层类）**
# - **功能**：封装多头注意力机制，包括输入投影、注意力计算和输出投影。
# - **参数**：
#   - `attention`：注意力计算模块（如 `FullAttention`）。
#   - `d_model`：输入特征维度。
#   - `n_heads`：注意力头数。
# - **关键步骤**：
#   1. **线性投影**：
#      - `queries`、`keys`、`values` 分别通过独立线性层投影到 `[B, L, H, d_keys]` 和 `[B, S, H, d_values]`。
#   2. **调整形状**：
#      ```python
#      queries.view(B, L, H, -1)  # 形状 [B, L, H, d_keys]
#      ```
#   3. **调用注意力模块**：
#      - 返回整合后的输出 `[B, L, d_model]` 和注意力权重（若 `output_attention=True`）。
#
# ---
#
# #### **3. 关键设计细节**
# - **因果掩码**：确保自回归建模时仅依赖历史信息，避免信息泄露。
# - **多头注意力**：将特征拆分为多个头并行计算，增强模型表达能力。
# - **缩放点积**：通过 `scale=1/sqrt(dim)` 防止点积数值过大导致梯度不稳定。
# - **爱因斯坦求和**：使用 `einsum` 高效处理多维张量运算。
#
# ---
#
# #### **4. 代码架构图**
# ```plaintext
# Attention 模块
# ├─ TriangularCausalMask
# │  └─ 生成上三角掩码矩阵（遮挡未来位置）
# │
# ├─ FullAttention
# │  ├─ 计算缩放点积注意力分数
# │  ├─ 应用因果掩码（可选）
# │  ├─ 计算 Softmax 注意力权重
# │  └─ 聚合 Value 向量
# │
# └─ AttentionLayer
#    ├─ 输入投影（Query/Key/Value 线性层）
#    ├─ 调整形状为多头
#    ├─ 调用 FullAttention
#    └─ 输出投影（合并多头结果）
# ```
#
# ---
#
# #### **5. 应用场景**
# - **自回归模型**：如 GPT 系列、时间序列预测模型。
# - **序列生成任务**：文本生成、语音合成、音乐生成。
# - **需因果约束的场景**：任何要求模型仅依赖历史信息的任务。