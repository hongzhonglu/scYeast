import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    def __init__(self, D):
        super(GatedFusion, self).__init__()
        self.fn1 = nn.Linear(in_features=D, out_features=D, bias=True, dtype=torch.float32)
        self.fn2 = nn.Linear(in_features=D, out_features=D, bias=True, dtype=torch.float32)
        self.fn3 = nn.Linear(in_features=D, out_features=D, bias=True, dtype=torch.float32)
        self.sigmoid = nn.Sigmoid()

    def forward(self, HS, HT):
        XS = self.fn1(HS)
        XT = self.fn2(HT)
        z = torch.add(XS, XT)
        z = self.sigmoid(z)
        H = torch.add(torch.multiply(z, HS), torch.multiply(1 - z, HT))
        H = self.fn3(H)
        return H

### **代码解析：`gated_fusion.py` 内容架构**
#
# #### **1. 模块依赖**
# - **PyTorch**：使用 `torch` 和 `torch.nn` 定义神经网络层及张量操作。
#
# ---
#
# #### **2. 核心类 `GatedFusion`**
# - **功能**：通过门控机制动态融合两个输入特征（`HS` 和 `HT`），生成综合特征表示。
#
# ---
#
# #### **3. 类初始化 `__init__`**
# - **参数**：
#   - `D`：输入特征的维度（`HS` 和 `HT` 的维度需相同）。
# - **组件**：
#   1. **线性变换层**：
#      - `fn1`, `fn2`, `fn3`：三个独立的 `nn.Linear` 层，输入输出维度均为 `D`。
#   2. **激活函数**：
#      - `sigmoid`：Sigmoid 函数，生成门控信号（范围 `[0, 1]`）。
#
# ---
#
# #### **4. 前向传播 `forward`**
# - **输入**：
#   - `HS`：源特征 1（形状 `[B, L, D]` 或 `[B, D]`）。
#   - `HT`：源特征 2（形状需与 `HS` 一致）。
# - **流程**：
#   1. **特征变换**：
#      - `XS = fn1(HS)`：对 `HS` 进行线性变换。
#      - `XT = fn2(HT)`：对 `HT` 进行线性变换。
#   2. **门控信号生成**：
#      - `z = XS + XT`：将两个变换后的特征相加。
#      - `z = sigmoid(z)`：生成门控权重（范围 `[0, 1]`）。
#   3. **特征融合**：
#      - `H = z * HS + (1 - z) * HT`：动态加权融合原始输入。
#   4. **最终变换**：
#      - `H = fn3(H)`：对融合结果进行最终线性变换。
# - **输出**：
#   - `H`：融合后的特征（形状与输入一致）。
#
# ---
#
# #### **5. 核心机制**
# - **门控加权**：
#   - 通过 `Sigmoid` 生成权重 `z`，控制 `HS` 和 `HT` 的贡献比例。
#   - 公式：
#     \[
#     H = \sigma(W_1 \cdot HS + W_2 \cdot HT) \odot HS + (1 - \sigma(W_1 \cdot HS + W_2 \cdot HT)) \odot HT
#     \]
#   - 其中，\( W_1, W_2, W_3 \) 为线性层参数，\( \sigma \) 为 Sigmoid 函数。
#
# ---
#
# #### **6. 应用场景**
# - **多源特征融合**：例如结合静态编码器（先验知识）和动态编码器（数据驱动）的输出。
# - **模型灵活性**：自适应调整不同特征的权重，适用于需动态权衡多来源信息的任务（如基因关系预测、多模态融合）。
#
# ---
#
# #### **7. 代码架构图**
# ```plaintext
# GatedFusion
# ├─ fn1: Linear(D → D)       # 变换 HS
# ├─ fn2: Linear(D → D)       # 变换 HT
# ├─ sigmoid: Sigmoid()       # 生成门控信号 z
# ├─ fn3: Linear(D → D)       # 最终特征变换
# └─ forward(HS, HT):
#    ├─ XS = fn1(HS)
#    ├─ XT = fn2(HT)
#    ├─ z = Sigmoid(XS + XT)
#    ├─ H = z * HS + (1-z) * HT
#    └─ H = fn3(H)
# ```#