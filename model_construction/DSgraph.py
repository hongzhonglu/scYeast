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

    def forward(self, x, attn_mask=None):
        Q = x
        K = x
        V = x
        A = torch.tensor(self.relation, dtype=torch.float32).reshape(1, 5812, 5812).to('cuda')
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