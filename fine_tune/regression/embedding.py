import torch
from torch import nn
import numpy as np
import random
import math
import os
import sys
from gene2vec import read_gene2vec_result


class AutoDiscretizationEmbedding2(nn.Module):
    def __init__(self, dim, bin_num=10, bin_alpha=1.0, mask_token_id=-1, pad_token_id=0):
        super().__init__()

        self.dim = dim

        self.bin_num = bin_num
        self.bin_alpha = bin_alpha

        self.mlp = nn.Linear(1, self.bin_num)
        self.mlp2 = nn.Linear(self.bin_num, self.bin_num)
        self.LeakyReLU = nn.LeakyReLU(0.1)
        self.Softmax = nn.Softmax(dim=-1)
        self.emb = nn.Embedding(self.bin_num, self.dim)

        self.emb_mask = nn.Embedding(1, self.dim)
        self.emb_pad = nn.Embedding(1, self.dim)

        self.bin_num_idx = torch.tensor(range(self.bin_num))
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id

        self.tensor0 = torch.tensor(0, dtype=torch.long)

    def forward(self, x, output_weight=0):
        x = x.to(torch.float32)
        x_mask_idx = (x == self.mask_token_id).nonzero()
        x_pad_idx = (x == self.pad_token_id).nonzero()

        x = x.unsqueeze(-1)
        x = self.mlp(x)  # [B,N,1] -> [B,N,H]
        x = self.LeakyReLU(x)  # [B,N,H]
        x_crosslayer = self.mlp2(x)  # [B,N,H]
        x = self.bin_alpha * x + x_crosslayer  # [B,N,H]
        weight = self.Softmax(x)  # [B, N, H]

        bin_num_idx = self.bin_num_idx.to(x.device)  # [H,]

        token_emb = self.emb(bin_num_idx)  # [H, D]

        x = torch.matmul(weight, token_emb)  # [B, N, D]

        tensor0 = torch.tensor(0, dtype=torch.long, device=x.device)

        mask_token_emb = self.emb_mask(tensor0).to(x.device).type(x.dtype)

        x[x_mask_idx[:, 0], x_mask_idx[:, 1], :] = mask_token_emb.repeat(x_mask_idx.shape[0], 1)

        pad_token_emb = self.emb_pad(tensor0).to(x.device).type(x.dtype)
        x[x_pad_idx[:, 0], x_pad_idx[:, 1], :] = pad_token_emb.repeat(x_pad_idx.shape[0], 1)

        if output_weight:
            return x, weight
        return x


class Gene2VecPositionalEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        gene2vec_weight = read_gene2vec_result()
        gene2vec_weight = torch.from_numpy(gene2vec_weight)
        self.emb = nn.Embedding.from_pretrained(gene2vec_weight)

    def forward(self, x):
        t = torch.arange(x.shape[1], device=x.device)
        return self.emb(t)


class ExpressionTokenEmbedding(nn.Module):
    def __init__(self, dim, max_value):
        super().__init__()
        self.emb = nn.Embedding(max_value, dim, dtype=torch.float32)

    def forward(self, x):
        y = self.emb(x)
        return y


def random_mask(y):
    B, L = y.shape
    new_y = y.clone()
    mask_seqs = []
    for i in range(B):
        mask_number = math.ceil(L * 0.15)
        mask_squeue = random.sample(range(0, L - 1), mask_number)
        mask_seqs.append(mask_squeue)
        for j in range(len(mask_squeue)):
            new_y[i, mask_squeue[j]] = -1
    return new_y, mask_seqs


if __name__ == '__main__':
    os.chdir('../data')
    a, b = random_mask(torch.tensor([[[0,1,2,3,4,0,0,0,1,32,0,4,5,6,0]],[[0,1,2,3,4,0,0,0,1,32,0,4,5,6,0]],[[0,1,2,3,4,0,0,0,1,32,0,4,5,6,0]],[[0,1,2,3,4,0,0,0,1,32,0,4,5,6,0]]]), torch.tensor([0,1,1]))


    '''
    dim = 200
    max_seq_len = 5812
    max_value = 10
    token = ExpressionTokenEmbedding(dim, max_value)
    a = torch.tensor([[[1,2],[2,3]],[[3,4],[4,5]]])
    '''
