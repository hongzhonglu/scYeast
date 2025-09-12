import torch
from torch import nn
import numpy as np
import random
import math
import os
import sys
from gene2vec import read_gene2vec_result


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
    B, L, _ = y.shape
    new_y = y.clone()
    for i in range(B):
        mask_number = math.ceil(L * 0.15)
        mask_squeue = random.sample(range(0, L - 1), mask_number)
        for j in range(len(mask_squeue)):
            new_y[i, mask_squeue[j], :] = 0
    return new_y


if __name__ == '__main__':
    os.chdir('../data')
    '''
    dim = 200
    max_seq_len = 5812
    max_value = 10
    token = ExpressionTokenEmbedding(dim, max_value)
    a = torch.tensor([[[1,2],[2,3]],[[3,4],[4,5]]])
    '''
