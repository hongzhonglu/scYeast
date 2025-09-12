import numpy as np
import pandas as pd
import torch
import os
import csv
import time
import torch.nn as nn
import torch.optim as optim
from embedding import Gene2VecPositionalEmbedding, ExpressionTokenEmbedding, random_mask
from DSgraph import DynamicEncoder, DynamicEncoderLayer, StaticEncoder, StaticEncoderLayer
from self_attention import FullAttention, AttentionLayer, TriangularCausalMask
from gated_fusion import GatedFusion
from load_data import read_scaled_data
import pickle
from torch.cuda.amp import autocast, GradScaler


class DSgraphNet_without_knowlegde(nn.Module):
    def __init__(self, output_attention=True, d_model=200, d_ff=200, max_value=800,
                 factor=5, n_heads=1, activation='ReLu', dropout=0):
        super(DSgraphNet_without_knowlegde, self).__init__()
        self.output_attention = output_attention

        # Embedding
        self.pos_embedding = Gene2VecPositionalEmbedding()
        self.token_embedding = ExpressionTokenEmbedding(d_model, max_value)

        # Dynamic1
        self.Dynamicencoder1 = DynamicEncoder(
            [
                DynamicEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model=d_model, n_heads=n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Dynamic2
        self.Dynamicencoder2 = DynamicEncoder(
            [
                DynamicEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=output_attention), d_model=d_model, n_heads=n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                )
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        # Gate
        self.gate = GatedFusion(d_model)

        # Output
        self.output = nn.Linear(in_features=d_model, out_features=d_model, bias=True)

    def forward(self, x, attn_mask=None):

        # Embedding
        x1 = self.pos_embedding(x)
        x2 = self.token_embedding(x)
        x2_masked = random_mask(x2)
        x_nonmasked = x1 + x2
        x = x1 + x2_masked

        # parallel
        Dynamic1_out, Dynamic1_attns = self.Dynamicencoder1(x)
        Dynamic2_out, Dynamic2_attns = self.Dynamicencoder2(x)

        # Gate
        y = self.gate(Dynamic1_out, Dynamic2_out)

        # output
        y = self.output(y)

        return x_nonmasked, y, Dynamic1_attns


if __name__ == '__main__':
    current_dir = os.path.abspath(os.getcwd())
    os.chdir(current_dir + '/data')
    device = 'cuda'
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    torch.set_default_dtype(torch.float32)
    net = DSgraphNet_without_knowlegde().to(device).to(torch.float32)
    train_loader, test_loader = read_scaled_data(10)
    loss_func = nn.MSELoss(reduction='mean')
    optimizer = optim.AdamW(net.parameters(), lr=0.001, weight_decay=0.01)
    scaler = GradScaler()
    accumulation_steps = 4
    final_attns = torch.zeros((5812, 5812)).to(device).to(torch.float32)
    sample_count = 0
    max_epoch = 10
    checkpoint_step = 100
    for epoch in range(max_epoch):
        for batch_idx, (x) in enumerate(train_loader):
            start_time = time.time()
            with autocast():
                x = x.to(device)
                target_out, net_out, attn = net(x)
                loss = loss_func(net_out, target_out)
                loss = loss / accumulation_steps
            scaler.scale(loss).backward()
            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            if (batch_idx + 1) % checkpoint_step == 0:
                state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, 'checkpoint_without_knowledge.pth')
            end_time = time.time()
            print('Trained    epoch=%d    batch_id=%d    using time=%f    loss=%f'
                  % (epoch, batch_idx, end_time-start_time, loss*accumulation_steps))
    state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': max_epoch}
    torch.save(state, 'final_checkpoint_without_knowledge.pth')
    net.eval()
    csv_file = 'DSGraph_without_knowledge_valid.csv'
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