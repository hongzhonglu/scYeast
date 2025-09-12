import numpy as np
import pandas as pd
import torch
import os
import time
import torch.nn as nn
import torch.optim as optim
from embedding import Gene2VecPositionalEmbedding, ExpressionTokenEmbedding, random_mask
from DSgraph import DynamicEncoder, DynamicEncoderLayer, StaticEncoder, StaticEncoderLayer
from self_attention import FullAttention, AttentionLayer, TriangularCausalMask
from gated_fusion import GatedFusion
from load_data import read_scaled_data
import pickle
import csv
from torch.cuda.amp import autocast, GradScaler


class DSgraphNetWithoutMask(nn.Module):
    def __init__(self, output_attention=True, d_model=200, d_ff=200, max_value=800,
                 factor=5, n_heads=1, activation='ReLu', dropout=0):
        super(DSgraphNetWithoutMask, self).__init__()
        self.output_attention = output_attention

        # Embedding
        self.pos_embedding = Gene2VecPositionalEmbedding()
        self.token_embedding = ExpressionTokenEmbedding(d_model, max_value)

        # Dynamic
        self.Dynamicencoder = DynamicEncoder(
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

        # Static
        self.Staticencoder = StaticEncoder(
            [
                StaticEncoderLayer(
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
        x_nonmasked = x1 + x2
        x = x1 + x2

        # parallel
        Dynamic_out, Dynamic_attns = self.Dynamicencoder(x)
        Static_out, Static_attns = self.Staticencoder(x)

        # Gate
        y = self.gate(Dynamic_out, Static_out)

        # output
        y = self.output(y)

        return x_nonmasked, y, Dynamic_attns


class DSgraphNet(nn.Module):
    def __init__(self, output_attention=True, d_model=200, d_ff=200, max_value=800,
                 factor=5, n_heads=1, activation='ReLu', dropout=0):
        super(DSgraphNet, self).__init__()
        self.output_attention = output_attention

        # Embedding
        self.pos_embedding = Gene2VecPositionalEmbedding()
        self.token_embedding = ExpressionTokenEmbedding(d_model, max_value)

        # Dynamic
        self.Dynamicencoder = DynamicEncoder(
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

        # Static
        self.Staticencoder = StaticEncoder(
            [
                StaticEncoderLayer(
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
        Dynamic_out, Dynamic_attns = self.Dynamicencoder(x)
        Static_out, Static_attns = self.Staticencoder(x)

        # Gate
        y = self.gate(Dynamic_out, Static_out)

        # output
        y = self.output(y)

        return x_nonmasked, y, Dynamic_attns


if __name__ == '__main__':
    current_dir = os.path.abspath(os.getcwd())
    os.chdir('../data')
    device = 'cuda'
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    torch.set_default_dtype(torch.float32)
    net = DSgraphNet().to(device).to(torch.float32)
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
            if epoch == max_epoch-1:
                attn[0] = attn[0].to(torch.float32)
                batch_size, _, _, gene_numbers = attn[0].shape
                with torch.no_grad():
                    for i in range(batch_size):
                        temp_attns = torch.zeros((5812, 5812)).to(device).to(torch.float32)
                        for j in range(gene_numbers):
                            max = torch.max(attn[0][i][0][j])
                            min = torch.min(attn[0][i][0][j])
                            if max == min:
                                coff = 1
                                temp_attns[j] = (attn[0][i][0][j] - min + 0.5) * coff
                            else:
                                coff = 1 / (max - min)
                                temp_attns[j] = (attn[0][i][0][j] - min) * coff
                        final_attns = final_attns + temp_attns
                        sample_count = sample_count + 1
            if (batch_idx + 1) % checkpoint_step == 0:
                state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, 'checkpoint_spearman0.5.pth')
                with open("dynamic_attn_spearman0.5.pkl", "wb") as f:
                    pickle.dump(final_attns, f)
            end_time = time.time()
            print('Trained    epoch=%d    batch_id=%d    using time=%f    loss=%f'
                  % (epoch, batch_idx, end_time-start_time, loss*accumulation_steps))
    state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': max_epoch}
    torch.save(state, 'final_checkpoint_spearman0.5.pth')
    with torch.no_grad():
        final_attns = final_attns / sample_count
    with open("final_dynamic_attn_spearman0.5.pkl", "wb") as f:
        pickle.dump(final_attns, f)
    print(sample_count)
    net.eval()
    csv_file = 'DSGraph_valid_spearman0.5.csv'
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