import numpy as np
import pandas as pd
import torch
import os
import time
import torch.nn as nn
import torch.optim as optim

# import torch.distributed as dist
# import torch.multiprocessing as mp
# from torch.nn.parallel import DistributedDataParallel as DDP


from embedding_pro  import Gene2VecPositionalEmbedding, ExpressionTokenEmbedding, random_mask,AutoDiscretizationEmbedding2
from DSgraph_pro import DynamicEncoder, DynamicEncoderLayer, StaticEncoder, StaticEncoderLayer
from self_attention_pro import FullAttention, AttentionLayer, TriangularCausalMask
from gated_fusion import GatedFusion
from load_data import read_scaled_data_pro
import pickle
from torch.cuda.amp import autocast, GradScaler

with open("./tmp3-3/filter_proindex.pkl","rb") as f:
    proindex=torch.tensor(pickle.load(f))
def pad_and_mask(batch, index, total_genes=5812):
    """
    batch: [B, 1780] 蛋白质组数据
    index: [1780] 有效基因位置索引
    return:
        padded_data: [B, 5812]
        mask: [B, 5812] (True表示填充位置)
    """
    device = batch.device
    B = batch.size(0)
    padded_data = torch.full((B, total_genes),
                              fill_value=-2, 
                              dtype=batch.dtype,
                              device=device)
    padded_data[:, index] = batch

    mask = torch.ones((B, total_genes),
                      dtype=torch.bool,
                      device=device)
    mask[:, index] = False

    return padded_data,mask

class DSgraphNetWithoutMask(nn.Module):
    def __init__(self, output_attention=True, d_model=200, d_ff=800, max_value=8,
                 factor=5, n_heads=1, activation='ReLu', dropout=0):
        super(DSgraphNetWithoutMask, self).__init__()
        self.output_attention = output_attention

        # Embedding
        self.pos_embedding = Gene2VecPositionalEmbedding()
        self.token_embedding = AutoDiscretizationEmbedding2(d_model)

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
        self.output = nn.Linear(in_features=d_model, out_features=1, bias=True)

    def forward(self, x, attn_mask=None):
        
        # Embedding
        x1 = self.pos_embedding(x)
        x2 = self.token_embedding(x)
        x_nonmasked = x.clone().float()
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
    def __init__(self, output_attention=True, d_model=200, d_ff=800, max_value=800,
                 factor=5, n_heads=1, activation='ReLu', dropout=0.1):
        super(DSgraphNet, self).__init__()
        self.output_attention = output_attention

        # Embedding
        self.pos_embedding = Gene2VecPositionalEmbedding()
        self.token_embedding = AutoDiscretizationEmbedding2(dim=d_model)

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
        self.output1 = nn.Linear(in_features=d_model, out_features=100, bias=True)
        self.output2 = nn.Linear(in_features=100,out_features=1,bias=True)

    def forward(self, x, attn_mask=None):
        x_nonmasked = x.clone().float()
        masked_x,mask_seqs=random_mask(x)
        padded_x,promask = pad_and_mask(masked_x,proindex)
        promask = promask.unsqueeze(1).unsqueeze(1)
        # Embedding
        x1 = self.pos_embedding(padded_x)
        x2,x2_mask_idx,x2_zero_idx,x2_pad_idx = self.token_embedding(padded_x)
        #x2_masked, mask_seqs = random_mask(x2)
        x = x1 + x2

        # parallel
        Dynamic_out, Dynamic_attns = self.Dynamicencoder(x,promask)
        Static_out, Static_attns = self.Staticencoder(x,promask)

        # Gate
        y = self.gate(Dynamic_out, Static_out)

        # output
        y = self.output1(y)
        y = self.output2(y)
        
        y = y[:,proindex,:]

        return x_nonmasked, y, Dynamic_attns, mask_seqs,x2_mask_idx,x2_pad_idx


if __name__ == '__main__':
    os.chdir('./data')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
    # dist.init_process_group(backend="nccl")
    # local_rank = int(os.environ['LOCAL_RANK'])
    # world_size = dist.get_world_size() 
    device = torch.device(f"cuda:0")
    print(torch.cuda.is_available())
    print(torch.version.cuda)
    torch.set_default_dtype(torch.float32)
    net = DSgraphNet().to(device).to(torch.float32)
    # 加载旧权重（允许部分加载）
    # pretrained_dict = torch.load('../data/final_checkpoint_spearman0.5_w_knowledge_v4pro.pth')
    # model_dict = net.state_dict()

# 1. 过滤掉不可用的预训练参数
    # pretrained_dict = pretrained_dict["net"]
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

# 2. 更新当前模型参数
    # model_dict.update(pretrained_dict) 

# 3. 加载并允许严格模式关闭
    # net.load_state_dict(model_dict, strict=True)
    #net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    #train_loader, val_loader = read_scaled_data(16,local_rank,world_size)
    train_loader, val_loader = read_scaled_data_pro(4)
    loss_func = nn.HuberLoss(delta=1.0, reduction='none')
    optimizer = optim.AdamW(net.parameters(), lr=0.00005, weight_decay=0.01)
    scaler = GradScaler()
    max_epoch = 10
    for epoch in range(max_epoch):
        #train_loader.sampler.set_epoch(epoch)
        net.train()
        for batch_idx, (x) in enumerate(train_loader):
            start_time = time.time()
            with autocast():
                x = x.to(device)
                target_out, net_out, attn, mask_seqs ,mask_idx,pad_idx= net(x)
                net_out = net_out.squeeze(-1)
                loss_per_element = loss_func(net_out, target_out)
                mask = torch.ones_like(target_out)
                for i, mask_seq in enumerate(mask_seqs):
                    for idx in mask_seq:
                        mask[i, idx] = 0 
                masked_zero_mask = (target_out == 0).float() * (mask == 0).float()
                masked_zero_loss = loss_per_element * masked_zero_mask
                masked_zero_loss = masked_zero_loss.sum() / masked_zero_mask.sum() if masked_zero_mask.sum() > 0 else torch.tensor(0.0, device=loss_per_element.device)
                masked_nonzero_mask = (target_out != 0).float() * (mask == 0).float()
                masked_nonzero_loss = loss_per_element * masked_nonzero_mask
                masked_nonzero_loss = masked_nonzero_loss.sum() / masked_nonzero_mask.sum() if masked_nonzero_mask.sum() > 0 else torch.tensor(0.0, device=loss_per_element.device)
                nonzero_mask = (target_out != 0).float()
                valid_nonzero_mask = nonzero_mask * (mask == 1).float()
                nonzero_loss = loss_per_element * valid_nonzero_mask
                nonzero_loss = nonzero_loss.sum() / valid_nonzero_mask.sum() if valid_nonzero_mask.sum() > 0 else torch.tensor(0.0, device=loss_per_element.device)
                zero_mask = (target_out == 0).float()
                valid_zero_mask = zero_mask * (mask == 1).float()
                zero_loss = loss_per_element * valid_zero_mask
                zero_loss = zero_loss.sum() / valid_zero_mask.sum() if valid_zero_mask.sum() > 0 else torch.tensor(0.0, device=loss_per_element.device)
                loss = 0.6 * masked_nonzero_loss + 0.3 * nonzero_loss + 0.4 * masked_zero_loss + 0.1 * zero_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            end_time = time.time()
            with open("output_pro_nopretrain.txt","a") as f:
                f.write(f"Epoch {epoch} | Batch {batch_idx} | Time {end_time-start_time:.3f}s | Total Loss {loss.item():.6f} | Mask Zero Loss {masked_zero_loss.item():.6f} |Mask Nonzero loss {masked_nonzero_loss.item():.6f} | Zero Loss {zero_loss.item():.6f} | Nonzero Loss {nonzero_loss.item():.6f}\n")

            print(f"Epoch {epoch} | Batch {batch_idx} | Time {end_time-start_time:.3f}s | Total Loss {loss.item():.6f} | Mask Zero Loss {masked_zero_loss.item():.6f} |Mask Nonzero loss {masked_nonzero_loss.item():.6f} | Zero Loss {zero_loss.item():.6f} | Nonzero Loss {nonzero_loss.item():.6f}")
        net.eval()        
        with torch.no_grad():
            for batch_idx, (x) in enumerate(val_loader):
                start_time = time.time()
                x = x.to(device)
                target_out, net_out, attn, mask_seqs ,mask_idx,pad_idx= net(x)
                net_out = net_out.squeeze(-1)
                loss_per_element = loss_func(net_out, target_out)
                mask = torch.ones_like(target_out)
                for i, mask_seq in enumerate(mask_seqs):
                    for idx in mask_seq:
                        mask[i, idx] = 0 
                masked_zero_mask = (target_out == 0).float() * (mask == 0).float()
                masked_zero_loss = loss_per_element * masked_zero_mask
                masked_zero_loss = masked_zero_loss.sum() / masked_zero_mask.sum() if masked_zero_mask.sum() > 0 else torch.tensor(0.0, device=loss_per_element.device)
                masked_nonzero_mask = (target_out != 0).float() * (mask == 0).float()
                masked_nonzero_loss = loss_per_element * masked_nonzero_mask
                masked_nonzero_loss = masked_nonzero_loss.sum() / masked_nonzero_mask.sum() if masked_nonzero_mask.sum() > 0 else torch.tensor(0.0, device=loss_per_element.device)
                nonzero_mask = (target_out != 0).float()
                valid_nonzero_mask = nonzero_mask * (mask == 1).float()
                nonzero_loss = loss_per_element * valid_nonzero_mask
                nonzero_loss = nonzero_loss.sum() / valid_nonzero_mask.sum() if valid_nonzero_mask.sum() > 0 else torch.tensor(0.0, device=loss_per_element.device)
                zero_mask = (target_out == 0).float()
                valid_zero_mask = zero_mask * (mask == 1).float()
                zero_loss = loss_per_element * valid_zero_mask
                zero_loss = zero_loss.sum() / valid_zero_mask.sum() if valid_zero_mask.sum() > 0 else torch.tensor(0.0, device=loss_per_element.device)
                loss = 0.6 * masked_nonzero_loss + 0.3 * nonzero_loss + 0.4 * masked_zero_loss + 0.1 * zero_loss
                end_time = time.time()
                print(f"Validation | Epoch {epoch} | Batch {batch_idx} | Time {end_time - start_time:.3f}s | Total Loss {loss.item():.6f} | Mask Zero Loss {masked_zero_loss.item():.6f} |Mask Nonzero loss {masked_nonzero_loss.item():.6f} | Zero Loss {zero_loss.item():.6f} | Nonzero Loss {nonzero_loss.item():.6f}")
                with open("output_pro_nopretrain.txt","a") as f:
                    f.write(f"Validation | Epoch {epoch} | Batch {batch_idx} | Time {end_time-start_time:.3f}s | Total Loss {loss.item():.6f} | Mask Zero Loss {masked_zero_loss.item():.6f} |Mask Nonzero loss {masked_nonzero_loss.item():.6f} | Zero Loss {zero_loss.item():.6f} | Nonzero Loss {nonzero_loss.item():.6f}\n")
        state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, f'checkpoint_epoch_{epoch}_spearman0.5_w_knowledge_v4pro_nopretrain.pth')
   
    state = {'net': net.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': max_epoch}
    torch.save(state, 'final_checkpoint_spearman0.5_w_knowledge_v4pro_nopretrain.pth')

