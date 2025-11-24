import sys
import torch
import os
import torch.nn as nn
sys.path.append('../model_construction')
from embedding import Gene2VecPositionalEmbedding, AutoDiscretizationEmbedding2
from DSgraph import DynamicEncoder, DynamicEncoderLayer, StaticEncoder, StaticEncoderLayer
from self_attention import FullAttention, AttentionLayer
from gated_fusion import GatedFusion



class DSgraphNetWO_Output(nn.Module):
    def __init__(self, output_attention=True, d_model=200, d_ff=800,
                 factor=5, n_heads=1, activation='ReLu', dropout=0):
        super(DSgraphNetWO_Output, self).__init__()
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


    def forward(self, x):
        # Embedding
        x1 = self.pos_embedding(x)
        x2 = self.token_embedding(x)
        x = x1 + x2

        # parallel
        Dynamic_out, Dynamic_attns = self.Dynamicencoder(x)
        Static_out, Static_attns = self.Staticencoder(x)

        # Gate
        y = self.gate(Dynamic_out, Static_out)

        return y



class DSgraphNet_WO_knowlegde(nn.Module):
    def __init__(self, output_attention=True, d_model=200, d_ff=800, max_value=80,
                 factor=5, n_heads=1, activation='ReLu', dropout=0):
        super(DSgraphNet_WO_knowlegde, self).__init__()
        self.output_attention = output_attention

        # Embedding
        self.pos_embedding = Gene2VecPositionalEmbedding()
        self.token_embedding = AutoDiscretizationEmbedding2(dim=d_model)

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


    def forward(self, x, attn_mask=None):

        # Embedding
        x1 = self.pos_embedding(x)
        x2 = self.token_embedding(x)
        x = x1 + x2

        # parallel
        Dynamic1_out, Dynamic1_attns = self.Dynamicencoder1(x)
        Dynamic2_out, Dynamic2_attns = self.Dynamicencoder2(x)

        # Gate
        y = self.gate(Dynamic1_out, Dynamic2_out)


        return y



class AdditionalNetwork(nn.Module):
    def __init__(self):
        super(AdditionalNetwork, self).__init__()
        self.conv1d1 = nn.Conv1d(in_channels=200, out_channels=1, kernel_size=1)
        self.conv1d2 = nn.Conv1d(in_channels=5812, out_channels=512, kernel_size=1)
        self.conv1d3 = nn.Conv1d(in_channels=512, out_channels=1, kernel_size=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d1(x)
        x = nn.ReLU()(x)
        x = x.transpose(1, 2)
        x = self.conv1d2(x)
        x = nn.ReLU()(x)
        x = self.conv1d3(x)
        x = nn.Sigmoid()(x)
        return x.squeeze()


class CombinedNet(nn.Module):
    def __init__(self, net1, net2):
        super(CombinedNet, self).__init__()
        self.net1 = net1
        self.net2 = net2
        for param in self.net1.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.net1(x)
        x = self.net2(x)
        return x


if __name__ == '__main__':
    os.chdir('../data')
    torch.set_default_dtype(torch.float32)
    device = 'cuda'
    net = DSgraphNetWO_Output().to(device).to(torch.float32)
    modified_state_dict = net.state_dict()
    addition_net = AdditionalNetwork().to(device).to(torch.float32)
    checkpoint = torch.load('final_checkpoint_spearman0.5_w_knowledge_huber_v4.pth')
    checkpoint_state_dict = checkpoint['net']
    for name, param in checkpoint_state_dict.items():
        if name in modified_state_dict:
            modified_state_dict[name].copy_(param)
    net.load_state_dict(modified_state_dict)
    net.eval()
    combined_model = CombinedNet(net, addition_net)
    a = torch.randn((1, 5812), device=device, dtype=torch.float32)
    y = combined_model(a)
    print(1)