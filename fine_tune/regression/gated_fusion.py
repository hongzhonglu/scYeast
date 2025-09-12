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