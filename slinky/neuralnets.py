import torch
import torch.nn as nn
import torch.optim as optim

class Square(nn.Module):
    def forward(self,x):
        return torch.square(x)

class DenseBlock(nn.Module):
    def __init__(self, NeuronsPerLayer, NumLayer):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(NumLayer):
            layer.append(
                nn.Sequential(
                nn.Linear(NeuronsPerLayer * i + NeuronsPerLayer, NeuronsPerLayer), 
                nn.Softplus(beta=1e1)
                )
            )
        self.net = nn.Sequential(*layer)
    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=-1)
        return X
