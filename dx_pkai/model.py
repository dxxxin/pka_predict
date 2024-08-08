import torch
from torch.nn import Sequential, Conv2d, ReLU, MaxPool2d, Linear
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(nn.Module):
    def __init__(self, in_channels=16, out_channels=64, negative_slope=0.2, nheads=1):
        """ version of GAT."""
        super(GAT, self).__init__()
        self.attentions = GATConv(in_channels, out_channels, nheads, True, negative_slope=negative_slope)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight=None):
        x = self.relu(self.attentions(x, edge_index))
        return x


class Net(nn.Module):
    def __init__(self, in_channels=4008):
        super(Net, self).__init__()
        self.gat = GAT(in_channels=18, out_channels=48, negative_slope=0.2, nheads=1)
        self.linear = Linear(12000,4000)
        self.linear1 = Linear(4028, 800)
        self.linear2 = Linear(800, 400)
        self.linear3 = Linear(400, 200)
        self.linear4 = Linear(200, 1)
        self.dropout = torch.nn.Dropout()
        self.relu = nn.ReLU()
        # 前向过程

    def forward(self, x, edge_index, edge_weight, residue_onehot, pssm):
        x = self.gat(x, edge_index, edge_weight)
        x = x.view(-1)
        x = self.relu(self.linear(x))

        x = torch.cat((residue_onehot.view(-1), pssm, x))
        #x = torch.cat((residue_onehot, x))
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.relu(self.linear3(x))
        x = self.dropout(x)
        x = self.linear4(x)
        return x

