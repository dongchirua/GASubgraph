import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, GATv2Conv, Linear, SAGEConv


class GAT(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels,
                 heads, 
                 dropout = 0.5, **kwargs):
        super(GAT, self).__init__()


        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads=heads, 
                                dropout=dropout)
        self.conv2 = GATv2Conv(hidden_channels*heads, hidden_channels, heads=heads, 
                                dropout=dropout, concat=False)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, edge_index, batch):
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
        x = global_max_pool(x, batch)

        x = self.classifier(x)

        return x
