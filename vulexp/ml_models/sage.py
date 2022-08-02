from typing import Optional
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, SAGEConv, GINConv


class SAGE(torch.nn.Module):

    def __init__(self, in_channels, 
                    hidden_channels, 
                    num_layers, 
                    normalize: bool = True,
                    reduce_channels: Optional[int] = -1,
                    out_channels: int = 1,
                    dropout: float = 0.5, **kwargs):

        super(SAGE, self).__init__()
        self.dropout = dropout

        convs_list = [
            SAGEConv(in_channels, hidden_channels, normalize=normalize)
        ]

        if reduce_channels < 0:
            convs_list += [
                SAGEConv(hidden_channels, hidden_channels, normalize=normalize)
                for _ in range(num_layers - 1)
            ]
            self.convs = nn.ModuleList(convs_list)

            self.classifier = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(hidden_channels, out_channels)
            )

        else:
            convs_list += [
                SAGEConv(hidden_channels, hidden_channels, normalize=normalize)
                for _ in range(num_layers - 2)
            ]
            convs_list += [ 
                SAGEConv(hidden_channels, reduce_channels, normalize=normalize)
            ]
            self.convs = nn.ModuleList(convs_list)

            self.classifier = nn.Sequential(
                nn.Linear(reduce_channels, reduce_channels),
                nn.ReLU(),
                nn.Dropout(self.dropout),
                nn.Linear(reduce_channels, out_channels)
            )

    def forward(self, x, edge_index, batch=None, **kwargs):
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        return x
