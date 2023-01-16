import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor

from torch_geometric.nn import GINConv, global_max_pool


class GIN(torch.nn.Module):
    def __init__(self,
                 input_channel: int,
                 hidden_channels: int,
                 num_layers: int,
                 out_channels: int = 1,
                 dropout: float = 0.3, **kwargs):
        super(GIN, self).__init__()

        self.dropout = dropout

        convs_list = [
            GINConv(nn.Sequential(
                nn.Linear(input_channel, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU())
            )
        ]
        convs_list += [
            GINConv(nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU())
            )
            for _ in range(num_layers - 1)
        ]

        self.convs = nn.ModuleList(convs_list)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self,
                x: LongTensor,
                edge_index: LongTensor,
                batch: LongTensor) -> FloatTensor:
        for i in range(len(self.convs)):
            x = self.convs[i](x, edge_index)

        x = global_max_pool(x, batch)

        x = self.classifier(x)

        return x
