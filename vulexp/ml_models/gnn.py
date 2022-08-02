import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, SAGEConv, Linear
from pytorch_lightning import LightningModule


class GNN(LightningModule):
    name = 'GNN'

    def __init__(self, input_channel, hidden_channels, out_channels,
                 n_class, num_layers, normalize, dropout, **kwargs):
        super(GNN, self).__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(input_channel, hidden_channels, normalize=normalize))
        for layer in range(self.num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.lin = Linear(out_channels, n_class)

    def forward(self, x, edge_index, batch=None, **kwargs):

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = x.relu()

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin(x)

        return x
