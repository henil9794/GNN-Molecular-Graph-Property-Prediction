import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, BatchNorm


class EdgeNetwork(nn.Module):
    """Maps edge features → weight matrix for NNConv."""
    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim),
        )

    def forward(self, edge_attr):
        return self.net(edge_attr)


class MPNN(nn.Module):
    def __init__(
        self,
        node_dim:   int,
        edge_dim:   int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout:    float = 0.1,
    ):
        super().__init__()

        # Project raw node features to hidden space
        self.node_emb = nn.Linear(node_dim, hidden_dim)

        # Stack of edge-conditioned message passing layers
        self.convs = nn.ModuleList()
        self.bns   = nn.ModuleList()
        for _ in range(num_layers):
            edge_net = EdgeNetwork(edge_dim, hidden_dim)
            self.convs.append(NNConv(hidden_dim, hidden_dim, edge_net, aggr="mean"))
            self.bns.append(BatchNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Graph-level readout MLP
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x          = data.x.float()
        edge_index = data.edge_index
        edge_attr  = data.edge_attr.float()
        batch      = data.batch

        # Node embedding
        x = F.relu(self.node_emb(x))

        # Message passing
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index, edge_attr)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)

        # Global pooling: aggregate all node embeddings → 1 vector per molecule
        x = global_mean_pool(x, batch)   # shape: [batch_size, hidden_dim]

        return self.readout(x)           # shape: [batch_size, 1]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)