import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch.nn import Linear

class GCN(torch.nn.Module):
    def __init__(self, num_features=3, out_dim=512, num_hidden=1024, pool=global_mean_pool):
        super().__init__()
        self.outdim = out_dim
        self.conv1 = GCNConv(num_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, num_hidden)
        self.conv3 = GCNConv(num_hidden, out_dim)
        self.pool = pool
        self.out = Linear(out_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.pool(x, data.batch)
        return self.out(x)