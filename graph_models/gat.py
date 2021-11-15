import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch.nn import Linear

class GAT(torch.nn.Module):
    def __init__(self, num_features=3, out_dim=512, num_hidden=1024, pool=global_mean_pool):
        super(GAT, self).__init__()
        self.hid = num_hidden
        self.in_head = 8
        self.out_head = 1
        self.outdim = out_dim
        self.pool = pool
        self.conv1 = GATConv(num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, out_dim, concat=False,
                             heads=self.out_head, dropout=0.6)
        self.out = Linear(out_dim, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index        
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.pool(x, data.batch)
        return self.out(x)