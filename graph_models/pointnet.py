# Largely from https://colab.research.google.com/drive/1D45E5bUK3gQ40YpZo65ozs7hg5l-eo_U?usp=sharing#scrollTo=TkV7DDBxLmUR

import torch
import torch.nn.functional as F
from torch_cluster import knn_graph
from torch_geometric.nn import global_max_pool

from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import MessagePassing


class PointNetLayer(MessagePassing):
    def __init__(self, in_channels, out_channels):
        # Message passing with "max" aggregation.
        super(PointNetLayer, self).__init__('max')
        
        # Initialization of the MLP:
        # Here, the number of input features correspond to the hidden node
        # dimensionality plus point dimensionality (=3).
        self.mlp = Sequential(Linear(in_channels + 3, out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels))
        
    def forward(self, h, pos, edge_index):
        # Start propagating messages.
        return self.propagate(edge_index, h=h, pos=pos)
    
    def message(self, h_j, pos_j, pos_i):
        # h_j defines the features of neighboring nodes as shape [num_edges, in_channels]
        # pos_j defines the position of neighboring nodes as shape [num_edges, 3]
        # pos_i defines the position of central nodes as shape [num_edges, 3]

        input = pos_j - pos_i  # Compute spatial relation.

        if h_j is not None:
            # In the first layer, we may not have any hidden node features,
            # so we only combine them in case they are present.
            input = torch.cat([h_j, input], dim=-1)

        return self.mlp(input)  # Apply our final MLP.


class PointNet(torch.nn.Module):
    def __init__(self, num_features=3, out_dim=512, num_hidden=1024, pool=global_max_pool):
        super(PointNet, self).__init__()

        torch.manual_seed(12345)
        self.conv1 = PointNetLayer(num_features, num_hidden)
        self.conv2 = PointNetLayer(num_hidden, num_hidden)
        self.out = Linear(num_hidden, out_dim)
        self.pool = pool
        self.outdim = out_dim
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 3. Start bipartite message passing.
        h = self.conv1(h=x, pos=x, edge_index=edge_index)
        h = h.relu()
        h = self.conv2(h=h, pos=x, edge_index=edge_index)
        h = h.relu()

        # 4. Global Pooling.
        h = self.pool(h, data.batch)  # [num_examples, hidden_channels]
        
        # 5. Linear Out.
        return self.out(h)