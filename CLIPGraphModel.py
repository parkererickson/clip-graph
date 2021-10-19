import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch.nn import Linear, LogSoftmax, Dropout, ReLU

import torchvision.models as models
from transformers import ViTFeatureExtractor

import torchvision.models as models

def cross_entropy(logits, target, reduce='none'):
    ls = LogSoftmax(dim=-1)
    loss = (-target * ls(logits)).sum(dim=1)
    if reduce == "none":
        return loss
    elif reduce == "mean":
        return loss.mean()

class GAT(torch.nn.Module):
    def __init__(self, num_features=3, out_dim=512, pool=global_mean_pool):
        super(GAT, self).__init__()
        self.hid = 8
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

class Projection(torch.nn.Module):
    def __init__(self, embedding_dim, projection_dim, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.dropout = dropout
        self.projection = Linear(embedding_dim, projection_dim)
        self.dropout = Dropout(dropout)
        self.relu = ReLU()
        self.fc = Linear(projection_dim, projection_dim)

    def forward(self, x):
        x = self.projection(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc(x)
        return x

class CLIPGraphModel(torch.nn.Module):
    def __init__(self,
                 image_model="vit", 
                 graph_model="gcn", 
                 embedding_dim=512, 
                 graph_pool="max", 
                 linear_proj_dropout=0.1,
                 pretrained_image_model=True):
        super().__init__()
        if graph_model == "gcn":
            if graph_pool == "mean":
                self.graph_model = GCN(pool=global_mean_pool).double()
            elif graph_pool == "max":
                self.graph_model = GCN(pool=global_max_pool).double()
            else:
                raise ValueError("graph_pool must be either 'mean' or 'max'")
        elif graph_model == "gat":
            if graph_pool == "mean":
                self.graph_model = GAT(pool=global_mean_pool).double()
            elif graph_pool == "max":
                self.graph_model = GAT(pool=global_max_pool).double()
            else:
                raise ValueError("graph_pool must be either 'mean' or 'max'")
        else:
            raise ValueError("graph_model must be either 'gcn' or 'gat'")
        self.graph_projection = Projection(self.graph_model.outdim, embedding_dim, dropout=linear_proj_dropout).double()
        if image_model == 'vit':  # TODO Make this work
            self.image_model_name = 'vit'
            self.image_model = ViTFeatureExtractor(model_name='vit')
            self.image_projection = Projection(embedding_dim, embedding_dim).double()
        elif image_model == 'resnet':
            self.image_model_name = 'resnet'
            self.image_model = models.resnet18(pretrained=pretrained_image_model)
            self.image_model = torch.nn.Sequential(*list(self.image_model.children())[:-1])
            self.image_projection = Projection(512, embedding_dim, dropout=linear_proj_dropout).double()
        else:
            raise ValueError("image_model must be either 'vit' or 'resnet'")

    def forward(self, data):
        graph_output = self.graph_model.forward(data)
        graph_emb = self.graph_projection(graph_output)
        images = data.image.view(data.num_graphs, 3, 224, 224)
        if self.image_model_name == 'vit': # TODO Make this work (Output)
            images = torch.split(images, 1, dim=0)
            image_output = [self.image_model(image.squeeze(), return_tensors='pt')['pixel_values'][0] for image in images]
            print(len(image_output))
            print(image_output[0].shape)
            print(image_output[0])
            image_output = self.image_model(images[0].squeeze(), return_tensors='pt')
            image_emb = self.image_projection(image_output)
        elif self.image_model_name == 'resnet': 
            image_output = self.image_model(images).flatten(start_dim=1).double()
            image_emb = self.image_projection(image_output)

        logits = image_emb @ graph_emb.T
        image_similarity = image_emb @ image_emb.T
        graph_similarity = graph_emb @ graph_emb.T
        target = F.softmax((image_similarity + graph_similarity)/2, dim=-1)
        graph_loss = cross_entropy(logits, target)
        image_loss = cross_entropy(logits.T, target.T)
        loss = (graph_loss + image_loss)/2
        return loss.mean()