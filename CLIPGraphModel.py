import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch.nn import Linear, LogSoftmax, Dropout, ReLU

import torchvision.models as models
from transformers import ViTFeatureExtractor

import torchvision.models as models

from graph_models.gat import GAT
from graph_models.gcn import GCN
from graph_models.pointnet import PointNet

def cross_entropy(logits, target, reduce='none'):
    ls = LogSoftmax(dim=-1)
    loss = (-target * ls(logits)).sum(dim=1)
    if reduce == "none":
        return loss
    elif reduce == "mean":
        return loss.mean()

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
                 graph_hidden_dim=256, 
                 graph_out_dim=256,
                 graph_pool="max", 
                 linear_proj_dropout=0.1,
                 pretrained_image_model=True):
        super().__init__()
        if graph_model == "gcn":
            if graph_pool == "mean":
                self.graph_model = GCN(num_hidden=graph_hidden_dim, out_dim=graph_out_dim, pool=global_mean_pool).double()
            elif graph_pool == "max":
                self.graph_model = GCN(num_hidden=graph_hidden_dim, out_dim=graph_out_dim, pool=global_max_pool).double()
            else:
                raise ValueError("graph_pool must be either 'mean' or 'max'")
        elif graph_model == "gat":
            if graph_pool == "mean":
                self.graph_model = GAT(num_hidden=graph_hidden_dim, out_dim=graph_out_dim, pool=global_mean_pool).double()
            elif graph_pool == "max":
                self.graph_model = GAT(num_hidden=graph_hidden_dim, out_dim=graph_out_dim, pool=global_max_pool).double()
            else:
                raise ValueError("graph_pool must be either 'mean' or 'max'")
        elif graph_model == "pn":
            if graph_pool == "mean":
                self.graph_model = PointNet(num_hidden=graph_hidden_dim, out_dim=graph_out_dim, pool=global_mean_pool).double()
            elif graph_pool == "max":
                self.graph_model = PointNet(num_hidden=graph_hidden_dim, out_dim=graph_out_dim, pool=global_max_pool).double()
            else:
                raise ValueError("graph_pool must be either 'mean' or 'max'")
        else:
            raise ValueError("graph_model must be either 'gcn', 'gat', or 'pn'")
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