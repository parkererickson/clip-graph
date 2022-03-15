import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch.nn import Linear, LogSoftmax, Dropout, ReLU

import torchvision.models as models
from transformers import ViTFeatureExtractor
from transformers import BertModel, BertConfig, BertTokenizer, AdamW

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
                 language_model="bert", 
                 graph_model="gcn", 
                 embedding_dim=512,
                 language_embedding_dim=786,
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
        if language_model == 'bert':  # TODO Make this work
            self.language_model_name = 'bert'
            self.language_model = BertModel.from_pretrained('bert-base-uncased')
            self.language_projection = Projection(language_embedding_dim, embedding_dim).double()
        else:
            raise ValueError("image_model must be either 'vit' or 'resnet'")

    def forward(self, data):
        graph_output = self.graph_model.forward(data)
        graph_emb = self.graph_projection(graph_output)
        
        language_output = self.language_model(tokens).last_hidden_state[:,0]
        language_emb = self.language_projection(language_output)
        
        language_emb = language_emb / language_emb.norm(dim=-1, keepdim=True)
        graph_emb = graph_emb / graph_emb.norm(dim=-1, keepdim=True)
        logits = language_emb @ graph_emb.T
        #out = F.softmax(logits, dim=-1)
        image_similarity = language_emb @ language_emb.T
        graph_similarity = graph_emb @ graph_emb.T
        target = F.softmax((language_emb + graph_similarity)/2, dim=-1)
        graph_loss = cross_entropy(logits, target)
        image_loss = cross_entropy(logits.T, target.T)
        loss = (graph_loss + image_loss)/2
        return loss.mean(), logits