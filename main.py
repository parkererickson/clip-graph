import dataset
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import CLIPGraphModel as cgm
import torch
import wandb

#wandb.init(project="CLIP-Graph-Model", entity="parkererickson")

config = wandb.config
config.epochs = 100
config.batch_size = 24
config.learning_rate = 5e-4
config.image_model="resnet"
config.graph_model="gat"
config.embedding_dim=128
config.graph_pool="max"
config.linear_proj_dropout=0.1
config.pretrained_image_model=False
config.lr_patience = 5
config.T_0 = 10

#data_sample = "sample"
data_sample = "2014-05-14-13-59-05"

ds = dataset.GraphCLIP(lidar_timestamp_path='./data/'+data_sample+'/lms_front.timestamps', 
                       image_timestamp_path='./data/'+data_sample+'/stereo.timestamps',
                       lidar_path='./data/'+data_sample+'/lms_front/',
                       image_path='./data/'+data_sample+'/stereo/centre/',
                       ins_path='./data/'+data_sample+'/gps/ins.csv',
                       data_name=data_sample,
                       use_cache=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train, val = torch.utils.data.random_split(ds, [int(0.8 * len(ds)), (len(ds)-int(0.8 * len(ds)))])
train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=config.batch_size, shuffle=True)

model = cgm.CLIPGraphModel(image_model=config.image_model,
                           graph_model=config.graph_model,
                           embedding_dim=config.embedding_dim,
                           graph_pool=config.graph_pool,
                           linear_proj_dropout=config.linear_proj_dropout,
                           pretrained_image_model=config.pretrained_image_model)

model.to(device)

#wandb.watch(model)

def train(model, loader, optimizer, epoch):
    for data in loader:
        data.to(device)
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()
    print('Epoch: {:02d}, Loss: {:.4f}'.format(epoch, loss.item()))
    #wandb.log({'train_loss': loss.item()}, step=epoch)

def valid(model, loader, epoch):
    model.eval()
    with torch.no_grad():
        for data in loader:
            data.to(device)
            loss = model(data)
    print('Epoch: {:02d}, Validation Loss: {:.4f}'.format(epoch, loss.item()))
    #wandb.log({'val_loss': loss.item()}, step=epoch)
    model.train()
    return loss.item()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0)
for epoch in range(config.epochs):
    train(model, train_loader, optimizer, epoch)
    valid_loss = valid(model, val_loader, epoch)
    scheduler.step(valid_loss)
#    wandb.log({'learning_rate': get_lr(optimizer)})
