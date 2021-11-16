import dataset
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import CLIPGraphModel as cgm
import torch
import wandb

wandb.init(project="CLIP-Graph-Model-Final", entity="parkererickson")

config = wandb.config
config.epochs = 400           # CONSTANT
config.batch_size = 48        # CONSTANT 
config.learning_rate = 5e-4   # CONSTANT
config.image_model = "resnet" # CONSTANT
config.graph_model = "pn"
config.embedding_dim = 256    # CONSTANT
config.graph_pool = "max"     # CONSTANT
config.graph_hidden_dim = 64  # CONSTANT
config.graph_out_dim = 256    # CONSTANT
config.linear_proj_dropout = 0.1  # CONSTANT
config.pretrained_image_model = False
config.lr_patience = 5        # CONSTANT
config.T_0 = 10               # CONSTANT
config.data_sample = "2014-07-14-14-49-50"
#config.data_sample = "sample"

data_sample = config.data_sample

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
                           graph_hidden_dim=config.graph_hidden_dim,
                           graph_out_dim=config.graph_out_dim,
                           linear_proj_dropout=config.linear_proj_dropout,
                           pretrained_image_model=config.pretrained_image_model)

model.to(device)

wandb.watch(model)

def train(model, loader, optimizer, epoch):
    for data in loader:
        data.to(device)
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()
    return loss.item()

def valid(model, loader):
    model.eval()
    with torch.no_grad():
        for data in loader:
            data.to(device)
            loss = model(data)
    model.train()
    return loss.item()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0)
best_loss = float('inf')
for epoch in range(config.epochs):
    train_loss = train(model, train_loader, optimizer, epoch)
    valid_loss = valid(model, val_loader)
    scheduler.step(valid_loss)
    
    if valid_loss < best_loss:
        best_loss = valid_loss
        torch.save(model.state_dict(), "./checkpoints/"+str(config.graph_model)+"_imagept_"+str(config.pretrained_image_model)+".pt")
    
    print('Epoch: {:02d}, Train Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, train_loss, valid_loss))
    wandb.log({
                'val_loss': valid_loss,
                'train_loss': train_loss,
                'learning_rate': get_lr(optimizer)
              }, step=epoch)
