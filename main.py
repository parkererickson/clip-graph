import dataset
from torch_geometric.loader import DataLoader
import CLIPGraphModel as cgm
import torch
import wandb

wandb.init(project="CLIP-Graph-Model", entity="parkererickson")

config = wandb.config
config.epochs = 100
config.batch_size = 16
config.learning_rate = 0.01

ds = dataset.GraphCLIP(lidar_timestamp_path='./data/sample/lms_front.timestamps', 
                       image_timestamp_path='./data/sample/stereo.timestamps',
                       lidar_path='./data/sample/lms_front/',
                       image_path='./data/sample/stereo/centre/',
                       ins_path='./data/sample/gps/ins.csv',
                       use_cache=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train, val = torch.utils.data.random_split(ds, [int(0.8 * len(ds)), int(0.2 * len(ds))])
train_loader = DataLoader(train, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val, batch_size=config.batch_size, shuffle=True)

model = cgm.CLIPGraphModel(image_model="vit")
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

optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
for epoch in range(config.epochs):
    train(model, train_loader, optimizer, epoch)
    if epoch % 10 == 0:
        valid(model, val_loader, epoch)



