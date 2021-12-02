import dataset
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import CLIPGraphModel as cgm
import torch
from torch import nn
import wandb
import argparse

def eval(logits, thresholds=[0.5, 0.75, 0.9, 0.95, 0.99]):
    with torch.no_grad():
        metrics = ["tp", "tn", "fp", "fn", "acc", "f1"]
        results= {x:{k:None for k in metrics} for x in thresholds}
        for threshold in thresholds:
            threshLogits = (logits > threshold)
            tp = torch.trace(threshLogits)
            fn = threshLogits.shape[0] - tp
            fp = torch.triu(threshLogits, diagonal=1).sum()
            denom = torch.triu_indices(threshLogits.shape[0], threshLogits.shape[1]).shape[1]
            results[threshold]["tp"] = (tp/denom)
            results[threshold]["fn"] = (fn/denom)
            results[threshold]["fp"] = (fp/denom)
            results[threshold]["tn"] = 1-results[threshold]["fp"]
            results[threshold]["acc"] = results[threshold]["tn"]+results[threshold]["tp"]
            results[threshold]["f1"] = tp/(tp+0.5*(fp+fn))
        return results

def train(model, loader, optimizer, device):
    res = {}
    for data in loader:
        data.to(device)
        optimizer.zero_grad()
        loss, out = model(data)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            res = eval(out)
    return loss.item(), res

def valid(model, loader, device):
    model.eval()
    with torch.no_grad():
        for data in loader:
            data.to(device)
            loss, out = model(data)
        res = eval(out) 
    model.train()
    return loss.item(), res

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main(args):
    wandb.init(project="CLIP-Graph-Model-Final", entity="parkererickson")

    config = wandb.config
    config.epochs = args.num_epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.image_model = args.image_model
    config.graph_model = args.graph_model
    config.embedding_dim = args.embedding_dim
    config.graph_pool = args.graph_pool
    config.graph_hidden_dim = args.graph_hidden_dim
    config.graph_out_dim = args.graph_out_dim  
    config.linear_proj_dropout = args.linear_proj_dropout
    config.pretrained_image_model = args.pretrained_image_model
    config.lr_patience = 5        # CONSTANT
    config.T_0 = 10               # CONSTANT
    config.data_sample = "2014-07-14-14-49-50"
    #config.data_sample = "sample"

    ds = dataset.GraphCLIP(lidar_timestamp_path='./data/'+config.data_sample+'/lms_front.timestamps', 
                        image_timestamp_path='./data/'+config.data_sample+'/stereo.timestamps',
                        lidar_path='./data/'+config.data_sample+'/lms_front/',
                        image_path='./data/'+config.data_sample+'/stereo/centre/',
                        ins_path='./data/'+config.data_sample+'/gps/ins.csv',
                        data_name=config.data_sample,
                        use_cache=True)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    wandb.watch(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0)
    best_loss = float('inf')
    for epoch in range(config.epochs):
        train_loss, train_metrics = train(model, train_loader, optimizer, device)
        valid_loss, valid_metrics = valid(model, val_loader, device)
        scheduler.step(valid_loss)
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 
                    "./checkpoints/"+str(config.graph_model)+"_ghd_"+str(config.graph_hidden_dim)+"_go_"+str(config.graph_out_dim)+"_imagept_"+str(config.pretrained_image_model)+"_embdim_"+str(config.embedding_dim)+".pt")
        metrics = ["tp", "tn", "fp", "fn", "acc", "f1"]
        thresholds=[0.5, 0.75, 0.9, 0.95, 0.99]
        log = {}
        for metric in metrics:
            for threshold in thresholds:
                log["train_"+metric+"_"+str(threshold)] = train_metrics[threshold][metric]
                log["valid_"+metric+"_"+str(threshold)] = valid_metrics[threshold][metric]
        log["val_loss"] = valid_loss
        log["train_loss"] = train_loss
        log["learning_rate"] = get_lr(optimizer)
        print('Epoch: {:02d}, Train Loss: {:.4f}, Validation Loss: {:.4f}'.format(epoch, train_loss, valid_loss))
        wandb.log(log, step=epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for Contrastive Point Cloud-Image Pretraining")
    parser.add_argument("--num_epochs", "-ne", type=int, default=400, help="Number of Training Epochs")
    parser.add_argument("--batch_size", "-bs", type=int, default=48, help="Batch Size")
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-4, help="What is your desired learning rate")
    parser.add_argument("--image_model", "-im", type=str, default="resnet", help="Type of image model. Currently only resnet is available")
    parser.add_argument("--graph_model", "-gm", type=str, default="pn", help="Type of point cloud model. Default is PointNet, designated 'pn'. Other options are gcn and gat")
    parser.add_argument("--embedding_dim", "-ed", type=int, default=256, help="Joint embedding size")
    parser.add_argument("--graph_pool", "-gp", type=str, default="mean", help="Pooling method of point cloud models. Either 'mean' or 'max'")
    parser.add_argument("--graph_hidden_dim", "-ghd", type=int, default=128, help="Hidden dimension of point cloud models.")
    parser.add_argument("--graph_out_dim", "-gd", type=int, default=128, help="Output dimension of point cloud model")
    parser.add_argument("--linear_proj_dropout", "-lpd", type=float, default=0.1, help="Dropout value of linear projection layer")
    parser.add_argument("--pretrained_image_model", "-pim", type=bool, default=True, help="Use pretrained image model")
    args = parser.parse_args()
    main(args)
