# Contrastive Point Cloud-Image Pretraining
### Parker Erickson
### CSCI 8980

## Environment Setup
It is recommended to run this code in a conda environment. Once that is setup, you can install the dependencies with the following command:
```
pip install -r requirements.txt
```
from the home directory of this repository. Sometimes the PyTorch Geometric package and its dependencies are not installed correctly, and are highly system-dependent. Google is your friend here.

## Running Inference
You can run inference on a single image/point cloud pair using the ```exploreData.py``` script. The script takes three arguments:
* ```--data_sample```: The name of the data you want to load. The data must be in the ```./data/``` directory. Defaults to ```sample```.
* ```--data_index```: The index of the data sample you want to load. Defaults to index 0.
* ```--model_path```: The path to the model you want to load. Does not load a model for inference by default.

An example of running the script is as follows:
```bash
python exploreData.py -ds sample_small -di 3 -mp ./checkpoints/gat_ghd_128_go_128_imagept_True_embdim_256_pool_max.pt
```

This will load the data sample ```sample_small``` and the model ```gat_ghd_128_go_128_imagept_True_embdim_256_pool_max.pt``` and run inference on the data sample ```sample_small[3]```. The similarity inferred by the model will be printed to the terminal, and the image and point cloud will be displayed. 

**Note:** If the data has never been processed before (and doesn't show up in ```./processed/```), the script will process the data. This can take a while. Additionally, it will save the processed data to ```./processed/```, so you don't have to process it again. Sometimes after saving, the script will error, but the next time it is ran, it will use the processed data and work just fine.

The model names mean:
* First argument: ```pn``` for PointNet, ```gcn``` for Graph Convolutional Neural Network, or ```gat``` for Graph Attention Network.

* ```ghd``` and ```go``` are the hidden and output dimensions of the point cloud model. (The provided ones are all 128.)

* ```imagept``` is a boolean indicating whether or not the model originally used a pretrained ResNet18 image model.

* ```embdim``` is the dimension of the embedding layer. (The provided one is 256.)

* ```pool``` is the pooling method used by the point cloud model. Either ```max``` or ```mean```.


## Training the Model
You can train the model using the ```main.py``` file. The script takes the following arguments:
```bash
usage: main.py [-h] [--num_epochs NUM_EPOCHS] [--batch_size BATCH_SIZE]
               [--learning_rate LEARNING_RATE] [--image_model IMAGE_MODEL]
               [--graph_model GRAPH_MODEL] [--embedding_dim EMBEDDING_DIM]
               [--graph_pool GRAPH_POOL] [--graph_hidden_dim GRAPH_HIDDEN_DIM]
               [--graph_out_dim GRAPH_OUT_DIM]
               [--linear_proj_dropout LINEAR_PROJ_DROPOUT]
               [--pretrained_image_model PRETRAINED_IMAGE_MODEL]

Arguments for Contrastive Point Cloud-Image Pretraining

optional arguments:
  -h, --help            show this help message and exit
  --num_epochs NUM_EPOCHS, -ne NUM_EPOCHS
                        Number of Training Epochs
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        Batch Size
  --learning_rate LEARNING_RATE, -lr LEARNING_RATE
                        What is your desired learning rate
  --image_model IMAGE_MODEL, -im IMAGE_MODEL
                        Type of image model. Currently only resnet is
                        available
  --graph_model GRAPH_MODEL, -gm GRAPH_MODEL
                        Type of point cloud model. Default is PointNet,
                        designated 'pn'. Other options are gcn and gat
  --embedding_dim EMBEDDING_DIM, -ed EMBEDDING_DIM
                        Joint embedding size
  --graph_pool GRAPH_POOL, -gp GRAPH_POOL
                        Pooling method of point cloud models. Either 'mean' or
                        'max'
  --graph_hidden_dim GRAPH_HIDDEN_DIM, -ghd GRAPH_HIDDEN_DIM
                        Hidden dimension of point cloud models.
  --graph_out_dim GRAPH_OUT_DIM, -gd GRAPH_OUT_DIM
                        Output dimension of point cloud model
  --linear_proj_dropout LINEAR_PROJ_DROPOUT, -lpd LINEAR_PROJ_DROPOUT
                        Dropout value of linear projection layer
  --pretrained_image_model PRETRAINED_IMAGE_MODEL, -pim PRETRAINED_IMAGE_MODEL
                        Use pretrained image model
```

You can train the model on a different dataset by changing the config.data_sample variable in the main.py file. The model will automatically take advantage of GPUs if they are available.