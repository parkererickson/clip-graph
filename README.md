# Contrastive Point Cloud-Image Pretraining
### Parker Erickson
### CSCI 8980

## Running Inference
You can run inference on a single image/point cloud pair using the ```exploreData.py``` script. The script takes three arguments:
* ```--data_sample```: The name of the data you want to load. The data must be in the ```./data/``` directory. Defaults to ```sample```.
* ```--data_index```: The index of the data sample you want to load. Defaults to index 0.
* ```--model_path```: The path to the model you want to load. Does not load a model for inference by default.

An example of running the script is as follows:
```bash
python exploreData.py -ds sample_small -di 3 -mp ./checkpoints/gat_ghd_128_go_128_imagept_True_embdim_256_pool_max.pt
```

**Note:** If the data has never been processed before (and doesn't show up in ```./processed/```), the script will process the data. This can take a while. Additionally, it will save the processed data to ```./processed/```, so you don't have to process it again. Sometimes after saving, the script will error, but the next time it is ran, it will use the processed data and work just fine.

## Training the Model