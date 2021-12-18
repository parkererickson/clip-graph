import dataset
import numpy as np
import argparse
import matplotlib.pyplot as plt
import open3d
from robotcar_dataset_sdk.python.transform import build_se3_transform

def infer(model_path, data_point):
    import torch
    import CLIPGraphModel as cgm
    from torch_geometric.loader import DataLoader
    modelConfig = model_path.split('/')[-1].split('.')[0]
    model = cgm.CLIPGraphModel(image_model="resnet",
                            graph_model=modelConfig.split('_')[0],
                            embedding_dim=int(modelConfig.split('_')[-3]),
                            graph_pool=str(modelConfig.split('_')[-1]),
                            graph_hidden_dim=int(modelConfig.split('_')[2]),
                            graph_out_dim=int(modelConfig.split('_')[4]),
                            linear_proj_dropout=0.1,
                            pretrained_image_model=bool(modelConfig.split('_')[-3]))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    similarities = []
    loader = DataLoader([data_point], batch_size=1, shuffle=False)
    for batch in loader:
        loss, out = model(batch)
    similarities += [np.diagonal(out.detach().numpy())]
    return out.item()

def main(args):
    #data_sample = "2014-05-14-13-59-05"
    data_sample = args.data_sample

    ds = dataset.GraphCLIP(lidar_timestamp_path='./data/'+data_sample+'/lms_front.timestamps', 
                            image_timestamp_path='./data/'+data_sample+'/stereo.timestamps',
                            lidar_path='./data/'+data_sample+'/lms_front/',
                            image_path='./data/'+data_sample+'/stereo/centre/',
                            ins_path='./data/'+data_sample+'/gps/ins.csv',
                            data_name=data_sample,
                            use_cache=True)


    data_index = args.data_index

    print("Number of Image/Point Cloud Pairs:", len(ds))
    if args.model_path is not None:
        sim = infer(args.model_path, ds[data_index])
        print("Similarity Between Embeddings: %.4f" % sim)
    img = np.rot90(ds[data_index]['image'].T, k=-1) # Reshape back to (H, W, C)

    plt.imshow(img)
    plt.show(block=False)

    pointcloud = ds[data_index]["x"].numpy()

    vis = open3d.visualization.Visualizer()
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pointcloud)
    pcd.transform(build_se3_transform([0, 0, 0, np.pi, 0, -np.pi / 2]))
    vis.create_window("CSCI 8980 Point Cloud Visualization")
    view_control = vis.get_view_control()
    params = view_control.convert_to_pinhole_camera_parameters()
    params.extrinsic = build_se3_transform([0, 3, 10, 0, -np.pi * 0.42, -np.pi / 2])
    view_control.convert_from_pinhole_camera_parameters(params)
    vis.add_geometry(pcd)
    vis.run()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--data_sample", "-ds", type=str, default="sample")
    args.add_argument("--data_index", "-di", type=int, default=0)
    args.add_argument("--model_path", "-mp", type=str, default=None)
    args = args.parse_args()
    main(args)