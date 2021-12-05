import dataset
import numpy as np

#data_sample = "2014-05-14-13-59-05"
data_sample = "sample_small"

ds = dataset.GraphCLIP(lidar_timestamp_path='./data/'+data_sample+'/lms_front.timestamps', 
                        image_timestamp_path='./data/'+data_sample+'/stereo.timestamps',
                        lidar_path='./data/'+data_sample+'/lms_front/',
                        image_path='./data/'+data_sample+'/stereo/centre/',
                        ins_path='./data/'+data_sample+'/gps/ins.csv',
                        data_name=data_sample,
                        use_cache=True)


dataSample = 30
print("Number of Image/Point Cloud Pairs:", len(ds))
img = np.rot90(ds[dataSample]['image'].T, k=-1) # Reshape back to (H, W, C)


import matplotlib.pyplot as plt
plt.imshow(img)
plt.show(block=False)

import open3d

pointcloud = ds[dataSample]["x"].numpy()
from robotcar_dataset_sdk.python.transform import build_se3_transform


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