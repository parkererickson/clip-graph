import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from robotcar_dataset_sdk.python import build_pointcloud as bpc
import numpy as np
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import os
from torch_cluster import knn_graph

class GraphCLIP(Dataset):
    def __init__(self,
                 processed_dir='./processed/',
                 lidar_timestamp_path='./data/sample/lms_front.timestamps', 
                 image_timestamp_path='./data/sample/stereo.timestamps',
                 lidar_path = './data/sample/lms_front',
                 image_path = './data/sample/stereo',
                 ins_path='./data/sample/gps/ins.csv',
                 extrinsics_path='./robotcar_dataset_sdk/extrinsics',
                 data_name="sample",
                 use_cache=True):
        self.lidar_path = lidar_path
        self.lidar_timestamp_path = lidar_timestamp_path
        self.image_timestamp_path = image_timestamp_path
        self.image_path = image_path
        self.ins_path = ins_path
        self.extrinsics_path = extrinsics_path
        self.data_name = data_name
        if os.path.isdir(processed_dir):
            if os.path.isfile(processed_dir+data_name+'_data.pt') and use_cache:
                self.data = torch.load(processed_dir+data_name+'_data.pt')
            else:
                self.data = self.process()
                torch.save(self.data, processed_dir+data_name+'_data.pt')
        else:
            os.mkdir(processed_dir)
            self.data = self.process()
            torch.save(self.data, processed_dir+data_name+'_data.pt')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @property
    def processed_file_names(self):
        return [self.data_name+'_data.pt']

    def _getTimestamps(self, timestep_path):
        times = []
        timestamps_file = open(timestep_path)
        for line in timestamps_file.readlines():
            times.append(int(line.split()[0]))
        return times

    def _build_pc(self, pc_filepath, ins_filepath, extrinsics_path, start_time, end_time):
        pc = bpc.build_pointcloud(pc_filepath, ins_filepath, extrinsics_path, start_time, end_time, origin_time=start_time)[0][:3].T
        return pc

    def process(self):
        data_list = []
        # Read data into huge `Data` list.
        imageTimes = self._getTimestamps(self.image_timestamp_path)
        lidarTimes = self._getTimestamps(self.lidar_timestamp_path)

        imagePCTimes = {}
        for i in range(0, len(imageTimes)):
            if i > 0:
                imagePCTimes[imageTimes[i]] = [x for x in lidarTimes if x <= imageTimes[i] and x > imageTimes[i-1]]
            else:
                imagePCTimes[imageTimes[i]] = [x for x in lidarTimes if x <= imageTimes[i]]

        for item in tqdm(imagePCTimes.keys()):
            if imagePCTimes[item] != []:
                pc = np.asarray(self._build_pc(self.lidar_path, 
                                               self.ins_path, 
                                               self.extrinsics_path, 
                                               imagePCTimes[item][0], 
                                               imagePCTimes[item][-1]))

                
                X = torch.from_numpy(pc).double()
                edge_index = knn_graph(X, k=6)
                im = Image.open(self.image_path+str(item)+'.png')
                transform = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor()])
                imageData = transform(im).expand(3, -1, -1)
                data = Data(x=X, edge_index=edge_index, image=imageData)
                data_list.append(data)

        return data_list
