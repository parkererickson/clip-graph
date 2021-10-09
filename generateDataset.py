from robotcar_dataset_sdk.python import build_pointcloud as bpc
import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors

def getTimestamps(timestep_path):
    times = []
    timestamps_file = open(timestep_path)
    for line in timestamps_file.readlines():
        times.append(int(line.split()[0]))
    return times


def build_pc(pc_filepath, ins_filepath, extrinsics_path, start_time, end_time):
    pc = bpc.build_pointcloud(pc_filepath, ins_filepath, extrinsics_path, start_time, end_time, origin_time=start_time)[0][:3].T
    return pc

imageTimes = getTimestamps("./data/sample/stereo.timestamps")
lmsFrontTimes = getTimestamps("./data/sample/lms_front.timestamps")


imagePCTimes = {}
for i in range(0, len(imageTimes)):
    if i > 0:
        imagePCTimes[imageTimes[i]] = [x for x in lmsFrontTimes if x <= imageTimes[i] and x > imageTimes[i-1]]
    else:
        imagePCTimes[imageTimes[i]] = [x for x in lmsFrontTimes if x <= imageTimes[i]]


pointGraphs = {}

for item in imagePCTimes.keys():
    if imagePCTimes[item] != []:
        print(imagePCTimes[item][0], imagePCTimes[item][-1])
        pc = np.asarray(build_pc("./data/sample/lms_front", "./data/sample/gps/ins.csv", "./robotcar_dataset_sdk/extrinsics", imagePCTimes[item][0], imagePCTimes[item][-1]))
        print(pc.shape)
        nn = NearestNeighbors()
        nn.fit(pc)
        graph = nn.kneighbors_graph(pc, n_neighbors=6, mode="distance")
        print(graph[0])