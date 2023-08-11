import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import json
from torch.utils.data import DataLoader
from torchvision import transforms as T


# from PIL import Image

class LoadData(Dataset):
    def __init__(self, model='train', image_size=224):
        self.root = "datasets\\"
        self.size = (image_size, image_size)
        self.transforms = T.Compose([T.ToTensor()])
        # if model == 'training':
        self.data = [json.loads(line) for line in open(self.root + model + '.json')]
        self.maps = [json.loads(line) for line in open(self.root + model + '_maps.json')]

    def __getitem__(self, index):
        batch_t = self.data[index]
        image = []
        seg = []
        flow = []
        area = []
        for rgb_path in batch_t[:4]:
            Img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            Img = (cv2.resize(Img, self.size)/255).astype('float32')
            image.append(self.transforms(Img))
        for seg_path in batch_t[4:8]:
            Seg = cv2.imread(seg_path, cv2.IMREAD_COLOR)
            Seg = (cv2.resize(Seg, self.size)/255).astype('float32')
            seg.append(self.transforms(Seg))
        for flow_path in batch_t[8:12]:
            Flow = cv2.imread(flow_path, cv2.IMREAD_COLOR)
            Flow = (cv2.resize(Flow, self.size)/255).astype('float32')
            flow.append(self.transforms(Flow))
        for area_path in batch_t[12:16]:
            Area = cv2.imread(area_path, cv2.IMREAD_COLOR)
            Area = (cv2.resize(Area, self.size)/255).astype('float32')
            area.append(self.transforms(Area))
        image_tensor = torch.stack(image, 0)
        seg_tensor = torch.stack(seg, 0)
        flow_tensor = torch.stack(flow, 0)
        area_tensor = torch.stack(area, 0)
        # print(image_tensor.shape)
        # print(seg_tensor.shape)
        # print(flow_tensor.shape)
        # print(area_tensor.shape)

        map_path = self.maps[index][0]
        Map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        Map = Map[96:-96,:]
        Map = (cv2.resize(Map, self.size)/255).astype('float32')
        map_tensor = self.transforms(Map)

        assert isinstance(map_tensor, object)

        data_tensor = torch.stack((image_tensor, seg_tensor, flow_tensor, area_tensor), 0)

        return data_tensor, map_tensor, map_path

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    root = "C:\\Users\\admin\\Desktop\\GateDAP\\json\\"
    root_data = 'D:/DADA_dataset/'
    train_data = [json.loads(line) for line in open(root + 'train.json')]

    map_data = [json.loads(line) for line in open(root + 'train_maps.json')]

    train_loader = DataLoader(
        LoadData(model='train'),
        batch_size=8, shuffle=True,
        num_workers=0,
        pin_memory=True)

    valid_loader = DataLoader(
        LoadData(model='val'),
        batch_size=1, shuffle=False,
        num_workers=0,
        pin_memory=True)

    for epoch in range(4):
        print(train_loader.__len__())
        for i, (x1, x2) in enumerate(train_loader):
            x1 = x1.permute(1, 0, 2, 3, 4, 5)
            print(x1.shape)
            print(x2.shape)
            exit(0)
