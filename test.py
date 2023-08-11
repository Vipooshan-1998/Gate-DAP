import os

from config.config import args
from net import GateDAP
import torch
from torch.utils.data import DataLoader
from data_load import LoadData
import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from tqdm import tqdm
import torch.backends.cudnn as cudnn
#SF0.mat为保存的自定义文件名，label_test、predlabel为需要保存的数据

save_path_prediction = r'E:\DADAV2\result\test_5555555'
# save_path_gt = r'E:\test_1\gt_maps'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_dataset = LoadData(model='test')
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0,
                        pin_memory=True)
Model = GateDAP().to(device)
statedict = torch.load('ckpts/model_best_attention.tar')
weight = statedict['state_dict']
# print(weight)

# state_dict = Model.state_dict()
# state_dict = {k: v for k, v in weight.items() if k in state_dict and state_dict[k].size() == v.size()}
# Model.load_state_dict(state_dict, strict=False)
Model.load_state_dict(weight)

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
# Model.load_state_dict(state_dict['state_dict'])
for i, (input, target, path) in enumerate(tqdm(val_loader)):
    with torch.no_grad():
        input = input.to(device)
        # compute output
        output = Model(input)
        for j in range(target.shape[0]):
            aaa = output[j][0].cpu().detach().numpy()
            bbb = target[j][0].cpu().detach().numpy()

            print(path[j])
            mat_path = os.path.join(save_path_prediction, path[j][-17:-14])
            # gt_path = os.path.join(save_path_gt, path[j][-17:-14])
            if not os.path.exists(mat_path):
                os.makedirs(mat_path)
            # if not os.path.exists(gt_path):
            #     os.makedirs(gt_path)
            # number_mat = numbers(path[0][-8:-4]) + 4

            scipy.io.savemat(mat_path + '/' + path[j][-8:-4] + '.mat', mdict={'prediction': aaa})
            # cv2.imwrite(gt_path + '/' + path[j][-8:-4] + '.png', bbb*255)

