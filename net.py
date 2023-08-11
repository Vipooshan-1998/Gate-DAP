from models_vit import vit_base_patch16
from einops.layers.torch import Rearrange
import torch.nn as nn
import torch
from ConvGRU import ConvGRU

import torch.nn.functional as F
import matplotlib.pyplot as plt

class InfoGate(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(InfoGate, self).__init__()
        self.a =input_dim
        self.conv1d = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=int(input_dim/2), kernel_size=(3, 3), padding=(1, 1)),
                                    nn.ReLU(),
                                    nn.Conv1d(in_channels=int(input_dim/2), out_channels=output_dim, kernel_size=(3, 3), padding=(1, 1)),
                                    nn.ReLU())
    def forward(self, input):
        x = torch.cat((input[0], input[1], input[2], input[3]), 1)
        x = self.conv1d(x)

        h = x.shape[2]
        w = x.shape[3]
        gate = Rearrange('b c h w -> b (h w) c', h=h, w=w)(x)
        gate = F.gumbel_softmax(gate, tau=0.3)
        gate = Rearrange('b (h w) c -> b c h w', h=h, w=w)(gate)

        out0 = input[0] * gate[:, 0, :, :].view(-1, 1, h, w)
        out1 = input[1] * gate[:, 1, :, :].view(-1, 1, h, w)
        out2 = input[2] * gate[:, 2, :, :].view(-1, 1, h, w)
        out3 = input[3] * gate[:, 3, :, :].view(-1, 1, h, w)

        return out0, out1, out2, out3

class LongshortGate(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LongshortGate, self).__init__()
        self.a =input_dim
        self.conv1d = nn.Sequential(nn.Conv1d(in_channels=input_dim, out_channels=int(input_dim/2), kernel_size=(3, 3), padding=(1, 1)),
                                        nn.ReLU(),
                                        nn.Conv1d(in_channels=int(input_dim/2), out_channels=output_dim, kernel_size=(3, 3), padding=(1, 1)),
                                        nn.ReLU())
    def forward(self, input):
        T, B, C, H, W = input.shape
        vector = torch.zeros((input.shape)).to(input.device)
        y = input.view(B, -1, H, W)
        x = self.conv1d(y)
        gate = Rearrange('b c h w -> b (h w) c', h=H, w=W)(x)
        gate = F.gumbel_softmax(gate, tau=0.3)
        gate = Rearrange('b (h w) c -> b c h w', h=H, w=W)(gate)
        for i in range(T):
            vector[i] = input[i] + input[i] * gate[:, i, :, :].view(-1, 1, H, W)
        return vector
# （2）空间注意力机制
class spatial_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(spatial_attention, self).__init__()
        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)
        # 空间权重归一化
        x = self.sigmoid(x)
        # plt.imshow(x[0,0,:,:].cpu().numpy())
        # plt.show()
        # 输入特征图和空间权重相乘
        outputs = inputs * x

        return outputs


class Decoder2D(nn.Module):
    def __init__(self, in_channels=1024, out_channels=1,
                 features=[512, 256, 128, 64]):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.decoder_1 = nn.Sequential(
            nn.Conv2d(in_channels, features[0], 3, padding=1),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_2 = nn.Sequential(
            nn.Conv2d(features[0], features[1], 3, padding=1),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_3 = nn.Sequential(
            nn.Conv2d(features[1], features[2], 3, padding=1),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.decoder_4 = nn.Sequential(
            nn.Conv2d(features[2], features[3], 3, padding=1),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        )
        self.final_out = nn.Conv2d(features[-1], out_channels, 3, padding=1)

    def forward(self, x):
        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)
        x = self.decoder_4(x)
        x = self.final_out(x)
        x = self.sigmoid(x)
        return x

class GateDAP(nn.Module):
    def __init__(self, input_t=4, patch_size=16, image_size=224, embed_dim=768):
        super().__init__()
        self.vit = vit_base_patch16()
        self.t = input_t
        self.patch_size = patch_size
        self.image_size = image_size
        self.h = self.image_size // self.patch_size
        self.w = self.image_size // self.patch_size
        self.kernel_size = (3, 3)
        self.padding = self.kernel_size[0] // 2, self.kernel_size[1] // 2
        self.dim = 128
        self.embed_dim = embed_dim

        self.rgb_spatial_gate = nn.ModuleList(spatial_attention() for i in range(self.t))
        self.seg_spatial_gate = nn.ModuleList(spatial_attention() for i in range(self.t))
        self.flow_spatial_gate = nn.ModuleList(spatial_attention() for i in range(self.t))
        self.area_spatial_gate = nn.ModuleList(spatial_attention() for i in range(self.t))

        self.convGRU_rgb = ConvGRU(input_size=(self.h, self.w), input_dim=self.dim,
                                   hidden_dim=self.dim, kernel_size=self.kernel_size,
                                   num_layers=1, dtype='torch.cuda.FloatTensor', batch_first=False,
                                   bias=True, return_all_layers=False, use_trace=False)
        self.convGRU_seg = ConvGRU(input_size=(self.h, self.w), input_dim=self.dim,
                                   hidden_dim=self.dim, kernel_size=self.kernel_size,
                                   num_layers=1, dtype='torch.cuda.FloatTensor', batch_first=False,
                                   bias=True, return_all_layers=False, use_trace=False)
        self.convGRU_flow = ConvGRU(input_size=(self.h, self.w), input_dim=self.dim,
                                    hidden_dim=self.dim, kernel_size=self.kernel_size,
                                    num_layers=1, dtype='torch.cuda.FloatTensor', batch_first=False,
                                    bias=True, return_all_layers=False, use_trace=False)
        self.convGRU_area = ConvGRU(input_size=(self.h, self.w), input_dim=self.dim,
                                    hidden_dim=self.dim, kernel_size=self.kernel_size,
                                    num_layers=1, dtype='torch.cuda.FloatTensor', batch_first=False,
                                    bias=True, return_all_layers=False, use_trace=False)

        self.decord = Decoder2D(in_channels=4 * self.dim)
        self.conv1d_rgb = nn.Conv1d(in_channels=embed_dim, out_channels=self.dim, kernel_size=(3, 3), padding=(1, 1))
        self.conv1d_seg = nn.Conv1d(in_channels=embed_dim, out_channels=self.dim, kernel_size=(3, 3), padding=(1, 1))
        self.conv1d_flow = nn.Conv1d(in_channels=embed_dim, out_channels=self.dim, kernel_size=(3, 3), padding=(1, 1))
        self.conv1d_area = nn.Conv1d(in_channels=embed_dim, out_channels=self.dim, kernel_size=(3, 3), padding=(1, 1))
        self.InfoGate = InfoGate(input_dim=4*self.dim, output_dim=4)
        self.LongshortGate_rgb = LongshortGate(input_dim=input_t*self.dim, output_dim=input_t)
        self.LongshortGate_seg = LongshortGate(input_dim=input_t*self.dim, output_dim=input_t)
        self.LongshortGate_flow = LongshortGate(input_dim=input_t*self.dim, output_dim=input_t)
        self.LongshortGate_area = LongshortGate(input_dim=input_t*self.dim, output_dim=input_t)

    def forward(self, x):  # (B,M,T,C,H,W)
        x = x.permute(1, 0, 2, 3, 4, 5)  # (M.B,T,C,H,W)
        rgb_tensor, seg_tensor, flow_tensor, area_tensor = [], [], [], []
        rgb = x[0].permute(1, 0, 2, 3, 4)  # (B,T,C,H,W) -> (T,B,C,H,W)
        seg = x[1].permute(1, 0, 2, 3, 4)
        flow = x[2].permute(1, 0, 2, 3, 4)
        area = x[3].permute(1, 0, 2, 3, 4)
        for i in range(self.t):
            rgb1 = self.vit(rgb[i])  # (B,c,h,w)
            seg1 = self.vit(seg[i])
            flow1 = self.vit(flow[i])
            area1 = self.vit(area[i])

            rgb1 = self.conv1d_rgb(rgb1)
            seg1 = self.conv1d_seg(seg1)
            flow1 = self.conv1d_flow(flow1)
            area1 = self.conv1d_area(area1)

            rgb_gate = self.rgb_spatial_gate[i](rgb1)
            seg_gate = self.seg_spatial_gate[i](seg1)
            flow_gate = self.flow_spatial_gate[i](flow1)
            area_gate = self.area_spatial_gate[i](area1)

            rgb_tensor.append(rgb1)
            seg_tensor.append(seg1)
            flow_tensor.append(flow1)
            area_tensor.append(area1)

        rgb_encord = torch.stack(rgb_tensor, 0)  # (T,B,c,H,W)
        seg_encord = torch.stack(seg_tensor, 0)
        flow_encord = torch.stack(flow_tensor, 0)
        area_encord = torch.stack(area_tensor, 0)

        rgb_encord = self.LongshortGate_rgb(rgb_encord)
        seg_encord = self.LongshortGate_seg(seg_encord)
        flow_encord = self.LongshortGate_flow(flow_encord)
        area_encord = self.LongshortGate_area(area_encord)

        # print(rgb_encord.shape)
        rgb_predict = self.convGRU_rgb(rgb_encord)  # (B,T,c,H,W) -> (B,c,H,W)
        seg_predict = self.convGRU_seg(seg_encord)
        flow_predict = self.convGRU_flow(flow_encord)
        area_predict = self.convGRU_area(area_encord)
        # print(rgb_predict.shape)
        rgb_gate, seg_gate, flow_gate, area_gate = self.InfoGate([rgb_predict[:, -1, :],
                                                                  seg_predict[:, -1, :],
                                                                  flow_predict[:, -1, :],
                                                                  area_predict[:, -1, :]])
        # print(rgb_gate.shape)
        embeding = torch.cat((rgb_gate, seg_gate, flow_gate, area_gate), 1)  # (B,4*c,H,W)
        out = self.decord(embeding)
        return out

