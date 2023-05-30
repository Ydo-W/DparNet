import utils
import torch
import modules
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F


class DeepModel(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.frames = para.frame_length
        self.neighbors = para.neighboring_frames
        self.extractor = modules.BRNN()
        self.reconstructor = modules.Reconstructor(para)

    def forward(self, input_data):
        feature_list = self.extractor(input_data)
        outputs = []
        for i in range(self.neighbors, self.frames - self.neighbors):
            out = torch.cat(feature_list[i-self.neighbors: i+self.neighbors+1], dim=1)
            out = self.reconstructor(out)
            outputs.append(out)
        return outputs


class WideModel(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.frames = para.frame_length
        self.neighbors = para.neighboring_frames
        self.block = nn.Sequential(
            modules.conv1x1(2, para.n_feats),
            modules.conv5x5(para.n_feats, 2 * para.n_feats, stride=2),
            modules.RDB(2 * para.n_feats, para.n_feats, 3),
            nn.ConvTranspose2d(2 * para.n_feats, para.n_feats, \
                               kernel_size=3, stride=2, padding=1, output_padding=1),
            modules.conv5x5(para.n_feats, 1, stride=1)
        )

    def forward(self, input_data, deg_map):
        outputs = []
        for i in range(self.neighbors, self.frames - self.neighbors):
            out = torch.cat([input_data[:, i], deg_map], dim=1)
            out = self.block(out)
            outputs.append(out)
        return outputs


#  Degradation parameter assisted restoration network
class DparNet(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.frames = para.frame_length
        self.neighbors = para.neighboring_frames
        self.deep_model = DeepModel(para)
        self.wide_model = WideModel(para)
        self.merge = modules.conv1x1(2, 1)

    def forward(self, input_data, deg_map):
        y1_list = self.deep_model(input_data)
        y2_list = self.wide_model(input_data, deg_map)
        outputs = []
        for i in range(len(y1_list)):
            out1, out2 = y1_list[i], y2_list[i]
            out = self.merge(torch.cat([out1, out2], dim=1))
            outputs.append(out.unsqueeze(dim=1))
        return torch.cat(outputs, dim=1)


# Parameter prediction module
class PPN(nn.Module):
    def __init__(self, para):
        super().__init__()
        self.para = para
        self.pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.CNN1 = modules.CNN3D(para, 3, 3, 1 * para.n_feats, stride=(1, 2, 2))
        self.CNN2 = nn.Sequential(
            modules.conv1x1(para.frame_length * para.n_feats, para.n_feats),
            nn.ConvTranspose2d(para.n_feats, para.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(para.n_feats, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, input_data):
        B, F, C, H, W = input_data.shape
        x0 = input_data.permute(0, 2, 1, 3, 4)
        x1 = self.pool(x0).repeat(1, 1, F, 1, 1)
        x = torch.cat([x0, x0 - x1, x1 - x0], dim=1)
        f = self.CNN1(x)
        f = rearrange(f, 'b c f h w -> b (c f) h w')
        y = self.CNN2(f)
        return y
