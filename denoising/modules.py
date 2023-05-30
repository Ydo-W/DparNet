import torch
import torch.nn as nn
import torch.nn.functional as F


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)


def conv1x1_3d(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=True)


def conv3x3_3d(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True)


def conv5x5_3d(in_channels, out_channels, stride=1):
    return nn.Conv3d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2, bias=True)


class ResBlock(nn.Module):
    def __init__(self, chans):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(chans, chans)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3(chans, chans)
        self.relu2 = nn.ReLU()
        self.conv3 = conv3x3(chans, chans)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = out + x
        return out


class ResBlock3D(nn.Module):
    def __init__(self, chans):
        super(ResBlock3D, self).__init__()
        self.conv1 = conv3x3_3d(chans, chans)
        self.relu1 = nn.ReLU()
        self.conv2 = conv3x3_3d(chans, chans)
        self.relu2 = nn.ReLU()
        self.conv3 = conv3x3_3d(chans, chans)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = out + x
        return out


class CNN3D(nn.Module):
    def __init__(self, para, n_blocks, in_channels=1, out_channels=80, stride=1):
        super(CNN3D, self).__init__()
        self.n_feats = para.n_feats
        self.CNN_B0 = conv5x5_3d(in_channels, self.n_feats, stride=1)
        self.CNN_B1 = conv5x5_3d(self.n_feats, 2 * self.n_feats, stride=stride)
        self.CNN_B3 = conv5x5_3d(2 * self.n_feats, out_channels, stride=stride)
        self.CNN_B2 = nn.ModuleList()
        self.CNN_B4 = nn.ModuleList()
        for i in range(n_blocks):
            self.CNN_B2.append(ResBlock3D(2 * self.n_feats))
            self.CNN_B2.append(nn.BatchNorm3d(2 * self.n_feats))
            self.CNN_B4.append(ResBlock3D(out_channels))
            self.CNN_B4.append(nn.BatchNorm3d(out_channels))

    def forward(self, x):
        out = self.CNN_B0(x)
        out = self.CNN_B1(out)
        for layer in self.CNN_B2:
            out = layer(out)
        out = self.CNN_B3(out)
        for layer in self.CNN_B4:
            out = layer(out)
        return out


# Reconstructor for clean sequences
class Reconstructor(nn.Module):
    def __init__(self, para):
        super(Reconstructor, self).__init__()
        self.num_ff = para.neighboring_frames
        self.num_fb = para.neighboring_frames
        self.related_f = 5
        self.n_feats = para.n_feats
        self.model = nn.Sequential(
            nn.ConvTranspose2d((5*self.n_feats)*self.related_f, 4*self.n_feats, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            ResBlock(4 * self.n_feats),
            ResBlock(4 * self.n_feats),
            nn.ConvTranspose2d(4*self.n_feats, 2*self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResBlock(2*self.n_feats),
            ResBlock(2*self.n_feats),
            conv5x5(2*self.n_feats, 3, stride=1)
        )

    def forward(self, x):
        return self.model(x)


class para_Reconstructor(nn.Module):
    def __init__(self, para, in_channels, out_channels):
        super(para_Reconstructor, self).__init__()
        self.n_feats = para.n_features
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 2*self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResBlock(2*self.n_feats),
            ResBlock(2*self.n_feats),
            nn.ConvTranspose2d(2*self.n_feats, self.n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
            ResBlock(self.n_feats),
            ResBlock(self.n_feats),
            conv3x3(self.n_feats, out_channels)
        )

    def forward(self, x):
        return self.model(x)


class dense_layer(nn.Module):
    def __init__(self, in_channels, growthRate):
        super(dense_layer, self).__init__()
        self.conv = conv3x3(in_channels, growthRate)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)  # C = in_channels + growthRate
        return out


class RDB(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer):
        super(RDB, self).__init__()
        in_channels_ = in_channels
        modules = []
        for i in range(num_layer):
            modules.append(dense_layer(in_channels_, growthRate))  # 16->32->48->64
            in_channels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv1x1 = conv1x1(in_channels_, in_channels)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv1x1(out)
        out += x
        return out


class RDNet(nn.Module):
    def __init__(self, in_channels, growthRate, num_layer, num_blocks):
        super(RDNet, self).__init__()
        self.num_blocks = num_blocks
        self.RDBs = nn.ModuleList()
        for i in range(num_blocks):
            self.RDBs.append(RDB(in_channels, growthRate, num_layer))
        self.conv1x1 = conv1x1(num_blocks * in_channels, in_channels)
        self.conv3x3 = conv3x3(in_channels, in_channels)

    def forward(self, x):
        out = []
        h = x
        for i in range(self.num_blocks):  # i = 0, 1, 2
            h = self.RDBs[i](h)
            out.append(h)
        out = torch.cat(out, dim=1)  # (bs, 240, 64, 64)
        out = self.conv1x1(out)
        out = self.conv3x3(out)
        return out


class RNNcell(nn.Module):
    def __init__(self, n_blocks=9, n_feats=16):
        super(RNNcell, self).__init__()
        self.n_feats = n_feats
        self.n_blocks = n_blocks
        self.F_B0 = conv5x5(3, self.n_feats, stride=1)
        self.F_B1 = nn.Sequential(
            RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3),
            conv5x5(self.n_feats, 2 * self.n_feats, stride=2)
        )
        self.F_B2 = nn.Sequential(
            RDB(in_channels=2*self.n_feats, growthRate=2*self.n_feats, num_layer=3),
            conv5x5(2 * self.n_feats, 4 * self.n_feats, stride=2)
        )
        self.F_R = RDNet(in_channels=(1 + 4) * self.n_feats, growthRate=2 * self.n_feats, num_layer=3,
                         num_blocks=self.n_blocks)  # in: 80
        self.F_h = nn.Sequential(
            conv3x3((1 + 4) * self.n_feats, self.n_feats),
            RDB(in_channels=self.n_feats, growthRate=self.n_feats, num_layer=3),
            conv3x3(self.n_feats, self.n_feats)
        )

    def forward(self, x, s_last):
        out = self.F_B0(x)
        out = self.F_B1(out)
        out = self.F_B2(out)
        out = torch.cat([out, s_last], dim=1)
        out = self.F_R(out)
        s = self.F_h(out)
        return out, s


# BRNN网络
class BRNN(nn.Module):
    def __init__(self, ff=2, fb=2, n_blocks=9, n_feats=16):
        super(BRNN, self).__init__()
        self.n_feats = n_feats
        self.num_ff = ff
        self.num_fb = fb
        self.ds_ratio = 4
        self.device = torch.device('cuda')
        self.cell = RNNcell(n_blocks)
        self.cell1 = RNNcell(n_blocks)
        self.y_Merge = conv1x1(10 * self.n_feats, 5 * self.n_feats)

    def forward(self, x):
        outputs, hs0, hs1, hs = [], [], [], []
        batch_size, frames, channels, height, width = x.shape
        s_height = int(height / self.ds_ratio)
        s_width = int(width / self.ds_ratio)
        s = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)
        for i in range(frames):
            h, s = self.cell(x[:, i, :, :, :], s)
            hs0.append(h)
        s = torch.zeros(batch_size, self.n_feats, s_height, s_width).to(self.device)
        for i in range(frames - 1, -1, -1):
            h, s = self.cell1(x[:, i, :, :, :], s)
            hs1.append(h)
        for i in range(frames):
            s = torch.cat([hs1[frames - 1 - i], hs0[i]], dim=1)
            s = self.y_Merge(s)
            hs.append(s)
        return hs



