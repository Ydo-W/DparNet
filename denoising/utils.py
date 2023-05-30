import os
import torch
import random
from tqdm import tqdm
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torchvision.models.vgg import vgg19


# Getting files from a path
def get_files(path):
    files = os.listdir(path)
    files.sort(key=lambda x: int(x[:-4]))
    return files


# Getting a shuffled list
def shuffle_list(num):
    lis = list(range(num))
    random.shuffle(lis)
    return lis


# Normalization of the imaging data
def prepare(centralized, normalized, x):
    rgb = 255.0
    if centralized:
        x = x - rgb / 2
    if normalized:
        x = x / rgb
    return x


def prepare_reverse(centralized, normalized, x):
    rgb = 255.0
    if normalized:
        x = x * rgb
    if centralized:
        x = x + rgb / 2
    return x


def train(device, train_loader, DparNet, PNN, opt, criterion, perception, metrics, epoch, batch_size):
    DparNet.train()
    PSNR_meter = AverageMeter()
    pbar = tqdm(total=len(train_loader) * batch_size, ncols=100)
    for input_data, gt_data, para_data in train_loader:
        # ---------------- data pre -----------------------
        input_data, gt_data, para_data = input_data.to(device), gt_data.to(device), para_data.to(device)
        # ---------------- TIM training -------------------
        DparNet.zero_grad()
        para_pred = PNN(input_data)
        restoration_data = DparNet(input_data, para_pred)
        TIM_loss = 1.0 * criterion(restoration_data, gt_data) + 0.05 * perception(restoration_data, gt_data)
        PSNR = metrics(restoration_data.detach(), gt_data)
        PSNR_meter.update(PSNR.detach().item(), input_data.shape[0])
        TIM_loss.backward()
        opt.step()
        pbar.update(batch_size)
    pbar.close()
    # ---------------- info printing ------------------
    print('Epoch [{:03d}]'.format(epoch), end='--')
    print('Train_PSNR: {:.5f}'.format(PSNR_meter.avg), end='--')
    return PSNR_meter.avg


# Valid
def valid(device, valid_loader, IpairNet, PNN, metrics):
    IpairNet.eval()
    PSNR_meter = AverageMeter()
    with torch.no_grad():
        for input_data, gt_data, para_data in valid_loader:
            # ---------------- data pre -----------------------
            input_data, gt_data, para_data = input_data.to(device), gt_data.to(device), para_data.to(device)
            para_pred = PNN(input_data)
            restoration_data = IpairNet(input_data, para_pred)
            # restoration_data = IpairNet(input_data)
            PSNR = metrics(restoration_data.detach(), gt_data)
            PSNR_meter.update(PSNR.detach().item(), input_data.shape[0])
    # ---------------- info printing ------------------
    print('Valid_PSNR: {:.5f}'.format(PSNR_meter.avg), end='--')
    return PSNR_meter.avg


# Computing some values
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Computing the PSNR of clean sequence reconstruction
class PSNR(_Loss):
    def __init__(self):
        super(PSNR, self).__init__()
        self.rgb = 255

    def _quantize(self, x):
        return x.clamp(0, self.rgb).round()

    def forward(self, x, y):
        x, y = x*255.0, y*255.0
        diff = self._quantize(x) - y
        if x.dim() == 3:
            n = 1
        elif x.dim() == 4:
            n = x.size(0)
        elif x.dim() == 5:
            n = x.size(0) * x.size(1)
        mse = diff.div(self.rgb).pow(2).view(n, -1).mean(dim=-1) + 0.000001
        psnr = -10 * mse.log10()

        return psnr.mean()


# Computing the perceptual loss
class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()

        vgg = vgg19(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:35]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        if len(x.shape) == 5:
            b, n, c, h, w = x.shape
            x = x.reshape(b * n, c, h, w)
            y = y.reshape(b * n, c, h, w)
            if c == 1:
                x = x.repeat(1, 3, 1, 1)
                y = y.repeat(1, 3, 1, 1)
        perception_loss = self.l1_loss(self.loss_network(x), self.loss_network(y))
        return perception_loss

