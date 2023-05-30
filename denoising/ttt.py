import os
import cv2
import torch
import networks
import numpy as np
from para import Parameter


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    para = Parameter().args
    device = torch.device('cuda')

    # Loading models
    PPN = networks.PPN(para).to(device)
    IpairNet = networks.IpairNet(para).to(device)
    checkpoint_PPN = torch.load(para.save_dir + 'PPN/best.pth', map_location='cuda:0')
    checkpoint = torch.load(para.save_dir + 'IpairNet/best.pth', map_location='cuda:0')
    PPN.load_state_dict(checkpoint_PPN['model'])
    IpairNet.load_state_dict(checkpoint['IRM'])
    print('Models been loaded successfully.')

    ck1 = {'model': PPN.state_dict()}
    ck2 = {'model': IpairNet.state_dict()}
    torch.save(ck1, 'checkpoints/PPN/best.pth')
    torch.save(ck2, 'checkpoints/IpairNet/best.pth')