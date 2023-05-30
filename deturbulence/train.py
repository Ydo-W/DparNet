import os
import time
import utils
import torch
import random
import dataset
import networks
import datetime
import numpy as np
import torch.nn as nn
from para import Parameter
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    para = Parameter().args
    device = torch.device('cuda')

    # Setting the random seed
    torch.manual_seed(para.seed)
    torch.cuda.manual_seed(para.seed)
    random.seed(para.seed)
    np.random.seed(para.seed)

    # Dataset
    train_dataset = dataset.DeTurbulenceDataset(para, 'train/')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=para.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8
    )
    print('length_of_train: ', len(train_dataset))

    valid_dataset = dataset.DeTurbulenceDataset(para, 'valid/')
    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=8
    )
    print('length_of_valid: ', len(valid_dataset))

    # Networks
    PPN = networks.PPN(para).to(device)
    DparNet = networks.DparNet(para).to(device)

    # Load the pretrained PMM
    checkpoint_PPN = torch.load(para.save_dir + 'PPN/best.pth', map_location='cuda:0')
    PPN.load_state_dict(checkpoint_PPN['model'])
    for param in PPN.parameters():
        param.requires_grad = False

    # Setting the optimizers
    lr = 1e-4
    opt_IRM = optim.Adam(DparNet.parameters(), lr)

    # Loss functions and metrics
    criterion = nn.L1Loss().to(device)
    perception = utils.PerceptualLoss().to(device)
    metrics = utils.PSNR()

    train_PSNR, valid_PSNR = [], []
    train_l1, valid_l1 = [], []
    min_PSNR = 10
    date = datetime.datetime.now()
    model_path = para.save_dir + 'IparNet_tiny/'
    os.makedirs(model_path, exist_ok=True)

    # Training
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    for epoch in range(1, 101):
        start = time.time()
        a = utils.train(device, train_loader, DparNet, PPN, opt_IRM, criterion, perception, metrics, epoch, para.batch_size)
        b = utils.valid(device, valid_loader, DparNet, PPN, metrics)
        train_PSNR.append(a), valid_PSNR.append(b)
        end = time.time()

        print('time:{:.2f}s'.format(end - start))
        checkpoint = \
            {
                'model': DparNet.state_dict(),
            }
        torch.save(checkpoint, model_path + '/latest.pth')
        if valid_PSNR[-1] >= min_PSNR:
            torch.save(checkpoint, model_path + '/best.pth')
            min_PSNR = valid_PSNR[-1]

        # Plotting
        plt.switch_backend('agg')
        np.savetxt(model_path + '/train_PSNR.txt', train_PSNR)
        np.savetxt(model_path + '/valid_PSNR.txt', valid_PSNR)
        plt.figure(), plt.plot(train_PSNR), plt.plot(valid_PSNR), plt.savefig(model_path + '/PSNR.jpg'), plt.close()
