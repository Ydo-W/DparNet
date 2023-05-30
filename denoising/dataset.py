import cv2
import torch
import utils
import random
import numpy as np
import scipy.io as scio
from os.path import join
from torch.utils.data import Dataset


class DeNoisingDataset(Dataset):
    def __init__(self, para, data_type):
        self.data_type = data_type
        self.clean_videos_path = join(para.data_root, 'clean_seqs/')
        self.degraded_videos_path = join(para.data_root, 'deg_seqs/')
        self.para_labels_path = join(para.data_root, 'deg_paras/')
        self.clean_videos = utils.get_files(self.clean_videos_path)
        self.videos_num = len(self.clean_videos)
        self.H, self.W = 256, 448
        self.N_frames = para.frame_length
        self.nbor_frames = para.neighboring_frames
        self.crop_size = 256

    def __getitem__(self, idx):
        # idx of data to use
        if self.data_type == 'train/':
            seq_idx = idx + 1
            flip_lr_flag = random.randint(0, 1)
            flip_ud_flag = random.randint(0, 1)
            rotate_flag = random.randint(0, 1)
        else:
            seq_idx = idx + int(self.videos_num * 0.6) + 1
            flip_lr_flag, flip_ud_flag, rotate_flag = 0, 0, 0

        crop_h_idx = random.randint(0, self.H - self.crop_size)
        crop_w_idx = random.randint(0, self.W - self.crop_size)
        sample = {'flip_lr': flip_lr_flag, 'flip_ud': flip_ud_flag, 'rotate': rotate_flag}

        input_data, gt_data, para_data = self.get_data(seq_idx, crop_h_idx, crop_w_idx, sample)
        return input_data, gt_data, para_data

    def __len__(self):
        if self.data_type == 'train/':
            return int(self.videos_num * 0.6)
        else:
            return int(self.videos_num * 0.2)

    def get_data(self, seq_idx, crop_h_idx, crop_w_idx, sample):
        # Data paths
        gt_video_path = self.clean_videos_path + '{:06d}.avi'.format(seq_idx)
        input_video_path = self.degraded_videos_path + '{:06d}.avi'.format(seq_idx)
        para_path = self.para_labels_path + '{:06d}.mat'.format(seq_idx)

        # Sequences
        gt_video = cv2.VideoCapture(gt_video_path)
        input_video = cv2.VideoCapture(input_video_path)

        gt_data = np.zeros((self.N_frames, 3, self.crop_size, self.crop_size))
        input_data = np.zeros((self.N_frames, 3, self.crop_size, self.crop_size))
        para_data = np.zeros((self.crop_size, self.crop_size))

        # Reading data
        for i in range(self.N_frames):
            rval1, gt_frame = gt_video.read()
            rval2, input_frame = input_video.read()
            gt_block = gt_frame[crop_h_idx: crop_h_idx + self.crop_size, crop_w_idx: crop_w_idx + self.crop_size]
            input_block = input_frame[crop_h_idx: crop_h_idx + self.crop_size, crop_w_idx: crop_w_idx + self.crop_size]
            if sample['flip_lr'] == 1:
                gt_block = np.fliplr(gt_block)
                input_block = np.fliplr(input_block)
            if sample['flip_ud'] == 1:
                gt_block = np.flipud(gt_block)
                input_block = np.flipud(input_block)
            if sample['rotate'] == 1:
                gt_block = gt_block.transpose((1, 0, 2))
                input_block = input_block.transpose((1, 0, 2))
            gt_data[i, :, :, :] = np.ascontiguousarray(gt_block.transpose((2, 0, 1)))
            input_data[i, :, :, :] = np.ascontiguousarray(input_block.transpose((2, 0, 1)))
        label_dic = scio.loadmat(para_path)
        para_dic = label_dic['para_field'][crop_h_idx: crop_h_idx + self.crop_size, crop_w_idx: crop_w_idx + self.crop_size]

        # Normalization
        para_data[:, :] = para_dic / 100
        gt_data = gt_data / 255.0
        input_data = input_data / 255.0

        if sample['flip_lr'] == 1:
            para_data = np.fliplr(para_data)
        if sample['flip_ud'] == 1:
            para_data = np.flipud(para_data)
        if sample['rotate'] == 1:
            para_data = para_data.transpose((1, 0))

        para_data = np.ascontiguousarray(para_data[np.newaxis, :])

        # To tensor
        gt_data = torch.from_numpy(gt_data).float()
        input_data = torch.from_numpy(input_data).float()
        para_data = torch.from_numpy(para_data).float()
        gt_data = gt_data[self.nbor_frames: self.N_frames - self.nbor_frames]
        return input_data, gt_data, para_data  # (F, 3, 256, 256), (F-4, 3, 256, 256), (1, 256, 256)
