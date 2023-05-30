import os
import cv2
from para import Parameter
from skimage.metrics import normalized_root_mse as computate_NRMSE
from skimage.metrics import peak_signal_noise_ratio as computate_PSNR
from skimage.metrics import structural_similarity as computate_SSIM
from skimage.metrics import variation_of_information as computate_VI

if __name__ == '__main__':
    para = Parameter().args

    # Paths
    gt_video_dir = para.data_root + 'test/clean_seqs/'
    result_video_path = para.results_dir + 'restored_seqs/'
    gt_videos, res_videos = os.listdir(gt_video_dir), os.listdir(result_video_path)

    frame_count = 0
    neighboring_frames = 2
    NRMSE, PSNR, SSIM, VI = 0.0, 0.0, 0.0, 0.0
    for video_idx in range(1, len(gt_videos)+1):
        print('evaluating {:06d}.avi'.format(video_idx), '...')
        # --------- restoration evaluate ------------------------------------------
        gt_video_path = gt_video_dir + '{:06d}.avi'.format(video_idx)
        res_video_path = result_video_path + '{:06d}.avi'.format(video_idx)
        gt_video, res_video = cv2.VideoCapture(gt_video_path), cv2.VideoCapture(res_video_path)
        frame_length = int(gt_video.get(7))
        video_NRMSE, video_PSNR, video_SSIM, video_VI = 0.0, 0.0, 0.0, 0.0
        for frame_index in range(frame_length):
            rval1, frame1 = gt_video.read()
            if neighboring_frames <= frame_index < frame_length - neighboring_frames:
                rval2, frame2 = res_video.read()
                if 7 <= frame_index < frame_length - 7:
                    frame_count += 1
                    frame1, frame2 = frame1[:, :, 0], frame2[:, :, 0]
                    video_NRMSE = video_NRMSE + computate_NRMSE(frame1, frame2)
                    video_PSNR = video_PSNR + computate_PSNR(frame1, frame2, data_range=255.)
                    video_SSIM = video_SSIM + computate_SSIM(frame1, frame2, data_range=255.)
                    video_VI = video_VI + computate_VI(frame1, frame2)[0] + computate_VI(frame1, frame2)[1]
        NRMSE, PSNR, SSIM, VI = NRMSE+video_NRMSE, PSNR+video_PSNR, SSIM+video_SSIM, VI+video_VI

    print('Video restoration -- NRMSE:{:.4f}, PSNR:{:.4f}, SSIM:{:.4f}, VI:{:.4f}'
          .format(NRMSE/frame_count, PSNR/frame_count, SSIM/frame_count, VI/frame_count))
