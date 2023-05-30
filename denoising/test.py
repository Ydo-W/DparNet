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
    DparNet = networks.DparNet(para).to(device)
    checkpoint_PPN = torch.load(para.save_dir + 'PPN/best.pth', map_location='cuda:0')
    checkpoint = torch.load(para.save_dir + 'DparNet/best.pth', map_location='cuda:0')
    PPN.load_state_dict(checkpoint_PPN['model'])
    DparNet.load_state_dict(checkpoint['model'])
    print('Models been loaded successfully.')

    # Paths
    input_dir = para.data_root + 'deg_seqs/'
    result_video_path = para.results_dir + 'restored_seqs/'
    os.makedirs(result_video_path, exist_ok=True)

    # Videos' info
    input_videos = os.listdir(input_dir)
    input_videos.sort(key=lambda x: int(x[:-4]))
    frame_h, frame_w = 256, 448
    test_start_id = int(len(input_videos)*0.8) + 1

    # Startint test
    for video_idx in range(test_start_id, len(input_videos)+1):
        input_video_path = input_dir + '{:06d}.avi'.format(video_idx)
        print('processing {:06d}.avi'.format(video_idx), '...')
        input_video = cv2.VideoCapture(input_video_path)
        out_video = cv2.VideoWriter(result_video_path + '{:06d}.avi'.format(video_idx),
                                    cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 10.0,
                                    (frame_w, frame_h), isColor=True)

        # --------- input data ------------------------------------------
        input_seq = []
        while input_video.isOpened():
            rval, frame_input = input_video.read()
            if not rval:
                break
            frame_input = np.ascontiguousarray(frame_input.transpose((2, 0, 1))) / 255.0
            input_seq.append(frame_input[np.newaxis, :])
        input_seq = np.concatenate(input_seq, axis=0)
        numFrames = len(input_seq)

        # Start
        test_frames = para.frame_length
        start, end = 0, test_frames
        process_number = 0
        while True:
            torch.cuda.empty_cache()
            theInput = input_seq[start: end, :, :, :][np.newaxis, :]
            PPN.eval(), DparNet.eval()
            with torch.no_grad():
                theInput = torch.from_numpy(theInput).float().to(device)
                para_pred = PPN(theInput)
                restoration_data = DparNet(theInput, para_pred) * 255.0
                out_seq = restoration_data.clamp(0, 255).squeeze()

                for frame_idx in range(test_frames - 2 * para.neighboring_frames):
                    if process_number < start + frame_idx + 1:
                        img_deTurbulence = out_seq[frame_idx]
                        img_deTurbulence = img_deTurbulence.detach().cpu().numpy().transpose((1, 2, 0))
                        out_video.write(img_deTurbulence.astype(np.uint8))
                        process_number += 1

            if end == numFrames:
                break
            else:
                start = end - 2 * para.neighboring_frames
                end = start + test_frames
                if end > numFrames:
                    end = numFrames
                    start = end - test_frames

        input_video.release()
        out_video.release()