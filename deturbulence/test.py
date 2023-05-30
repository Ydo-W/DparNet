import os
import cv2
import torch
import networks
import numpy as np
from para import Parameter


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'  # windows
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
    test_path = para.data_root + 'test/'
    input_dir = test_path + 'deg_seqs/'
    result_video_path = para.results_dir + 'restored_seqs/'
    os.makedirs(result_video_path, exist_ok=True)

    # Videos' info
    input_videos = os.listdir(input_dir)
    frame_h, frame_w = 480, 640

    # Startint test
    for video_idx in range(len(input_videos)):
        input_video_path = input_dir + input_videos[video_idx]
        print('processing', input_videos[video_idx], '...')
        input_video = cv2.VideoCapture(input_video_path)
        out_video = cv2.VideoWriter(result_video_path + input_videos[video_idx],
                                    cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 25.0,
                                    (frame_w, frame_h), isColor=False)

        # --------- input data ------------------------------------------
        input_seq = []
        while input_video.isOpened():
            rval, frame_input = input_video.read()
            if not rval: break
            frame_input = frame_input[:, :, 0] / 255.
            frame_input = frame_input[np.newaxis, :]
            input_seq.append(frame_input[np.newaxis, :])
        input_seq = np.concatenate(input_seq, axis=0)
        numFrames = len(input_seq)

        # Start
        test_frames = para.frame_length
        start, end = 0, test_frames
        para_nums, process_number = 0, 0
        while True:
            torch.cuda.empty_cache()
            theInput = np.concatenate((input_seq[start: end, :, :, :]), axis=0)[np.newaxis, :]
            theInput = np.expand_dims(theInput, 2)
            DparNet.eval(), PPN.eval()
            with torch.no_grad():

                theInput = torch.from_numpy(theInput).float().to(device)
                para_pred = PPN(theInput)
                restoration = DparNet(theInput, para_pred) * 255.0
                out_seq = restoration.clamp(0, 255).squeeze()

            for frame_idx in range(test_frames - 2 * para.neighboring_frames):
                if process_number < start + frame_idx + 1:
                    img_deTurbulence = out_seq[frame_idx][np.newaxis, :]
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
