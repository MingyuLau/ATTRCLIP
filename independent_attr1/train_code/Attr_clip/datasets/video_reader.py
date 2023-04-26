import cv2
import torch
import random
import os
import numpy as np
import av
# import decord


# def sample_frames(num_frames, vlen, sample='rand', fix_start=None):
#     acc_samples = min(num_frames, vlen)
#     intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
#     ranges = []
#     for idx, interv in enumerate(intervals[:-1]):
#         ranges.append((interv, intervals[idx + 1] - 1))
#     if sample == 'rand':
#         frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
#     elif fix_start is not None:
#         frame_idxs = [x[0] + fix_start for x in ranges]
#     elif sample == 'uniform':
#         frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
#     else:
#         raise NotImplementedError

#     return frame_idxs

# def sample_frames_clips(start, end, vlen, acc_samples):
#     start = max(0, start)
#     end = min(vlen, end)

#     intervals = np.linspace(start=start, stop=end, num=int(acc_samples) + 1).astype(int)
#     ranges = []
#     for idx, interv in enumerate(intervals[:-1]):
#         ranges.append((interv, intervals[idx + 1] - 1))
#     frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
#     return frame_idxs

def sample_frames_start_end(num_frames, start, end, sample='rand', fix_start=None):
    acc_samples = min(num_frames, end)
    intervals = np.linspace(start=start, stop=end, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == 'rand':
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == 'uniform':
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    else:
        raise NotImplementedError

    return frame_idxs

# def read_frames_cv2(video_path, num_frames, sample='rand', fix_start=None):
#     cap = cv2.VideoCapture(video_path)
#     assert (cap.isOpened())
#     vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     # get indexes of sampled frames
#     frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
#     frames = []
#     success_idxs = []
#     for index in frame_idxs:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
#         ret, frame = cap.read()
#         if ret:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = torch.from_numpy(frame)
#             # (H x W x C) to (C x H x W)
#             frame = frame.permute(2, 0, 1)
#             frames.append(frame)
#             success_idxs.append(index)
#         else:
#             pass
#             # print(frame_idxs, ' fail ', index, f'  (vlen {vlen})')

#     frames = torch.stack(frames).float() / 255
#     cap.release()
#     return frames, success_idxs

def read_frames_cv2_egoclip(video_path_1, video_path_2, num_frames, sample,
                            start_sec, end_sec, bound_sec):
    if video_path_1 == video_path_2:
        cap1 = cv2.VideoCapture(video_path_1)
        cap2 = cap1
        vlen1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        vlen2 = vlen1
        assert (cap1.isOpened())
    else:   # some clips may span two segments.
        cap1 = cv2.VideoCapture(video_path_1)
        cap2 = cv2.VideoCapture(video_path_2)
        vlen1 = int(cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        vlen2 = int(cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        assert (cap1.isOpened())
        assert (cap2.isOpened())

    # get indexes of sampled frames
    start_f = max(0, int(start_sec * 30))
    end_f = max(0, int(end_sec * 30))
    bound_f = int(bound_sec * 30)
    frame_idxs = sample_frames_start_end(num_frames, start_f, end_f, sample=sample)

    frames = []
    success_idxs = []
    for index in frame_idxs:
        _index = index % (600 * 30)
        if index > bound_f: # frame from the last video
            _index = min(_index, vlen2)
            cap2.set(cv2.CAP_PROP_POS_FRAMES, _index - 1)
            ret, frame = cap2.read()
        else:   # frame from the first video
            _index = min(_index, vlen1)
            cap1.set(cv2.CAP_PROP_POS_FRAMES, _index - 1)
            ret, frame = cap1.read()

        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass

    while len(frames) < num_frames: # complete the frame
        frames.append(frames[-1])

    frames = torch.stack(frames).float() / 255
    cap1.release()
    cap2.release()
    return frames, success_idxs

# def read_frames_cv2_epic(video_path, start_frame, stop_frame, num_frames, sample='rand', fix_start=None):
#     # get indexes of sampled frames
#     frame_idxs = sample_frames_start_end(num_frames, start_frame, stop_frame, sample=sample, fix_start=fix_start)
#     frames = []
#     success_idxs = []
#     for index in frame_idxs:
#         img_name = 'frame_' + str(index).zfill(10) + '.jpg'
#         frame = cv2.imread(os.path.join(video_path, img_name),cv2.COLOR_BGR2RGB)

#         frame = torch.from_numpy(frame)
#         # (H x W x C) to (C x H x W)
#         frame = frame.permute(2, 0, 1)
#         frames.append(frame)
#         success_idxs.append(index)

#     frames = torch.stack(frames).float() / 255
#     return frames, success_idxs

# def read_frames_cv2_charades(video_path, num_frames, sample, start_sec=None, end_sec=None):
#     cap = cv2.VideoCapture(video_path)
#     assert (cap.isOpened())
#     vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     fps = cap.get(5)

#     # get indexes of sampled frames
#     if not start_sec and not end_sec:
#         frame_idxs = sample_frames(num_frames, vlen, sample=sample)
#     else:
#         start_f = max(0, int(start_sec * fps))
#         end_f = min(int(end_sec * fps), vlen)
#         frame_idxs = sample_frames_start_end(num_frames, start_f, end_f, sample=sample)

#     frames = []
#     success_idxs = []
#     for index in frame_idxs:
#         cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
#         ret, frame = cap.read()
#         if ret:
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             frame = torch.from_numpy(frame)
#             # (H x W x C) to (C x H x W)
#             frame = frame.permute(2, 0, 1)
#             frames.append(frame)
#             success_idxs.append(index)
#         else:
#             pass

#     frames = torch.stack(frames).float() / 255
#     cap.release()
#     return frames, success_idxs

# def read_frames_av(video_path, num_frames, sample='rand', fix_start=None):
#     reader = av.open(video_path)
#     frames = []
#     try:
#         frames = [torch.from_numpy(f.to_rgb().to_ndarray()) for f in reader.decode(video=0)]
#     except (RuntimeError, ZeroDivisionError) as exception:
#         print('{}: WEBM reader cannot open {}. Empty '
#               'list returned.'.format(type(exception).__name__, video_path))
#     vlen = len(frames)
#     frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
#     frames = torch.stack([frames[idx] for idx in frame_idxs]).float() / 255
#     frames = frames.permute(0, 3, 1, 2)
#     return frames, frame_idxs

# decord.bridge.set_bridge("torch")

# def read_frames_decord(video_path, num_frames, sample='rand', fix_start=None):
#     video_reader = decord.VideoReader(video_path, num_threads=1)
#     vlen = len(video_reader)
#     frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
#     video_reader.skip_frames(1)
#     frames = video_reader.get_batch(frame_idxs)

#     frames = frames.float() / 255
#     frames = frames.permute(0, 3, 1, 2)
#     return frames, frame_idxs

# def read_frames_decord_start_end(video_path, start, end, num_frames):
#     video_reader = decord.VideoReader(video_path, num_threads=1)
#     vlen = len(video_reader)
#     frame_idxs = sample_frames_clips(start, end, vlen, num_frames + 1)
#     video_reader.skip_frames(1)
#     frames = video_reader.get_batch(frame_idxs)

#     frames = frames.float() / 255
#     frames = frames.permute(0, 3, 1, 2)
#     return frames, frame_idxs
    


video_reader = {
    # 'av': read_frames_av,
    # 'cv2': read_frames_cv2,
    # 'cv2_epic': read_frames_cv2_epic,
    # 'cv2_charades': read_frames_cv2_charades,
    'cv2_egoclip': read_frames_cv2_egoclip,
    # 'decord': read_frames_decord,
    # 'decord_start_end': read_frames_decord_start_end,
}