# This script saves all annotated frames from the videos

import cv2
import os
import pandas as pd

pjoin = os.path.join
exists = os.path.exists

data_root = '../data'
video_dir = pjoin(data_root, 'train_videos')
save_frames_dir = pjoin(data_root, 'frames')
anno_path = pjoin(data_root, 'training.csv')

def save_frames_from(video_name, frames, frames_dir):
  assert len(frames)

  if not exists(frames_dir):
    os.mkdir(frames_dir)

  video_path = pjoin(video_dir, video_name + '.mp4')
  assert exists(video_path)
  cap = cv2.VideoCapture(video_path)
  assert cap.isOpened(), 'Could not open {}'.format(video_path)

  assert len(frames) == len(set(frames)), 'duplicate frames'
  frames = sorted(frames)

  iframe = 0
  while True:
    ret, frame = cap.read()
    assert ret, 'not all frames found'

    if iframe == frames[0]:
      frame_path = pjoin(frames_dir, str(iframe) + '.jpg')
      cv2.imwrite(frame_path, frame)

      if len(frames) > 1:
        frames = frames[1:]
      else:
        break

    iframe += 1


if __name__ == '__main__':
  assert exists(anno_path)

  anno = pd.read_csv(anno_path)

  video_names = anno.video_id.unique()

  if not exists(save_frames_dir):
    os.mkdir(save_frames_dir)

  for i, vn in enumerate(video_names):
    frames_dir = pjoin(data_root, save_frames_dir, vn)

    frames = anno[anno.video_id == vn].frame.tolist()

    print 'Processing {}, {}/{}'.format(vn, i+1, len(video_names))
    save_frames_from(vn, frames, frames_dir)


