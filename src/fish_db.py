# 'N+1 fish, N+2 fish' dataset class

import numpy as np
import cv2
import os
import pandas as pd

from imgaug import augmenters as iaa

from utils.util import batch_iou_circle as batch_iou

pjoin = os.path.join
exists = os.path.exists


class fish_db(object):
  def __init__(self, dataset, data_root, mc, keep_nofish_frames=True):
    self._dataset = dataset
    self._data_root = data_root
    self.mc = mc

    # batch reader
    self._perm_idx = None
    self._cur_img_idx = 0

    self._cur_video_idx = 0

    self._image_idx, self._objects = self._load_annotation(keep_nofish_frames)
    self._shuffle_image_idx()

  def _load_annotation(self, keep_nofish_frames=False):
    mc = self.mc

    anno_path = pjoin(self._data_root, self._dataset + '.csv')
    assert exists(anno_path)
    anno = pd.read_csv(anno_path)

    self._video_ids = list(set(anno.video_id))

    if not keep_nofish_frames:
      anno.dropna(inplace=True)
      anno.reset_index(inplace=True)

    image_idx = zip(anno.index, anno.video_id, anno.frame)

    cx = 0.5*(anno.x1 + anno.x2)
    cy = 0.5*(anno.y1 + anno.y2)
    r = 0.5*anno.length
    labels = anno.iloc[:, -7:].idxmax(axis=1)
    labels[anno.fish_number.isnull()] = 'species_none'
    objects = zip(cx, cy, r, labels)

    return image_idx, objects

  def _shuffle_image_idx(self):
    self._perm_idx = [self._image_idx[i] for i in
        np.random.permutation(np.arange(len(self._image_idx)))]
    self._cur_img_idx = 0

  def _image_path_at(self, idx):
    _, video_name, frame = idx
    return pjoin(self._data_root, 'frames', video_name, str(frame) + '.jpg')

  @staticmethod
  def preprocess_image(mc, im):
    # resize
    if im.shape[:2] != (mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH):
      im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))

    # normalize
    im = im.astype(np.float32)
    im -= mc.BGR_MEANS

    return im

  @staticmethod
  def augment(im, p_fliplr=0.5, p_flipud=0.5, p_blur=0.5, p_brightness=0.5, p_contrast=0.5):
    has_fliplr = p_fliplr > np.random.rand(1)
    has_flipud = p_flipud > np.random.rand(1)
    has_blur = p_blur > np.random.rand(1)
    has_brightness = p_brightness > np.random.rand(1)
    has_contrast = p_contrast > np.random.rand(1)
    
    # fliplr
    if has_fliplr:
      im = iaa.Fliplr(p=1, deterministic=True).augment_image(im)
      
    # flipud
    if has_flipud:
      im = iaa.Flipud(p=1, deterministic=True).augment_image(im)

    # gaussian blur
    if has_blur:
      im = iaa.GaussianBlur(sigma=(1, 2.5), deterministic=True).augment_image(im)

    if has_brightness:
      im = iaa.Multiply(mul=(0.5, 1.2), deterministic=True).augment_image(im)

    if has_contrast:
      im = iaa.ContrastNormalization(alpha=(0.4, 1.2), deterministic=True).augment_image(im)

    return im, (has_fliplr, has_flipud, has_blur, has_brightness, has_contrast)

  def read_image_batch(self, shuffle=True):
    mc = self.mc

    if shuffle:
      if self._cur_img_idx + mc.BATCH_SIZE >= len(self._image_idx):
        self._shuffle_image_idx()
      batch_idx = self._perm_idx[self._cur_img_idx:self._cur_img_idx + mc.BATCH_SIZE]
      self._cur_img_idx += mc.BATCH_SIZE
    else:
      if self._cur_img_idx + mc.BATCH_SIZE >= len(self._image_idx):
        batch_idx = self._image_idx[self._cur_img_idx:] \
            + self._image_idx[:self._cur_img_idx + mc.BATCH_SIZE - len(self._image_idx)]
        self._cur_img_idx += mc.BATCH_SIZE - len(self._image_idx)
      else:
        batch_idx = self._image_idx[self._cur_img_idx:self._cur_img_idx + mc.BATCH_SIZE]
        self._cur_img_idx += mc.BATCH_SIZE

    images = []
    video_ids = []
    frame_ids = []
    for idx in batch_idx:
      im_path = self._image_path_at(idx)
      assert exists(im_path), '{} not found'.format(im_path)
      im = self.preprocess_image(mc, cv2.imread(im_path))
      images.append(im)
      video_ids.append(idx[1])
      frame_ids.append(idx[2])

    return images, video_ids, frame_ids

  def read_batch(self, shuffle=True):
    mc = self.mc

    if shuffle:
      if self._cur_img_idx + mc.BATCH_SIZE >= len(self._image_idx):
        self._shuffle_image_idx()
      batch_idx = self._perm_idx[self._cur_img_idx:self._cur_img_idx + mc.BATCH_SIZE]
      self._cur_img_idx += mc.BATCH_SIZE
    else:
      if self._cur_img_idx + mc.BATCH_SIZE >= len(self._image_idx):
        batch_idx = self._image_idx[self._cur_img_idx:] \
            + self._image_idx[:self._cur_img_idx + mc.BATCH_SIZE - len(self._image_idx)]
        self._cur_img_idx += mc.BATCH_SIZE - len(self._image_idx)
      else:
        batch_idx = self._image_idx[self._cur_img_idx:self._cur_img_idx + mc.BATCH_SIZE]
        self._cur_img_idx += mc.BATCH_SIZE

    image_per_batch = []
    label_per_batch = []
    bbox_per_batch  = []
    delta_per_batch = []
    aidx_per_batch  = []

    for idx in batch_idx:
      # load the image
      im_path = self._image_path_at(idx)
      assert exists(im_path), '{} not found'.format(im_path)
      im = cv2.imread(im_path)

      im, aug_flag = self.augment(im)
      has_fliplr, has_flipud, _, _, _ = aug_flag

      im = self.preprocess_image(mc, im)

      raw_idx = idx[0]
      cx, cy, r, label = self._objects[raw_idx]

      if has_fliplr:
        cx = im.shape[1] - cx - 1
        
      if has_flipud:
        cy = im.shape[0] - cy - 1

      label_per_image = []
      bbox_per_image = []
      aidx_per_image = []
      delta_per_image = []
      if label != 'species_none':
        overlaps = batch_iou(mc.ANCHOR_BOX, [cx, cy, r])
        aidx = np.argmax(overlaps)
        aidx_per_image.append(aidx)

        delta = [0] * 3
        acx, acy, ar = mc.ANCHOR_BOX[aidx]
        delta[0] = (cx - acx)/ar
        delta[1] = (cy - acy)/ar
        delta[2] = np.log(r/ar)
        delta_per_image.append(delta)

        label_per_image.append(mc.CLASS_TO_ID[label])
        bbox_per_image.append([cx, cy, r])

      image_per_batch.append(im)
      label_per_batch.append(label_per_image)
      bbox_per_batch.append(bbox_per_image)
      aidx_per_batch.append(aidx_per_image)
      delta_per_batch.append(delta_per_image)

    return image_per_batch, label_per_batch, delta_per_batch, \
           aidx_per_batch, bbox_per_batch

  def num_videos(self):
    return len(self._video_ids)

  def _video_path_at(self, idx):
    return pjoin(self._data_root, 'train_videos', self._video_ids[idx] + '.mp4')

  def next_video(self):
    idx = self._cur_video_idx
    self._cur_video_idx = (self._cur_video_idx + 1) % len(self._video_ids)
    return self._video_path_at(idx), idx
