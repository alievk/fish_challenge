# 'N+1 fish, N+2 fish' dataset class

import numpy as np
import cv2
import os
import pandas as pd

from utils.util import batch_iou_circle as batch_iou

pjoin = os.path.join
exists = os.path.exists


class fish_db(object):
  def __init__(self, dataset, data_root, mc):
    self._dataset = dataset
    self._data_root = data_root
    self.mc = mc

    # batch reader
    self._perm_idx = None
    self._cur_idx = 0

    self._image_idx, self._objects = self._load_annotation(True)
    self._shuffle_image_idx()

  def _load_annotation(self, keep_nofish_frames=False):
    mc = self.mc

    anno_path = pjoin(self._data_root, self._dataset + '.csv')
    assert exists(anno_path)
    anno = pd.read_csv(anno_path)

    anno.drop

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

    self._label_to_id = dict(zip(mc.CLASS_NAMES, range(mc.CLASSES)))

    return image_idx, objects

  def _shuffle_image_idx(self):
    self._perm_idx = [self._image_idx[i] for i in
        np.random.permutation(np.arange(len(self._image_idx)))]
    self._cur_idx = 0

  def _image_path_at(self, idx):
    _, video_name, frame = idx
    return pjoin(self._data_root, 'frames', video_name, str(frame) + '.jpg')

  def read_batch(self, shuffle=True, normalize=True):
    mc = self.mc

    if shuffle:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        self._shuffle_image_idx()
      batch_idx = self._perm_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
      self._cur_idx += mc.BATCH_SIZE
    else:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        batch_idx = self._image_idx[self._cur_idx:] \
            + self._image_idx[:self._cur_idx + mc.BATCH_SIZE-len(self._image_idx)]
        self._cur_idx += mc.BATCH_SIZE - len(self._image_idx)
      else:
        batch_idx = self._image_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
        self._cur_idx += mc.BATCH_SIZE

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

      # resize
      if im.shape[:2] != (mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH):
        im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))

      if normalize:
        im = im.astype(np.float32, copy=False)
        im -= mc.BGR_MEANS

      raw_idx = idx[0]
      cx, cy, r, label = self._objects[raw_idx]

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

        label_per_image.append(self._label_to_id[label])
        bbox_per_image.append([cx, cy, r])

      image_per_batch.append(im)
      label_per_batch.append(label_per_image)
      bbox_per_batch.append(bbox_per_image)
      aidx_per_batch.append(aidx_per_image)
      delta_per_batch.append(delta_per_image)

    return image_per_batch, label_per_batch, delta_per_batch, \
           aidx_per_batch, bbox_per_batch