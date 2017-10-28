import os, sys

if 'src' not in os.listdir('.'):
  os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# We assume OpenCV with FFMPEG support is build and placed in ./opencv
sys.path.insert(0, 'opencv/release/lib')
sys.path.append('metrics')
import cv2
print 'opencv:', cv2.__file__
import numpy as np
import tensorflow as tf
import time
from collections import defaultdict
import pandas as pd

from fish_db import fish_db
from config import *
from nets import *


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_path', 'data', """""")
tf.app.flags.DEFINE_string('dataset', '', """can be {train_foldX, valid_foldX} or {train_videos, test_videos}""")
tf.app.flags.DEFINE_string('video_names', '', """""")
tf.app.flags.DEFINE_string('ckpt_path', '', """""")
tf.app.flags.DEFINE_integer('batch_size', 1, """""")
tf.app.flags.DEFINE_string('gpu', '0', """""")


COLUMNS = ['row_id','frame','video_id','fish_number','length','species_fourspot','species_grey sole',
           'species_other','species_plaice','species_summer','species_windowpane','species_winter']


def eval_video(sess, model, video_path, preprocess_func, label_to_id_map):
  mc = model.mc
  detections = defaultdict(list)

  cap = cv2.VideoCapture(video_path)
  assert cap.isOpened()

  print 'Processing {}'.format(video_path)

  ts = time.time()

  frame_num = 0
  frame_limit = -1
  while True:
    if frame_num == frame_limit:
        break

    ret, frame = cap.read()
    if not ret:
      print 'Processed {} frames'.format(frame_num)
      break

    frame_prep = np.expand_dims(preprocess_func(mc, frame), axis=0)

    det_boxes, post_class_probs, det_probs, det_class = sess.run(
      [model.det_boxes, model.post_class_probs, model.det_probs, model.det_class],
      feed_dict={model.image_input: frame_prep})

    b = 0 # in-batch idx
    best_det_idx = det_probs[b, :].argmax()
    #post_class_probs[b, best_det_idx] /= post_class_probs[b, best_det_idx].sum()
    for c in mc.CLASS_NAMES:
      label_id = label_to_id_map[c]
      detections[c].append(post_class_probs[b, best_det_idx, label_id])
    detections['frame'].append(frame_num)
    detections['length'].append(2 * det_boxes[b, best_det_idx, 2]) # note, we predict radius, they need diameter

    # cx, cy, r = det_boxes[b, best_det_idx]
    # cv2.circle(frame, (int(cx), int(cy)), int(r), (255,0,0), 2)
    # cv2.imwrite('/tmp/{}.jpg'.format(os.path.basename(video_path)+str(frame_num)), frame)
    # print frame_num, mc.CLASS_NAMES[det_class[b, best_det_idx]], det_probs[b, best_det_idx]

    frame_num += 1

  elapsed = time.time() - ts
  print 'Elapsed: {} sec, {:.2f} frames/sec'.format(elapsed, frame_num/elapsed)

  return frame_num, detections


def eval_frames(sess, model, imdb, results_dir):
  mc = model.mc

  ts = time.time()

  detections = defaultdict(list)
  n_total = len(imdb._image_idx)
  for i in range(n_total):
    images, video_ids, frames_ids = imdb.read_image_batch(shuffle=False)
    assert len(images) == 1

    det_boxes, post_class_probs, det_probs, det_class = sess.run(
        [model.det_boxes, model.post_class_probs, model.det_probs, model.det_class],
        feed_dict={model.image_input: images[0][np.newaxis,...]})

    b = 0 # in-batch idx
    best_det_idx = det_probs[b, :].argmax()
    for c in mc.CLASS_NAMES:
      label_id = imdb._label_to_id[c]
      detections[c].append(post_class_probs[b, best_det_idx, label_id])
    detections['frame'].append(frames_ids[b])
    detections['video_id'].append(video_ids[b])
    detections['length'].append(2 * det_boxes[b, best_det_idx, 2]) # note, we predict radius, they need diameter

    if i % 100 == 0:
      print '{}/{}'.format(i, n_total)

  print 'elapsed: ', time.time()-ts

  detections['row_id'] = range(len(detections['frame']))
  df = pd.DataFrame(detections, columns=COLUMNS)
  df.to_csv(os.path.join(results_dir, '{}.csv'.format(FLAGS.dataset)), index=False)


def main(args=None):
  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  results_dir = os.path.join(os.path.dirname(FLAGS.ckpt_path), 'results')
  if not os.path.exists(results_dir):
    os.mkdir(results_dir)

  with tf.Graph().as_default() as g:
    mc = fish_res50_config()
    mc.BATCH_SIZE = FLAGS.batch_size
    mc.LOAD_PRETRAINED_MODEL = False
    model = ResNet50ConvDet(mc)

    sess_config = tf.ConfigProto()
    sess_config.allow_soft_placement = True
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    saver = tf.train.Saver(model.model_params)
    saver.restore(sess, FLAGS.ckpt_path)

    dataset = FLAGS.dataset
    if 'train_fold' in dataset or 'valid_fold' in dataset: # make predictions for validation
      imdb = fish_db(FLAGS.dataset, FLAGS.data_path, mc, keep_nofish_frames=True)
      eval_frames(model, imdb, results_dir)
    elif '_videos' in dataset:
      assert os.path.exists(FLAGS.video_names)

      video_ids = []
      video_paths = []
      with open(FLAGS.video_names) as fin:
        for line in fin.readlines():
          line = line.strip()
          if line:
            video_ids.append(line)
            video_paths.append('{}/{}/{}.mp4'.format(FLAGS.data_path, FLAGS.dataset, line))

      for video_id, video_path in zip(video_ids, video_paths):
        frame_num, detections = eval_video(sess, model, video_path,
                                           fish_db.preprocess_image, mc.CLASS_TO_ID)

        detections['video_id'] = [video_id] * frame_num
        df = pd.DataFrame(detections, columns=COLUMNS)
        df.to_csv(os.path.join(results_dir, '{}_{}.csv'.format(FLAGS.dataset, video_id)), index=False)


if __name__ == '__main__':
  tf.app.run()
