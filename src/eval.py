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


COLUMNS = ['row_id','frame','video_id','fish_number','length',
           'species_fourspot','species_grey sole', 'species_other',
           'species_plaice','species_summer','species_windowpane','species_winter']
COLUMNS_VIDEO = ['row_id','frame','video_id','fish_number','length', 'cx', 'cy',
                 'species_fourspot','species_grey sole', 'species_other',
                 'species_plaice','species_summer','species_windowpane','species_winter']


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

    image_batch = []
    image_batch_framenum = []
    eof = False
    for j in range(mc.BATCH_SIZE):
      ret, frame = cap.read()

      if not ret:
        eof = True
        break

      frame_prep = np.expand_dims(preprocess_func(mc, frame), axis=0)
      image_batch.append(frame_prep)
      image_batch_framenum.append(frame_num)
      frame_num += 1

    if not image_batch:
      break

    image_batch = np.concatenate(image_batch, axis=0)

    det_boxes, post_class_probs, det_probs, det_class = sess.run(
      [model.det_boxes, model.post_class_probs, model.det_probs, model.det_class],
      feed_dict={model.image_input: image_batch})

    for b in range(len(image_batch_framenum)):
      best_det_idx = det_probs[b, :].argmax()
      for c in mc.CLASS_NAMES:
        label_id = label_to_id_map[c]
        detections[c].append(post_class_probs[b, best_det_idx, label_id])
      detections['frame'].append(image_batch_framenum[b])
      detections['length'].append(2 * det_boxes[b, best_det_idx, 2]) # note, we predict radius, they need diameter
      detections['cx'].append(det_boxes[b, best_det_idx, 0])
      detections['cy'].append(det_boxes[b, best_det_idx, 1])

    if eof:
      break

  elapsed = time.time() - ts
  print 'Elapsed: {} sec, {:.2f} frames/sec'.format(elapsed, frame_num/elapsed)

  return frame_num, detections


def eval_frames(sess, model, frames_dir, preprocess_func, label_to_id_map):
  mc = model.mc
  detections = defaultdict(list)

  print 'Processing {}'.format(frames_dir)

  ts = time.time()

  frame_num = 0
  for frame_file in os.listdir(frames_dir):
    frame_path = os.path.join(frames_dir, frame_file)
    frame = preprocess_func(mc, cv2.imread(frame_path))

    det_boxes, post_class_probs, det_probs, det_class = sess.run(
      [model.det_boxes, model.post_class_probs, model.det_probs, model.det_class],
      feed_dict={model.image_input: frame[None,:]})

    b = 0 # in-batch idx
    best_det_idx = det_probs[b, :].argmax()
    #post_class_probs[b, best_det_idx] /= post_class_probs[b, best_det_idx].sum()
    for c in mc.CLASS_NAMES:
      label_id = label_to_id_map[c]
      detections[c].append(post_class_probs[b, best_det_idx, label_id])
    detections['frame'].append(frame_file.split('.')[0])
    detections['length'].append(2 * det_boxes[b, best_det_idx, 2]) # note, we predict radius, they need diameter
    detections['cx'].append(det_boxes[b, best_det_idx, 0])
    detections['cy'].append(det_boxes[b, best_det_idx, 1])

    frame_num += 1

  elapsed = time.time() - ts
  print 'Elapsed: {} sec, {:.2f} frames/sec'.format(elapsed, frame_num/elapsed)

  return frame_num, detections


def eval_imdb(sess, model, imdb):
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
  return detections


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
      detections = eval_imdb(model, imdb, results_dir)
      df = pd.DataFrame(detections, columns=COLUMNS)
      df.to_csv(os.path.join(results_dir, '{}.csv'.format(dataset)), index=False)
    elif '_videos' in dataset: # make predictions for a set of videos
      assert os.path.exists(FLAGS.video_names)

      video_ids = []
      video_paths = []
      with open(FLAGS.video_names) as fin:
        for line in fin.readlines():
          line = line.strip()
          if line:
            video_ids.append(line)
            video_paths.append('{}/{}/{}.mp4'.format(FLAGS.data_path, dataset, line))

      for video_id, video_path in zip(video_ids, video_paths):
        frame_num, detections = eval_video(sess, model, video_path,
                                           fish_db.preprocess_image, mc.CLASS_TO_ID)

        detections['video_id'] = [video_id] * frame_num
        df = pd.DataFrame(detections, columns=COLUMNS_VIDEO)
        df.to_csv(os.path.join(results_dir, '{}_{}.csv'.format(dataset, video_id)), index=False)
    elif dataset == 'train_frames':
      for video_id in os.listdir(os.path.join(FLAGS.data_path, 'frames')):
        frames_dir = os.path.join(FLAGS.data_path, 'frames', video_id)
        frame_num, detections = eval_frames(sess, model, frames_dir,
                                            fish_db.preprocess_image, mc.CLASS_TO_ID)
        detections['video_id'] = [video_id] * frame_num
        df = pd.DataFrame(detections, columns=COLUMNS_VIDEO)
        df.to_csv(os.path.join(results_dir, '{}_{}.csv'.format(dataset, video_id)), index=False)


if __name__ == '__main__':
  tf.app.run()
