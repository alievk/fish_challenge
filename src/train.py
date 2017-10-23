from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
from datetime import datetime
import threading

import cv2
import numpy as np
import tensorflow as tf
import easydict

from fish_db import fish_db
from config import *
from nets import *
from utils.util import sparse_to_dense, bgr_to_rgb


FLAGS = easydict.EasyDict()
FLAGS.data_path = '../data'
FLAGS.image_set = 'train_fold0'
FLAGS.train_dir = '../train_logs'
FLAGS.max_steps = 100000
FLAGS.net = 'resnet50'
FLAGS.pretrained_model_path = '../data/models/ResNet/ResNet-50-weights.pkl'
FLAGS.summary_step = 20
FLAGS.summary_images_step = 2 # times summary_step
FLAGS.checkpoint_step = 1000
FLAGS.gpu = '0'
FLAGS.num_thread = 0
FLAGS.restore = True


def _draw_circle(im, circle_list, label_list, color=(0, 255, 0), cdict=None, scale=1.):
  for circle, label in zip(circle_list, label_list):
    cx, cy, r = [int(b * scale) for b in circle]

    l = label.split(':')[0] # text before "CLASS: (PROB)"
    if cdict and l in cdict:
      c = cdict[l]
    else:
      c = color

    cv2.circle(im, (cx, cy), r, c, 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(im, label, (cx + r, cy), font, 0.3, c, 1)


def _viz_prediction_result(model, images, shapes, labels, batch_det_shape,
                           batch_det_class, batch_det_prob):
  mc = model.mc

  for i in range(len(images)):
    ih, iw = [int(d) for d in model.image_to_show.shape[1:3]]
    im = cv2.resize(images[i], (iw, ih))

    scale = ih / images[i].shape[0]

    # draw ground truth
    _draw_circle(
        im, shapes[i],
        [mc.CLASS_NAMES[idx] for idx in labels[i]],
        (0, 255, 0), scale=scale)

    # draw prediction
    det_shape, det_prob, det_class = model.filter_prediction(
        batch_det_shape[i], batch_det_prob[i], batch_det_class[i])

    keep_idx    = [idx for idx in range(len(det_prob)) \
                      if det_prob[idx] > mc.PLOT_PROB_THRESH]
    det_shape    = [det_shape[idx] for idx in keep_idx]
    det_prob    = [det_prob[idx] for idx in keep_idx]
    det_class   = [det_class[idx] for idx in keep_idx]

    _draw_circle(
        im, det_shape,
        [mc.CLASS_NAMES[idx]+': (%.2f)'% prob \
            for idx, prob in zip(det_class, det_prob)],
        (0, 0, 255), scale=scale)

    images[i] = im


def train():
  # seed = 1991
  # np.random.seed(seed)
  # tf.set_random_seed(seed)

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  with tf.Graph().as_default():
    assert FLAGS.net == 'vgg16' or FLAGS.net == 'resnet50' \
        or FLAGS.net == 'squeezeDet' or FLAGS.net == 'squeezeDet+', \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)
    mc, model = None, None
    if FLAGS.net == 'vgg16':
      mc = fish_vgg16_config()
      mc.IS_TRAINING = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = VGG16ConvDet(mc)
    elif FLAGS.net == 'resnet50':
      mc = fish_res50_config()
      mc.IS_TRAINING = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = ResNet50ConvDet(mc)
    elif FLAGS.net == 'squeezeDet':
      mc = fish_squeezeDet_config()
      mc.IS_TRAINING = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = SqueezeDet(mc)
    elif FLAGS.net == 'squeezeDet+':
      mc = fish_squeezeDetPlus_config()
      mc.IS_TRAINING = True
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = SqueezeDetPlus(mc)

    mc.NUM_THREAD = FLAGS.num_thread

    imdb = fish_db(FLAGS.image_set, FLAGS.data_path, mc)

    def _load_data(load_to_placeholder=True):
      # read batch input
      image_per_batch, label_per_batch, box_delta_per_batch, aidx_per_batch, \
          bbox_per_batch = imdb.read_batch()

      label_indices, bbox_indices, box_delta_values, mask_indices, box_values, \
          = [], [], [], [], []
      for i in range(len(label_per_batch)): # batch_size
        for j in range(len(label_per_batch[i])): # number of annotations
          if label_per_batch[i][j] == 'species_none':
            continue
          label_indices.append(
              [i, aidx_per_batch[i][j], label_per_batch[i][j]])
          mask_indices.append([i, aidx_per_batch[i][j]])
          bbox_indices.extend(
              [[i, aidx_per_batch[i][j], k] for k in range(mc.SHAPE_DIM)])
          box_delta_values.extend(box_delta_per_batch[i][j])
          box_values.extend(bbox_per_batch[i][j])

      if load_to_placeholder:
        image_input = model.ph_image_input
        input_mask = model.ph_input_mask
        box_delta_input = model.ph_box_delta_input
        box_input = model.ph_box_input
        labels = model.ph_labels
      else:
        image_input = model.image_input
        input_mask = model.input_mask
        box_delta_input = model.box_delta_input
        box_input = model.box_input
        labels = model.labels

      feed_dict = {
        image_input: image_per_batch,
        input_mask: np.reshape(
          sparse_to_dense(
            mask_indices,
            [mc.BATCH_SIZE, mc.ANCHORS],
            [1.0] * len(mask_indices)),
          [mc.BATCH_SIZE, mc.ANCHORS, 1]),
        box_delta_input: sparse_to_dense(
          bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, mc.SHAPE_DIM],
          box_delta_values),
        box_input: sparse_to_dense(
          bbox_indices, [mc.BATCH_SIZE, mc.ANCHORS, mc.SHAPE_DIM],
          box_values),
        labels: sparse_to_dense(
          label_indices,
          [mc.BATCH_SIZE, mc.ANCHORS, mc.CLASSES],
          [1.0] * len(label_indices)),
      }

      return feed_dict, image_per_batch, label_per_batch, bbox_per_batch

    def _enqueue(sess, coord):
      try:
        while not coord.should_stop():
          feed_dict, _, _, _ = _load_data()
          sess.run(model.enqueue_op, feed_dict=feed_dict)
          if mc.DEBUG_MODE:
            print ("added to the queue")
        if mc.DEBUG_MODE:
          print ("Finished enqueue")
      except Exception, e:
        coord.request_stop(e)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    saver = tf.train.Saver(tf.global_variables())
    summary_op = tf.summary.merge_all()

    step_start = 0

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and ckpt.model_checkpoint_path and FLAGS.restore:
      saver.restore(sess, ckpt.model_checkpoint_path)
      ckpt_step = int(os.path.basename(ckpt.model_checkpoint_path).split('ckpt-')[-1])
      step_start = ckpt_step + 1

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    init = tf.global_variables_initializer()
    sess.run(init)

    coord = tf.train.Coordinator()

    if mc.NUM_THREAD > 0:
      enq_threads = []
      for _ in range(mc.NUM_THREAD):
        enq_thread = threading.Thread(target=_enqueue, args=[sess, coord])
        # enq_thread.isDaemon()
        enq_thread.start()
        enq_threads.append(enq_thread)

    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    run_options = tf.RunOptions(timeout_in_ms=60000)

    sess.run(model.global_step.assign(step_start))

    for step in xrange(step_start, FLAGS.max_steps):
      if coord.should_stop():
        sess.run(model.FIFOQueue.close(cancel_pending_enqueues=True))
        coord.request_stop()
        coord.join(threads)
        break

      start_time = time.time()

      if step % FLAGS.summary_step == 0:
        feed_dict, image_per_batch, label_per_batch, bbox_per_batch = \
            _load_data(load_to_placeholder=False)
        op_list = [
            model.train_op, model.loss, summary_op, model.det_boxes,
            model.det_probs, model.det_class, model.conf_loss,
            model.bbox_loss, model.class_loss
        ]
        _, loss_value, summary_str, det_boxes, det_probs, det_class, \
            conf_loss, bbox_loss, class_loss = sess.run(
                op_list, feed_dict=feed_dict)

        summary_writer.add_summary(summary_str, step)

        if step % FLAGS.summary_step * FLAGS.summary_images_step == 0:
          _viz_prediction_result(model, image_per_batch, bbox_per_batch, label_per_batch,
                         det_boxes, det_class, det_probs)
          image_per_batch = bgr_to_rgb(image_per_batch)
          viz_summary = sess.run(
              model.viz_op, feed_dict={model.image_to_show: image_per_batch})

          summary_writer.add_summary(viz_summary, step)

        summary_writer.flush()
      else:
        if mc.NUM_THREAD > 0:
          _, loss_value, conf_loss, bbox_loss, class_loss = sess.run(
              [model.train_op, model.loss, model.conf_loss, model.bbox_loss,
               model.class_loss], options=run_options)
        else:
          feed_dict, _, _, _ = _load_data(load_to_placeholder=False)
          _, loss_value, conf_loss, bbox_loss, class_loss = sess.run(
              [model.train_op, model.loss, model.conf_loss, model.bbox_loss,
               model.class_loss], feed_dict=feed_dict)

          # grads_vars = model.opt.compute_gradients(model.loss, tf.trainable_variables())
          # for i, (grad, var) in enumerate(grads_vars):
          #   print(var, ':', sess.run([tf.reduce_max(tf.abs(grad))], feed_dict=feed_dict))

      # print(sess.run([tf.reduce_max(tf.abs(model.res4f))], feed_dict=feed_dict))
      # print(sess.run([tf.reduce_max(tf.abs(model.pred_box_delta))], feed_dict=feed_dict))
      # sys.stdout.flush()

      #print(loss_value)
      # if np.isnan(loss_value):
      #   # checkpoint_path = os.path.join(FLAGS.train_dir, 'model_diverged.ckpt')
      #   # saver.save(sess, checkpoint_path, global_step=step)
      #   print(sess.run([
      #     tf.reduce_sum(tf.square(model.ious - model.pred_conf)),
      #     tf.reduce_max(model.ious),
      #     tf.reduce_sum(tf.to_int32(tf.is_nan(model.ious))),
      #     tf.reduce_max(tf.abs(model.pred_conf)),
      #     tf.reduce_sum(tf.to_int32(tf.is_nan(model.pred_conf)))
      #   ], feed_dict=feed_dict))
      #   sys.stdout.flush()

      duration = time.time() - start_time

      assert not np.isnan(loss_value), \
          'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
          'class_loss: {}'.format(loss_value, conf_loss, bbox_loss, class_loss)

      if step % 10 == 0:
        num_images_per_step = mc.BATCH_SIZE
        images_per_sec = num_images_per_step / duration
        sec_per_batch = float(duration)
        format_str = ('%s: step %d, conf loss = %.6f, bbox loss = %.6f, class loss = %.6f, total loss = %.6f '
                      '(%.1f images/sec; %.3f sec/batch)')
        print (format_str % (datetime.now(), step, conf_loss, bbox_loss, class_loss, loss_value,
                             images_per_sec, sec_per_batch))
        sys.stdout.flush()

      # Save the model checkpoint periodically.
      if step % FLAGS.checkpoint_step == 0 or (step + 1) == FLAGS.max_steps:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)


if __name__ == '__main__':
  train()