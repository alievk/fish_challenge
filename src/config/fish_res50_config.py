"""Model configuration for 'N+1 fish, N+2 fish' dataset"""

import numpy as np

from config import base_model_config

def fish_res50_config():
  """Specify the parameters to tune below."""
  mc                       = base_model_config('fish')

  mc.NET                   = 'resnet50'

  mc.IMAGE_WIDTH           = 1280
  mc.IMAGE_HEIGHT          = 720
  mc.BATCH_SIZE            = 10

  mc.WEIGHT_DECAY          = 0.0001#0.0001
  mc.LEARNING_RATE         = 0.01#0.02
  mc.DECAY_STEPS           = 10000
  mc.MAX_GRAD_NORM         = 1.0
  mc.MOMENTUM              = 0.9
  mc.LR_DECAY_FACTOR       = 0.5

  mc.LOSS_COEF_BBOX        = 0.5#5.
  mc.LOSS_COEF_CONF_POS    = 75.
  mc.LOSS_COEF_CONF_NEG    = 100.0
  mc.LOSS_COEF_CLASS       = 1.0

  mc.PLOT_PROB_THRESH      = 0.4
  mc.NMS_THRESH            = 0.4
  mc.PROB_THRESH           = 0.005
  mc.TOP_N_DETECTION       = 64

  mc.DATA_AUGMENTATION     = True
  mc.DRIFT_X               = 150
  mc.DRIFT_Y               = 100

  mc.SHAPE_DIM             = 3

  diameters = [87, 150, 202, 233, 278]
  radii = 0.5 * np.array(diameters)
  anchors = set_anchors(mc, radii)

  mc.ANCHOR_BOX            = anchors
  mc.ANCHORS               = len(anchors)
  mc.ANCHOR_PER_GRID       = len(radii)

  return mc


def set_anchors(mc, radii):
  from math import ceil
  H, W, B = int(ceil(mc.IMAGE_HEIGHT/16.)), int(ceil(mc.IMAGE_WIDTH/16.)), len(radii)
  anchor_radii = np.reshape(
      [np.array(radii)] * H * W,
      (H, W, B, 1)
  )
  center_x = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, W+1)*float(mc.IMAGE_WIDTH)/(W+1)]*H*B), 
              (B, H, W)
          ),
          (1, 2, 0)
      ),
      (H, W, B, 1)
  )
  center_y = np.reshape(
      np.transpose(
          np.reshape(
              np.array([np.arange(1, H+1)*float(mc.IMAGE_HEIGHT)/(H+1)]*W*B),
              (B, W, H)
          ),
          (2, 1, 0)
      ),
      (H, W, B, 1)
  )
  anchors = np.reshape(
      np.concatenate((center_x, center_y, anchor_radii), axis=3),
      (-1, 3)
  )

  return anchors
