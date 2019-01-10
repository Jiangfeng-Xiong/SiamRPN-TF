from numpy import loadtxt
import os
import tensorflow as tf
import glob

import os,sys
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

from configs import get_model
from utils.misc_utils import load_cfgs
from inference.Tracker import Tracker
from utils.misc_utils import auto_select_gpu

def get_topleft_bbox(region):
    import numpy as np
    region = np.array(region)
    cx = np.mean(region[0::2])
    cy = np.mean(region[1::2])
    x1 = min(region[0::2])
    x2 = max(region[0::2])
    y1 = min(region[1::2])
    y2 = max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1
    return [cx-(w-1)//2, cy-(h-1)//2, w, h]

if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES']=auto_select_gpu()
  
  MODEL="SiamRPN_Base"
  CHECKPOINT = "Logs/%s/track_model_checkpoints/%s"%(MODEL,MODEL)

  gpu_options = tf.GPUOptions(allow_growth=True)
  sess_config = tf.ConfigProto(gpu_options=gpu_options)
  with tf.Session(config=sess_config) as sess:

    #1. init Tracker Model
    model_config,_,track_config = load_cfgs(CHECKPOINT)
    model = get_model(model_config['Model'])(model_config=model_config,mode='inference')
    model.build(reuse=tf.AUTO_REUSE)

    global_variables_init_op = tf.global_variables_initializer()
    sess.run(global_variables_init_op)
    model.restore_weights_from_checkpoint(sess)
    tracker = Tracker(sess, model, track_config)

   #2. load tracking video
    tracking_dir = "dataset/demo/bag"
    gt_file = os.path.join(tracking_dir, "groundtruth.txt")
    try:
      first_bbox = loadtxt(gt_file, delimiter=',')[0]
    except:
      try:
        first_bbox = loadtxt(gt_file, delimiter='\t')[0]
      except:
        first_bbox = loadtxt(gt_file, delimiter=' ')[0]

    first_bbox = get_axis_aligned_bbox(first_bbox)
    frames = glob.glob(tracking_dir+"/img/*.jpg")
    frames.sort()
    print("Tracking Video Path: %s"%tracking_dir)

    #3. Main Tracking Process
    tracker.track(first_bbox,frames)
