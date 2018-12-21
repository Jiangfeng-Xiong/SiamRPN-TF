from numpy import loadtxt
import os
import tensorflow as tf
import glob

import os,sys
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))


from model.SiamRPN import SiamRPN
from utils.misc_utils import load_cfgs
from inference.Tracker import Tracker
from utils.misc_utils import auto_select_gpu


if __name__ == "__main__":
  os.environ['CUDA_VISIBLE_DEVICES']=auto_select_gpu()
  
  MODEL="SiameseRPN"
  CHECKPOINT = "Logs/%s/track_model_checkpoints/%s"%(MODEL,MODEL)

  gpu_options = tf.GPUOptions(allow_growth=True)
  sess_config = tf.ConfigProto(gpu_options=gpu_options)
  with tf.Session(config=sess_config) as sess:

    #1. init Tracker Model
    model_config,_,track_config = load_cfgs(CHECKPOINT)
    model = SiamRPN(model_config=model_config,mode='inference')
    model.build()

    global_variables_init_op = tf.global_variables_initializer()
    sess.run(global_variables_init_op)
    model.restore_weights_from_checkpoint(sess)
    tracker = Tracker(sess, model, track_config)

    #2. Setup TestDataSet
    test_data_root = "/home/lab/Dataset/TB50" #test
    tracking_dirs = os.listdir(test_data_root)
    tracking_dirs = [os.path.join(test_data_root,d) for d in tracking_dirs]

    #3. Tracking Process
    for tracking_dir in tracking_dirs:
      gt_file = os.path.join(tracking_dir, "groundtruth_rect.txt")
      if not os.path.exists(gt_file):
        continue
      try:
        first_bbox = loadtxt(gt_file, delimiter=',')[0]
      except:
        try:
          first_bbox = loadtxt(gt_file, delimiter='\t')[0]
        except:
          first_bbox = loadtxt(gt_file, delimiter=' ')[0]

      frames = glob.glob(tracking_dir+"/img/*.jpg")
      frames.sort()
      print("Tracking Video Path: %s"%tracking_dir)
      tracker.track(first_bbox,frames)
