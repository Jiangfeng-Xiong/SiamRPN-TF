import logging
import sys
import time
import os

CODE_ROOT = '/home/lab-xiong.jiangfeng/Projects/SiameseRPN'
sys.path.insert(0, CODE_ROOT)

import tensorflow as tf
from model.SiamRPN import SiamRPN
from utils.misc_utils import load_cfgs
from inference.Tracker import Tracker
from utils.infer_utils import Rectangle

from utils.misc_utils import auto_select_gpu

os.environ['CUDA_VISIBLE_DEVICES']=auto_select_gpu()

tracker_name="SiamRPN_bn_bz64_reg10_scratch"

def run_SiamRPN(seq, rp, bSaveImage):
  CHECKPOINT = '/home/lab-xiong.jiangfeng/Projects/SiameseRPN/Logs/%s/track_model_checkpoints/%s'%(tracker_name, tracker_name)
  logging.info('Evaluating {}...'.format(CHECKPOINT))
  # Read configurations from json
  model_config, _, track_config = load_cfgs(CHECKPOINT)
  track_config['log_level'] = 0  # Skip verbose logging for speed

  g = tf.Graph()
  with g.as_default():
    model = SiamRPN(model_config=model_config,mode='inference')
    model.build(reuse=tf.AUTO_REUSE)
    global_variables_init_op = tf.global_variables_initializer()

  gpu_options = tf.GPUOptions(allow_growth=True)
  sess_config = tf.ConfigProto(gpu_options=gpu_options)

  with tf.Session(graph=g, config=sess_config) as sess:
    sess.run(global_variables_init_op)
    model.restore_weights_from_checkpoint(sess)
    tracker = Tracker(sess, model, track_config)

    tic = time.clock()
    frames = seq.s_frames
    init_rect = seq.init_rect
    x, y, width, height = init_rect  # OTB format
    init_bb = Rectangle(x - 1, y - 1, width, height)

    trajectory_py = tracker.track(init_bb, frames, bSaveImage, rp)

    trajectory = [Rectangle(val.x + 1, val.y + 1, val.width, val.height) for val in
                  trajectory_py]  # x, y add one to match OTB format
    duration = time.clock() - tic

    result = dict()
    result['res'] = trajectory
    result['type'] = 'rect'
    result['fps'] = round(seq.len / duration, 3)
  return result
