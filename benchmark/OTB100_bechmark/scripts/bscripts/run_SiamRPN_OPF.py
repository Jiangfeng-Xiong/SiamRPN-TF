import logging
import sys
import time
import os
import numpy as np

CODE_ROOT = '/home/lab-xiong.jiangfeng/Projects/SiameseRPN'
sys.path.insert(0, CODE_ROOT)

import tensorflow as tf
from utils.misc_utils import load_cfgs
from inference.OnlineTracker import OnlineTracker,OnlineNet
from utils.infer_utils import Rectangle

from configs import get_model

from utils.misc_utils import auto_select_gpu
online_config={'select_channel_num':96,
                'template_size': 64,
                'conv_dims': [96],
                'conv1_ksizes': [3,7,11],
                'conv_ksizes': [],
                'use_part_filter':1,
                'finetune_part_filter': 1,
                'online_lr': 0.01,
                'epsilon': 0.1,
                'OnlineRankWeight':0.3,
                'dropout_keep_rate': 0.9,
                'bn_decay':0.8,
                'weight_decay': 1e-4,
                'conf_th':0.3,
                'debug': 0}
online_config['output_size'] = online_config['template_size'] - max(online_config['conv1_ksizes']) - sum(online_config['conv_ksizes']) + 1 + len(online_config['conv_ksizes'])

def run_SiamRPN_OPF(seq, rp, bSaveImage):
  os.environ['CUDA_VISIBLE_DEVICES']=auto_select_gpu()
  config_name = "SiamRPN_ftall"
  CHECKPOINT = '/home/lab-xiong.jiangfeng/Projects/SiameseRPN/Logs/%s/track_model_checkpoints/%s'%(config_name, config_name)
  logging.info('Evaluating {}...'.format(CHECKPOINT))
  
  # Read configurations from json
  model_config, _, track_config = load_cfgs(CHECKPOINT)
  track_config['log_level'] = 0  # Skip verbose logging for speed

 
  np.random.seed(1234)
  tf.set_random_seed(1234)
  g = tf.Graph()
  
  with g.as_default():
    model = get_model(model_config['Model'])(model_config=model_config,mode='inference')
    model.build(reuse=tf.AUTO_REUSE)
    model.online_net = OnlineNet(online_config,is_training=True, reuse=False)
    model.online_valnet = OnlineNet(online_config,is_training=False, reuse=True)
    global_variables_init_op = tf.global_variables_initializer()
    
  gpu_options = tf.GPUOptions(allow_growth=True)
  sess_config = tf.ConfigProto(gpu_options=gpu_options)
  sess_config.gpu_options.per_process_gpu_memory_fraction = 0.2

  with tf.Session(graph=g, config=sess_config) as sess:
    sess.run(global_variables_init_op)
    model.restore_weights_from_checkpoint(sess, 605000)
    tracker = OnlineTracker(sess, model, track_config, online_config, show_video=0)

    tic = time.clock()
    frames = seq.s_frames
    init_rect = seq.init_rect
    x, y, width, height = init_rect  # OTB format
    init_bb = Rectangle(x - 1, y - 1, width, height)
    trajectory_py = tracker.track(init_bb, frames, bSaveImage, rp)
    #print(trajectory_py)
    trajectory = [Rectangle(val.x + 1, val.y + 1, val.width, val.height) for val in
                  trajectory_py]  # x, y add one to match OTB format
    duration = time.clock() - tic

    result = dict()
    result['res'] = trajectory
    result['type'] = 'rect'
    result['fps'] = round(seq.len / duration, 3)
  return result
