import logging
import os
import os.path as osp
import sys

import numpy as np
import tensorflow as tf
import pickle

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))

from configs import get_model,get_config
from utils.misc_utils import auto_select_gpu, save_cfgs
from utils.train_utils import print_trainable

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()
tf.logging.set_verbosity(tf.logging.DEBUG)

from sacred import Experiment

config_name = "SiamRPN_Base"
config = get_config(config_name)

ex = Experiment(config.RUN_NAME)

def load_pickle_model(pickle_path='embeddings/pytorch_weights/siamrpn_model.pkl', embed_scope='featureExtract_alexnet/'):
  f = open(pickle_path, 'rb')
  params = pickle.load(f,encoding="bytes")
  
  assign_ops = []
  def _assign(ref_name, params, scope=embed_scope):
    print("assigning %s"%(scope + str(ref_name,encoding='utf-8')))
    var_in_model = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope + str(ref_name,encoding='utf-8'))[0]
    var_in_mat = params[ref_name]
    op = tf.assign(var_in_model, var_in_mat)
    assign_ops.append(op)

  for l in range(1, 6):
    _assign(b'conv%d/weights' % l, params)
    _assign(b'conv%d/biases' % l, params)
    _assign(b'bn%d/beta' % l, params)
    _assign(b'bn%d/gamma' % l, params)
    _assign(b'bn%d/moving_mean' % l, params)
    _assign(b'bn%d/moving_variance' % l, params)
 
  _assign(b'conv_r1/weights',params, 'regssion/')
  _assign(b'conv_r1/biases',params, 'regssion/')
  _assign(b'conv_r2/weights',params, 'regssion/')
  _assign(b'conv_r2/biases',params, 'regssion/')
  _assign(b'regress_adjust/weights',params, 'regssion/')
  _assign(b'regress_adjust/biases',params, 'regssion/')
  
  _assign(b'conv_cls1/weights',params, 'cls/')
  _assign(b'conv_cls1/biases',params, 'cls/')
  _assign(b'conv_cls2/weights',params, 'cls/')
  _assign(b'conv_cls2/biases',params, 'cls/')

  initialize = tf.group(*assign_ops)
  
  return initialize

@ex.config
def configurations():
  # Add configurations for current script, for more details please see the documentation of `sacred`.
  model_config = config.MODEL_CONFIG
  train_config = config.TRAIN_CONFIG
  track_config = config.TRACK_CONFIG

@ex.automain
def main(model_config, train_config, track_config):
  # Create training directory
  train_dir = train_config['train_dir']
  if not tf.gfile.IsDirectory(train_dir):
    tf.logging.info('Creating training directory: %s', train_dir)
    tf.gfile.MakeDirs(train_dir)

  # Build the Tensorflow graph
  g = tf.Graph()
  with g.as_default():
    # Set fixed seed
    np.random.seed(train_config['seed'])
    tf.set_random_seed(train_config['seed'])

    # Build the model
    model = get_model(model_config['Model'])(model_config, train_config, mode='inference')
    model.build()

    # Save configurations for future reference
    save_cfgs(train_dir, model_config, train_config, track_config)

    saver = tf.train.Saver(tf.global_variables(),
                           max_to_keep=train_config['max_checkpoints_to_keep'])

    # Dynamically allocate GPU memory
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options)

    sess = tf.Session(config=sess_config)
    model_path = tf.train.latest_checkpoint(train_config['train_dir'])

    if not model_path:
      # Initialize all variables
      #sess.run(tf.global_variables_initializer())
      #sess.run(tf.local_variables_initializer())
      start_step = 0

      # Load pretrained embedding model if needed
      sess.run(load_pickle_model())
      print_trainable(sess)
    else:
      logging.info('Restore from last checkpoint: {}'.format(model_path))
      sess.run(tf.local_variables_initializer())
      saver.restore(sess, model_path)
      start_step = tf.train.global_step(sess, model.global_step.name) + 1

    checkpoint_path = osp.join(train_config['train_dir'], 'model.ckpt')
    saver.save(sess, checkpoint_path, global_step=start_step)
