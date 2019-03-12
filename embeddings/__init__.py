from embeddings.convolutional_alexnet import convolutional_alexnet_arg_scope, convolutional_alexnet
from embeddings.convolutional_alexnet_gn import convolutional_alexnet_gn_arg_scope
from embeddings.featureExtract_alexnet import featureExtract_alexnet,featureExtract_alexnet_arg_scope
from embeddings.alexnet_tweak import alexnet_tweak_arg_scope, alexnet_tweak
from embeddings.featureExtract_alexnet_fixedconv3 import featureExtract_alexnet_fixedconv3
import tensorflow as tf
slim = tf.contrib.slim
import logging


def get_scope_and_backbone(config, is_training):
    embedding_name = config['embedding_name']
    if embedding_name== 'convolutional_alexnet':
        arg_scope = convolutional_alexnet_arg_scope(config, trainable=config['train_embedding'], is_training=is_training)
        backbone_fn = convolutional_alexnet
    elif embedding_name== 'convolutional_alexnet_gn':
        arg_scope = convolutional_alexnet_gn_arg_scope(config, trainable=config['train_embedding'])
        backbone_fn = convolutional_alexnet
    elif embedding_name == 'alexnet_tweak':
        arg_scope = alexnet_tweak_arg_scope(config, trainable=config['train_embedding'], is_training=is_training)
        backbone_fn = alexnet_tweak
    elif embedding_name == 'featureExtract_alexnet':
        arg_scope = featureExtract_alexnet_arg_scope(config, trainable=config['train_embedding'], is_training=is_training)
        backbone_fn = featureExtract_alexnet
    elif embedding_name == 'featureExtract_alexnet_fixedconv3':
        arg_scope = featureExtract_alexnet_arg_scope(config, trainable=config['train_embedding'], is_training=is_training)
        backbone_fn = featureExtract_alexnet_fixedconv3
    else:
        assert("support alexnet only now")
    return arg_scope,backbone_fn

def newAddNet_arg_scope(is_training=True):
  batch_norm_params = {
      "scale": True,
      # Decay for the moving averages.
      "decay": 0.95,
      # Epsilon to prevent 0s in variance.
      "epsilon": 1e-6,
      "trainable": is_training,
      "is_training": is_training,
      # Collection containing the moving mean and moving variance.
      "variables_collections": {
        "beta": None,
        "gamma": None,
        "moving_mean": ["moving_vars"],
        "moving_variance": ["moving_vars"],
      },
      'updates_collections': None,  # Ensure that updates are done within a frame
    }
  weight_decay = 1e-4
  if is_training:
    weights_regularizer = slim.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None
  init_method = 'kaiming_normal'
  
  if is_training:
    logging.info('embedding init method -- {}'.format(init_method))
  if init_method == 'kaiming_normal':
    initializer = slim.variance_scaling_initializer(factor=2.0, mode='FAN_OUT', uniform=False)
  else:
    initializer = slim.xavier_initializer()

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=weights_regularizer,
      weights_initializer=initializer,
      padding='VALID',
      trainable=is_training,
      activation_fn=tf.nn.relu,
      normalizer_fn=slim.batch_norm,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.batch_norm], is_training=is_training) as arg_sc:
        return arg_sc