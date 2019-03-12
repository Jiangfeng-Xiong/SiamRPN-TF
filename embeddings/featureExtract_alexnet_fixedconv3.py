import logging
import tensorflow as tf
from utils.misc_utils import get

slim = tf.contrib.slim

def featureExtract_alexnet_arg_scope(embed_config,
                                    trainable=True,
                                    is_training=False):
  """Defines the default arg scope.

  Args:
    embed_config: A dictionary which contains configurations for the embedding function.
    trainable: If the weights in the embedding function is trainable.
    is_training: If the embedding function is built for training.

  Returns:
    An `arg_scope` to use for the convolutional_alexnet models.
  """
  # Only consider the model to be in training mode if it's trainable.
  # This is vital for batch_norm since moving_mean and moving_variance
  # will get updated even if not trainable.
  is_model_training = trainable and is_training

  if get(embed_config, 'use_bn', True):
    batch_norm_scale = get(embed_config, 'bn_scale', True)
    batch_norm_decay = 1 - get(embed_config, 'bn_momentum', 3e-4)
    batch_norm_epsilon = get(embed_config, 'bn_epsilon', 1e-6)
    batch_norm_params = {
      "scale": batch_norm_scale,
      # Decay for the moving averages.
      "decay": batch_norm_decay,
      # Epsilon to prevent 0s in variance.
      "epsilon": batch_norm_epsilon,
      "trainable": trainable,
      "is_training": is_model_training,
      # Collection containing the moving mean and moving variance.
      "variables_collections": {
        "beta": None,
        "gamma": None,
        "moving_mean": ["moving_vars"],
        "moving_variance": ["moving_vars"],
      },
      'updates_collections': None,  # Ensure that updates are done within a frame
    }
  else:
    batch_norm_params = {}

  weight_decay = get(embed_config, 'weight_decay', 1e-4)
  if trainable:
    weights_regularizer = slim.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  init_method = get(embed_config, 'init_method', 'kaiming_normal')
  if is_model_training:
    logging.info('embedding init method -- {}'.format(init_method))
  if init_method == 'kaiming_normal':
    # The same setting as siamese-fc
    initializer = slim.variance_scaling_initializer(factor=2.0, mode='FAN_OUT', uniform=False)
  else:
    initializer = slim.xavier_initializer()

  with slim.arg_scope(
      [slim.conv2d],
      weights_regularizer=weights_regularizer,
      weights_initializer=initializer,
      padding='VALID',
      trainable=trainable,
      activation_fn=None,
      normalizer_fn=None,
      normalizer_params=batch_norm_params):
    with slim.arg_scope([slim.batch_norm], **batch_norm_params):
      with slim.arg_scope([slim.batch_norm], is_training=is_model_training) as arg_sc:
        return arg_sc

def get_fixed_bn_config():
    batch_norm_params = {
      "scale": True,
      # Decay for the moving averages.
      "decay": 0.95,
      # Epsilon to prevent 0s in variance.
      "epsilon": 1e-6,
      "trainable": False,
      "is_training": False,
      # Collection containing the moving mean and moving variance.
      "variables_collections": {
        "beta": None,
        "gamma": None,
        "moving_mean": ["moving_vars"],
        "moving_variance": ["moving_vars"],
      },
      'updates_collections': None,  # Ensure that updates are done within a frame
    }
    return batch_norm_params

def featureExtract_alexnet_fixedconv3(inputs, reuse=None, scope='featureExtract_alexnet'):
  with tf.variable_scope(scope, 'featureExtract_alexnet', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.max_pool2d,slim.batch_norm],
                        outputs_collections=end_points_collection):
      with slim.arg_scope([slim.batch_norm], **get_fixed_bn_config()):
          net = inputs #255/127
          net = slim.conv2d(net, 96, [11, 11], 2, scope='conv1',trainable=False)#123/59
          net = slim.batch_norm(net, scope='bn1')
          net = slim.max_pool2d(net, [3, 3], 2, scope='pool1') #61/29
          net = tf.nn.relu(net)
          net = slim.conv2d(net, 256, [5, 5], scope='conv2',trainable=False) #57/25
          net = slim.batch_norm(net, scope='bn2')
          net = slim.max_pool2d(net, [3, 3], 2, scope='pool2') #28/12
          net = tf.nn.relu(net)
          net = slim.conv2d(net, 384, [3, 3], 1, scope='conv3',trainable=False)#26/10
          net = slim.batch_norm(net, scope='bn3')
      
      net = tf.nn.relu(net)
      net = slim.conv2d(net, 384, [3, 3], 1, scope='conv4')#24/8
      net = slim.batch_norm(net, scope='bn4')
      
      net = tf.nn.relu(net)
      net = slim.conv2d(net, 256, [3, 3], 1, scope='conv5')#24/8
      net = slim.batch_norm(net, scope='bn5')
      # Convert end_points_collection into a dictionary of end_points.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net, end_points


