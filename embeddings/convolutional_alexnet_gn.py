import logging
import tensorflow as tf
from utils.misc_utils import get

slim = tf.contrib.slim

from tensorflow.contrib.layers import group_norm

def convolutional_alexnet_gn_arg_scope(embed_config,
                                    trainable=True,
                                    is_training=True):
  is_model_training = trainable and is_training
  if get(embed_config, 'use_gn', True):
    norm_params = {
      "trainable": trainable,
    }
    normalizer_fn = group_norm
  else:
    norm_params = {}
    normalizer_fn = None

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
      activation_fn=tf.nn.relu,
      normalizer_fn=normalizer_fn,
      normalizer_params=norm_params):
    with slim.arg_scope([group_norm], **norm_params):
      with slim.arg_scope([group_norm]) as arg_sc:
        return arg_sc

def convolutional_alexnet(inputs, reuse=None, scope='convolutional_alexnet'):
  """Defines the feature extractor of SiamFC.

  Args:
    inputs: a Tensor of shape [batch, h, w, c].
    reuse: if the weights in the embedding function are reused.
    scope: the variable scope of the computational graph.

  Returns:
    net: the computed features of the inputs.
    end_points: the intermediate outputs of the embedding function.
  """
  with tf.variable_scope(scope, 'convolutional_alexnet', [inputs], reuse=reuse) as sc:
    end_points_collection = sc.name + '_end_points'
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = inputs #255/127
      net = slim.conv2d(net, 96, [11, 11], 2, scope='conv1')#123/59
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool1') #61/29
      with tf.variable_scope('conv2'):
        b1, b2 = tf.split(net, 2, 3)
        b1 = slim.conv2d(b1, 128, [5, 5], scope='b1') #57/25
        # The original implementation has bias terms for all convolution, but
        # it actually isn't necessary if the convolution layer is followed by a batch
        # normalization layer since batch norm will subtract the mean.
        b2 = slim.conv2d(b2, 128, [5, 5], scope='b2') #57/25
        net = tf.concat([b1, b2], 3)
      net = slim.max_pool2d(net, [3, 3], 2, scope='pool2') #28/12
      net = slim.conv2d(net, 384, [3, 3], 1, scope='conv3')#26/10
      with tf.variable_scope('conv4'):
        b1, b2 = tf.split(net, 2, 3)
        b1 = slim.conv2d(b1, 192, [3, 3], 1, scope='b1')#24/8
        b2 = slim.conv2d(b2, 192, [3, 3], 1, scope='b2')#24/8
        net = tf.concat([b1, b2], 3)
      # Conv 5 with only convolution, has bias
      with tf.variable_scope('conv5'):
        #with slim.arg_scope([slim.conv2d],
        #                    activation_fn=None, normalizer_fn=None):
        b1, b2 = tf.split(net, 2, 3)
        b1 = slim.conv2d(b1, 128, [3, 3], 1, scope='b1')#22/6
        b2 = slim.conv2d(b2, 128, [3, 3], 1, scope='b2')#22/6
        net = tf.concat([b1, b2], 3)
      # Convert end_points_collection into a dictionary of end_points.
      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      return net, end_points


convolutional_alexnet.stride = 8
