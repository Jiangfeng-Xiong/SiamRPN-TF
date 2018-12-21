import numpy as np
import tensorflow as tf
import cv2
from scipy import io as sio
import re
def show_pred_bbox(imgs, Topk_bboxes, Topk_scores,gts=None):
  """
  imgs: N*H*W*3
  Topk_scores: N*Topk
  Topk_bboxes: N*Topk*4
  """
  nums = imgs.shape[0]
  for i in range(nums):
    bbox = Topk_bboxes[i]
    pred_prob = Topk_scores[i]
    min_val = np.min(imgs[i])
    if min_val<0:
      imgs[i] = (imgs[i] - min_val)/(np.max(imgs[i]) - min_val)
      imgs[i] = imgs[i] * 255

    cv2.rectangle(imgs[i],(gts[i][0],gts[i][1]), (gts[i][2],gts[i][3]),(255,255,255),2)
    for index,prob in enumerate(pred_prob):
      try:
        cv2.rectangle(imgs[i],(bbox[index][0],bbox[index][1]), (bbox[index][2],bbox[index][3]),(0,255,0),2)
        cv2.putText(imgs[i], "p: %.2f"%(prob), (bbox[index][0],bbox[index][1]), 0, 1, (0,255,0),2)
      except:
        print("inf or nan in pred boxes")
  return imgs
def get_params_from_mat(matpath):
  """Get parameter from .mat file into parms(dict)"""

  def squeeze(vars_):
    # Matlab save some params with shape (*, 1)
    # However, we don't need the trailing dimension in TensorFlow.
    if isinstance(vars_, (list, tuple)):
      return [np.squeeze(v, 1) for v in vars_]
    else:
      return np.squeeze(vars_, 1)

  netparams = sio.loadmat(matpath)["net"]["params"][0][0]
  params = dict()

  for i in range(netparams.size):
    param = netparams[0][i]
    name = param["name"][0]
    value = param["value"]
    value_size = param["value"].shape[0]

    match = re.match(r"([a-z]+)([0-9]+)([a-z]+)", name, re.I)
    if match:
      items = match.groups()
    elif name == 'adjust_f':
      params['detection/weights'] = squeeze(value)
      continue
    elif name == 'adjust_b':
      params['detection/biases'] = squeeze(value)
      continue
    else:
      raise Exception('unrecognized layer params')

    op, layer, types = items
    layer = int(layer)
    if layer in [1, 3]:
      if op == 'conv':  # convolution
        if types == 'f':
          params['conv%d/weights' % layer] = value
        elif types == 'b':
          value = squeeze(value)
          params['conv%d/biases' % layer] = value
      elif op == 'bn':  # batch normalization
        if types == 'x':
          m, v = squeeze(np.split(value, 2, 1))
          params['conv%d/BatchNorm/moving_mean' % layer] = m
          params['conv%d/BatchNorm/moving_variance' % layer] = np.square(v)
        elif types == 'm':
          value = squeeze(value)
          params['conv%d/BatchNorm/gamma' % layer] = value
        elif types == 'b':
          value = squeeze(value)
          params['conv%d/BatchNorm/beta' % layer] = value
      else:
        raise Exception
    elif layer in [2, 4]:
      if op == 'conv' and types == 'f':
        b1, b2 = np.split(value, 2, 3)
      else:
        b1, b2 = np.split(value, 2, 0)
      if op == 'conv':
        if types == 'f':
          params['conv%d/b1/weights' % layer] = b1
          params['conv%d/b2/weights' % layer] = b2
        elif types == 'b':
          b1, b2 = squeeze(np.split(value, 2, 0))
          params['conv%d/b1/biases' % layer] = b1
          params['conv%d/b2/biases' % layer] = b2
      elif op == 'bn':
        if types == 'x':
          m1, v1 = squeeze(np.split(b1, 2, 1))
          m2, v2 = squeeze(np.split(b2, 2, 1))
          params['conv%d/b1/BatchNorm/moving_mean' % layer] = m1
          params['conv%d/b2/BatchNorm/moving_mean' % layer] = m2
          params['conv%d/b1/BatchNorm/moving_variance' % layer] = np.square(v1)
          params['conv%d/b2/BatchNorm/moving_variance' % layer] = np.square(v2)
        elif types == 'm':
          params['conv%d/b1/BatchNorm/gamma' % layer] = squeeze(b1)
          params['conv%d/b2/BatchNorm/gamma' % layer] = squeeze(b2)
        elif types == 'b':
          params['conv%d/b1/BatchNorm/beta' % layer] = squeeze(b1)
          params['conv%d/b2/BatchNorm/beta' % layer] = squeeze(b2)
      else:
        raise Exception

    elif layer in [5]:
      if op == 'conv' and types == 'f':
        b1, b2 = np.split(value, 2, 3)
      else:
        b1, b2 = squeeze(np.split(value, 2, 0))
      assert op == 'conv', 'layer5 contains only convolution'
      if types == 'f':
        params['conv%d/b1/weights' % layer] = b1
        params['conv%d/b2/weights' % layer] = b2
      elif types == 'b':
        params['conv%d/b1/biases' % layer] = b1
        params['conv%d/b2/biases' % layer] = b2

  return params

def load_mat_model(matpath, embed_scope):
  """Restore SiameseFC models from .mat model files"""
  params = get_params_from_mat(matpath)

  assign_ops = []

  def _assign(ref_name, params, scope=embed_scope):
    var_in_model = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope + ref_name)[0]
    var_in_mat = params[ref_name]
    op = tf.assign(var_in_model, var_in_mat)
    assign_ops.append(op)

  for l in range(1, 6):
    if l in [1, 3]:
      _assign('conv%d/weights' % l, params)
      # _assign('conv%d/biases' % l, params)
      _assign('conv%d/BatchNorm/beta' % l, params)
      _assign('conv%d/BatchNorm/gamma' % l, params)
      _assign('conv%d/BatchNorm/moving_mean' % l, params)
      _assign('conv%d/BatchNorm/moving_variance' % l, params)
    elif l in [2, 4]:
      # Branch 1
      _assign('conv%d/b1/weights' % l, params)
      # _assign('conv%d/b1/biases' % l, params)
      _assign('conv%d/b1/BatchNorm/beta' % l, params)
      _assign('conv%d/b1/BatchNorm/gamma' % l, params)
      _assign('conv%d/b1/BatchNorm/moving_mean' % l, params)
      _assign('conv%d/b1/BatchNorm/moving_variance' % l, params)
      # Branch 2
      _assign('conv%d/b2/weights' % l, params)
      # _assign('conv%d/b2/biases' % l, params)
      _assign('conv%d/b2/BatchNorm/beta' % l, params)
      _assign('conv%d/b2/BatchNorm/gamma' % l, params)
      _assign('conv%d/b2/BatchNorm/moving_mean' % l, params)
      _assign('conv%d/b2/BatchNorm/moving_variance' % l, params)
    elif l in [5]:
      # Branch 1
      _assign('conv%d/b1/weights' % l, params)
      #_assign('conv%d/b1/biases' % l, params)
      # Branch 2
      _assign('conv%d/b2/weights' % l, params)
      #_assign('conv%d/b2/biases' % l, params)
    else:
      raise Exception('layer number must below 5')


  initialize = tf.group(*assign_ops)
  return initialize

def print_trainable(sess):
  variables_names = [v.name for v in tf.trainable_variables()]
  values = sess.run(variables_names)
  for k, v in zip(variables_names, values):
    print("Variable: ", k)
    print("Shape: ", v.shape)
