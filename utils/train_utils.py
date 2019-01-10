import numpy as np
import tensorflow as tf
import cv2
from scipy import io as sio
import re

#draw top confident box during training
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

def print_trainable(sess):
  variables_names = [v.name for v in tf.trainable_variables()]
  values = sess.run(variables_names)
  for k, v in zip(variables_names, values):
    print("Variable: ", k)
    print("Shape: ", v.shape)

# for multi-gpu training 
def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1] 
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

#load pretrained model for alexnet only
def load_pickle_model(pickle_path='embeddings/pytorch_weights/siamrpn_model.pkl', scope='featureExtract_alexnet/'):
  f = open(pickle_path, 'rb')
  params = pickle.load(f,encoding="bytes")
  assign_ops = []
  def _assign(ref_name, params, scope=scope):
    print("assigning %s"%(scope + str(ref_name,encoding='utf-8')))
    var_in_model = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope + str(ref_name,encoding='utf-8'))
    if len(var_in_model)>0:
      var_in_mat = params[ref_name]
      op = tf.assign(var_in_model, var_in_mat)
      assign_ops.append(op)
      print("sucess assign %s"%(scope + str(ref_name,encoding='utf-8')))
    else:
      print("key %s is not found in current session"%(scope + str(ref_name,encoding='utf-8')))

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