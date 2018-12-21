import tensorflow as tf

def HannWindows(feats):
  feats_shape = tf.shape(feats)
  windows = tf.matmul(tf.expand_dims(tf.contrib.signal.hann_window(feats_shape[1]),1),
                      tf.expand_dims(tf.contrib.signal.hann_window(feats_shape[2]),0))
  windows = windows/tf.reduce_max(windows)
  windows = tf.stop_gradient(tf.tile(tf.reshape(windows, [1,feats_shape[1],feats_shape[2],1]),[feats_shape[0],1,1,feats_shape[3]]))
  return tf.multiply(feats, tf.to_float(windows))

def GaussianWindows(feats, bboxes):
  def apply_gaussian(feature_map, bbox):
    w = bbox[2] - bbox[0] + 1
    h = bbox[3] - bbox[1] + 1
    feats_shape = tf.shape(feature_map)
    ys = tf.to_float(tf.range(feats_shape[0],dtype=tf.int32))
    xs = tf.to_float(tf.range(feats_shape[1],dtype=tf.int32))
    center_x = (bbox[0] + bbox[2])/2.0
    center_y = (bbox[1] + bbox[3])/2.0

    windows = tf.matmul(tf.expand_dims(tf.exp(-(ys-center_y)**2/(2*h*5)),1),
                      tf.expand_dims(tf.exp(-(xs-center_x)**2/(2*w*5)),0))

    windows = windows + tf.reduce_max(windows)/100.0
    windows = windows/tf.reduce_max(windows)
    windows = tf.tile(tf.reshape(windows, [feats_shape[0],feats_shape[1],1]),[1,1,feats_shape[2]])
    return tf.multiply(feature_map, tf.to_float(windows))

  return tf.map_fn(lambda x: apply_gaussian(x[0], x[1]), [feats, bboxes],dtype=tf.float32)

def BinWindows(feats, bboxes):
  def apply_bin(feature_map, bbox):
    w = bbox[2] - bbox[0] + 1
    h = bbox[3] - bbox[1] + 1
    feats_shape = tf.shape(feature_map)
    ys = tf.to_float(tf.range(feats_shape[0],dtype=tf.int32))
    xs = tf.to_float(tf.range(feats_shape[1],dtype=tf.int32))
    center_x = (bbox[0] + bbox[2])/2.0
    center_y = (bbox[1] + bbox[3])/2.0

    x = tf.to_float(tf.less_equal(tf.abs(xs-center_x), w/2.0))
    y = tf.to_float(tf.less_equal(tf.abs(ys-center_y), h/2.0))

    windows = tf.matmul(tf.expand_dims(y,1), tf.expand_dims(x,0))

    windows = windows + 0.3
    windows = windows/tf.reduce_max(windows)
    windows = tf.tile(tf.reshape(windows, [feats_shape[0],feats_shape[1],1]),[1,1,feats_shape[2]])
    return tf.multiply(feature_map, tf.to_float(windows))

  return tf.map_fn(lambda x: apply_bin(x[0], x[1]), [feats, bboxes],dtype=tf.float32)