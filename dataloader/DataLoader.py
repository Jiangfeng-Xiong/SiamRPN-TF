import tensorflow as tf
import numpy as np

import os,sys
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

from dataloader.sampler import Sampler,ShuffleSample
from dataloader.data_augmentation import RandomGray,RandomStretch,CenterCrop,RandomCrop,RandomColorAug,RandomFlip,RandomBlur

##########DataLoader
class DataLoader(object):
  def __init__(self, config, is_training):
    self.config = config
    self.is_training = is_training
    self.examplar_size = 127
    self.instance_size = 255
    self.dataset_py = Sampler(config['input_imdb'], config['max_frame_dist'])
    
    shuffle = False if self.config.get('lmdb_path', None) else is_training
    self.sampler = ShuffleSample(self.dataset_py, shuffle=is_training)

    if self.config.get('lmdb_path', None):
      import lmdb
      env = lmdb.open(self.config['lmdb_path'],map_size = 109951162777*6)
      self.txn = env.begin()

  def examplar_transform(self, input_image, gt_examplar_box):
    img = CenterCrop(input_image,self.examplar_size)
    shift_y = (self.instance_size - self.examplar_size)//2
    shift_x = (self.instance_size - self.examplar_size)//2
    x1 = gt_examplar_box[0] - shift_x
    y1 = gt_examplar_box[1] - shift_y
    x2 = gt_examplar_box[2] - shift_x
    y2 = gt_examplar_box[3] - shift_y
    gt_examplar_box = [x1, y1 ,x2 ,y2]

    return img, gt_examplar_box

  def instance_transform(self, img, gt_instance_box):
    if self.is_training:
      #Random flip doesnot affect the location of the target
      if self.config['augmentation_config']['random_flip']: 
        img = RandomFlip(img)
      if self.config['augmentation_config']['random_color']:
        img = RandomColorAug(img)
      if self.config['augmentation_config']['random_blur']:
        img = RandomBlur(img)

      img,scale = RandomStretch(img, max_stretch=0.4)
      img,shift_xy,pad_xy = RandomCrop(img, self.instance_size)

      gt_instance_box = gt_instance_box * scale
      w = gt_instance_box[2] - gt_instance_box[0] + 1.0
      h = gt_instance_box[3] - gt_instance_box[1] + 1.0
      cx = (gt_instance_box[0] + gt_instance_box[2])/2.0
      cy = (gt_instance_box[1] + gt_instance_box[3])/2.0

      cx = cx - tf.to_float(shift_xy[0] - pad_xy[0]) 
      cy = cy - tf.to_float(shift_xy[1] - pad_xy[1])

      gt_instance_box=[cx-w/2.0, cy-h/2.0, cx + w/2.0, cy + h/2.0]

    return img, gt_instance_box

  def build(self):
    self.build_dataset()
    self.build_iterator()
 
  def build_dataset(self):
    def sample_generator():
      for video_id in self.sampler:
        sample = self.dataset_py[video_id]
        yield sample

    def transform_fn(img_paths):
      def get_bytes_from_lmdb(key):
        buffer = self.txn.get(key)
        if isinstance(buffer,type(None)):
          print("%s not found in database, continue"%(key))
          return None
        img_buffer = np.frombuffer(buffer, dtype=np.uint8)
        img_size = int(np.sqrt(len(img_buffer)/3))
        value = np.reshape(img_buffer, [img_size, img_size, 3])
        return value
       
      if self.config.get('lmdb_path', None):
        exemplar_image = tf.py_func(get_bytes_from_lmdb, [img_paths[0]], tf.uint8, name = "exemplar_image")
        instance_image = tf.py_func(get_bytes_from_lmdb, [img_paths[1]], tf.uint8, name = "instance_image")
      else:
        examplar_file = tf.read_file(img_paths[0])
        instance_file = tf.read_file(img_paths[1])
        exemplar_image = tf.image.decode_jpeg(examplar_file, channels=3, dct_method="INTEGER_ACCURATE")
        instance_image = tf.image.decode_jpeg(instance_file, channels=3, dct_method="INTEGER_ACCURATE")

      def get_gt_box(bytes):
        string = str(bytes, encoding="utf-8")
        string = string.split('/')[-1]
        #print(string)
        w = int(string.split('.')[2]) #1.w.100.h.100.jpg
        h = int(string.split('.')[4])
        cx = (self.instance_size-1)/2.0
        cy = (self.instance_size-1)/2.0
        box = np.array([cx - w/2.0, cy - h/2.0, cx + w/2.0, cy + h/2.0],np.float32)
        return box

      gt_instance_box = tf.py_func(get_gt_box, [img_paths[1]], tf.float32,name="gt_instance_box")
      gt_instance_box.set_shape([4])

      gt_examplar_box = tf.py_func(get_gt_box, [img_paths[0]], tf.float32,name="gt_examplar_box")
      gt_examplar_box.set_shape([4])

      video = tf.stack([exemplar_image, instance_image])
      video = RandomGray(video)

      exemplar_image = video[0]
      instance_image = video[1]

      exemplar_image,gt_examplar_box = self.examplar_transform(exemplar_image, gt_examplar_box)
      instance_image,gt_instance_box = self.instance_transform(instance_image, gt_instance_box)

      return exemplar_image, instance_image, gt_examplar_box, gt_instance_box

    dataset = tf.data.Dataset.from_generator(sample_generator,
                                             output_types=(tf.string),
                                             output_shapes=(tf.TensorShape([2])))
    dataset = dataset.map(transform_fn, num_parallel_calls=self.config['prefetch_threads'])
    dataset = dataset.prefetch(self.config['prefetch_capacity'])
    dataset = dataset.repeat()
    dataset = dataset.batch(self.config['batch_size'])
    self.dataset_tf = dataset

  def build_iterator(self):
    self.iterator = self.dataset_tf.make_one_shot_iterator()

  def get_one_batch(self):
    return self.iterator.get_next()

if __name__ == "__main__":

  import cv2
  config={}
  config['input_imdb']="dataset/LASOT_DET2014/validation.pickle"
  config['max_frame_dist']=100
  config['prefetch_threads'] = 8
  config['prefetch_capacity'] = 8
  config['batch_size'] = 1
  config['lmdb_path'] = 'dataset/LASOT_DET2014/validation_lmdb'
  os.environ['CUDA_VISIBLE_DEVICES']=""

  with tf.device('/cpu:0'):
    test_loader = DataLoader(config,is_training=False)
    test_loader.build()
    with tf.Session() as sess:
      while True:
        batch = sess.run(test_loader.get_one_batch())

        assert(len(batch) == 4) #exemplar_image, instance_image, gt_examplar_box, gt_instance_box
        instance = np.uint8(batch[1][0])
        examplar = np.uint8(batch[0][0])
        try: 
          cv2.rectangle(examplar, (batch[2][0][0],batch[2][0][1]),(batch[2][0][2],batch[2][0][3]),(0,255,0),3)
          cv2.rectangle(instance, (batch[3][0][0],batch[3][0][1]),(batch[3][0][2],batch[3][0][3]),(0,255,0),3)
          cv2.imshow("examplar", examplar)
          cv2.imshow("instance", instance)
          cv2.waitKey(0)
        except:
          print(np.shape(examplar), np.shape(instance))