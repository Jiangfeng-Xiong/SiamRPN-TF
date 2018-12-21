"""Save the paths of crops from TrackingNet in pickle format"""
import glob
import os
import os.path as osp
import pickle
import sys
import argparse
import numpy as np

CURRENT_DIR = osp.dirname(__file__)
sys.path.append(osp.join(CURRENT_DIR, '..'))


class DataIter:
  """Container for dataset of one iteration"""
  pass

class Dataset:
  def __init__(self, args):
    self.config = args

  def dataset_iterator(self, video_dirs):
    video_num = len(video_dirs)
    iter_size = 150
    iter_num = int(np.ceil(video_num / float(iter_size)))
    for iter_ in range(iter_num):
      iter_start = iter_ * iter_size
      end = min(iter_start + iter_size, video_num)
      
      iter_videos = video_dirs[iter_start: end]

      data_iter = DataIter()
      num_videos = len(iter_videos)
      instance_videos = []
      for index in range(num_videos):
        print('Processing {}/{}...'.format(iter_start + index, video_num))
        video_dir = iter_videos[index]

        instance_image_paths = glob.glob(video_dir + '/'  + '*.jpg')

          # sort image paths by frame number
        #instance_image_paths = sorted(instance_image_paths,cmp=cmp)
        instance_image_paths = sorted(instance_image_paths,key=lambda x: int(x.split('/')[-1].split('.')[0]))

          # get image absolute path
        instance_image_paths = [os.path.abspath(p) for p in instance_image_paths]
        instance_videos.append(instance_image_paths)
      data_iter.num_videos = len(instance_videos)
      data_iter.instance_videos = instance_videos
      yield data_iter

  def get_all_video_dirs(self):
    ann_dir = os.path.join(self.config.dataset_dir)
    all_video_dirs = []
    train_dirs = os.listdir(ann_dir) #TRAIN0-TRAIN11
    for dir_ in train_dirs:
      train_sub_dir = os.path.join(ann_dir, dir_) # */TRAIN0
      video_names = os.listdir(train_sub_dir)
      train_video_dirs = [os.path.join(train_sub_dir, name) for name in video_names]
      print("%d videos in %s"%(len(train_video_dirs), train_sub_dir))
      all_video_dirs = all_video_dirs + train_video_dirs
    print("totally found %d vidoes"%len(all_video_dirs))
    temp = input("check dataset number, press enter to continue")
    return all_video_dirs

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_dir', type=str, nargs='?', default='dataset/TrackingNet_DET2014/raw_data')
  parser.add_argument('--save_dir', type=str, nargs='?', default='dataset/TrackingNet_DET2014')
  parser.add_argument('--validation_ratio', type=float, nargs='?', default=0.2)

  args = parser.parse_args()
  dataset = Dataset(args)

  all_video_dirs = dataset.get_all_video_dirs()
  import random
  random.seed(1234)
  random.shuffle(all_video_dirs)
  num_validation = int(len(all_video_dirs) * args.validation_ratio)

  ### validation  
  validation_dirs = all_video_dirs[:num_validation]
  validation_imdb = dict()
  validation_imdb['videos'] = []
  for i, data_iter in enumerate(dataset.dataset_iterator(validation_dirs)):
    validation_imdb['videos'] += data_iter.instance_videos
  validation_imdb['n_videos'] = len(validation_imdb['videos'])
  validation_imdb['image_shape'] = (255, 255, 3)
  with open(os.path.join(args.save_dir,'validation.pickle'), 'wb') as f:
    pickle.dump(validation_imdb, f)


  ### train
  train_dirs = all_video_dirs[num_validation:]
  train_imdb = dict()
  train_imdb['videos'] = []
  for i, data_iter in enumerate(dataset.dataset_iterator(train_dirs)):
    train_imdb['videos'] += data_iter.instance_videos
  train_imdb['n_videos'] = len(train_imdb['videos'])
  train_imdb['image_shape'] = (255, 255, 3)
  with open(os.path.join(args.save_dir, 'train.pickle'), 'wb') as f:
    pickle.dump(train_imdb, f)
