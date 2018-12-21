import os
import os.path as osp
import sys
from glob import glob
from multiprocessing.pool import ThreadPool

import cv2
from cv2 import imread, imwrite
import numpy as np

CURRENT_DIR = osp.dirname(__file__)
ROOT_DIR = osp.join(CURRENT_DIR, '..')
sys.path.append(ROOT_DIR)

from utils.infer_utils import get_crops, Rectangle, convert_bbox_format


def get_track_save_directory(save_dir, split, video):
  return osp.join(save_dir, split, video)

def parser_txt_anno(video_dir, video_id, txt_anno,track_save_dir):
  subfix=".jpg"

  if(len(os.listdir(track_save_dir))==len(os.listdir(video_dir))): return

  with open(txt_anno,'r') as f:
    for index, line in enumerate(f):
      img_name = str(index)
      img_file = os.path.join(video_dir,img_name+subfix)
      #assert os.path.exists(img_file),img_file
      if not os.path.exists(img_file):
        continue

      img = None
      img = imread(img_file)

      line_list = line.split(",")

      bbox = [float(x) for x in line_list]

      target_box = convert_bbox_format(Rectangle(*bbox), 'center-based')
      crop, scale,new_sizes = get_crops(img, target_box,
                            size_z=127, size_x=255,
                            context_amount=0.5)

      savename = osp.join(track_save_dir, '{}.w.{}.h.{}.jpg'.format(img_name,int(np.rint(new_sizes[0])),int(np.rint(new_sizes[1]))))
      if osp.exists(savename):
        continue
      imwrite(savename, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


def process_split(root_dir, save_dir, split):
  data_dir = osp.join(root_dir, split, "frames")
  anno_dir = osp.join(root_dir, split, "anno")
  video_names = os.listdir(data_dir)

  for idx, video in enumerate(video_names):
    print('{split} ({idx}/{total}): Processing {video}...'.format(split=split,idx=idx, total=len(video_names),
                                                                           video=video))
    video_dir = osp.join(data_dir, video)
    txt_anno = osp.join(anno_dir, video+".txt")

    track_save_dir = get_track_save_directory(save_dir, split, video)
    if not osp.exists(track_save_dir):
      os.makedirs(track_save_dir)
    parser_txt_anno(video_dir, video, txt_anno, track_save_dir)

if __name__ == '__main__':
  source_data = osp.join(ROOT_DIR, 'dataset/TrackingNet')

  save_dir = 'dataset/TrackingNet-Curation'

  
  pool = ThreadPool(processes=5)

  one_work = lambda a: process_split(source_data, save_dir, a)

  results = []
  results.append(pool.apply_async(one_work, ['TRAIN_0']))
  results.append(pool.apply_async(one_work, ['TRAIN_1']))
  results.append(pool.apply_async(one_work, ['TRAIN_2']))
  results.append(pool.apply_async(one_work, ['TRAIN_3']))
  results.append(pool.apply_async(one_work, ['TRAIN_4']))
  results.append(pool.apply_async(one_work, ['TRAIN_5']))
  results.append(pool.apply_async(one_work, ['TRAIN_6']))
  results.append(pool.apply_async(one_work, ['TRAIN_7']))
  results.append(pool.apply_async(one_work, ['TRAIN_8']))
  results.append(pool.apply_async(one_work, ['TRAIN_9']))
  results.append(pool.apply_async(one_work, ['TRAIN_10']))
  results.append(pool.apply_async(one_work, ['TRAIN_11']))
  ans = [res.get() for res in results]
