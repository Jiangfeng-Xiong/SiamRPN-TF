import os
import os.path as osp
import sys
from glob import glob
from multiprocessing.pool import ThreadPool

import cv2
from cv2 import imread, imwrite
import numpy as np
from tqdm import tqdm

CURRENT_DIR = osp.dirname(__file__)
ROOT_DIR = osp.join(CURRENT_DIR, '..')
sys.path.append(ROOT_DIR)

from utils.infer_utils import get_crops, Rectangle, convert_bbox_format

def get_track_save_directory(save_dir, split, video):
  return osp.join(save_dir, split, video)
def count_out_of_view_frame(txt_anno):
  count=0
  with open(txt_anno,'r') as f:
    for index, line in enumerate(f):
      line_list = line.split(",")
      bbox = [int(float(x)) for x in line_list]

      #skip out-of-view frames
      if bbox[2]<=0 or bbox[3]<=0:
        count = count + 1
  return count


def parser_txt_anno(video_dir, video_id, txt_anno,track_save_dir):
  if not osp.exists(track_save_dir):
    os.makedirs(track_save_dir)
    have_croped_list = []
  else:
    count = count_out_of_view_frame(txt_anno)
    saved_list = os.listdir(track_save_dir)
    crop_imgs_size = len(saved_list)
    origin_img_size = len(os.listdir(video_dir))
    if((crop_imgs_size+count)==origin_img_size):
      print("video already croped, skip this video")
      return
    else:
      print("crop_imgs_size: %d, origin_img_size: %d, out-of-view: %d"%(crop_imgs_size, origin_img_size, count))
    #return
    have_croped_list = [ i.split('.')[0]+'.jpg' for i in saved_list]

  #img_files = os.listdir(video_dir)
  #img_files.sort()
  img_files = glob(video_dir+"/*.jpg")
  img_files.sort()

  with open(txt_anno,'r') as f:
    for index, line in enumerate(tqdm(f)):
      if img_files[index] in have_croped_list:
        print("img %s has been croped, skip"%(img_files[index]))
        continue
      img = None
      img = imread(img_files[index])

      line_list = line.split(",")
      bbox = [int(float(x)) for x in line_list]

      #skip out-of-view frames
      if bbox[2]==0 or bbox[3]==0:
        print("found out-of-view frame, skip this frame")
        continue

      #convert from 1-based to 0-based
      bbox[0] = bbox[0]-1
      bbox[1] = bbox[1]-1

      target_box = convert_bbox_format(Rectangle(*bbox), 'center-based')
      #target_box = Rectangle(*bbox)
      if target_box.width<=0 or target_box.height<=0:
        print("target_box error in",txt_anno, index)
        continue 

      crop, scale,new_sizes = get_crops(img, target_box,
                            size_z=127, size_x=255,
                            context_amount=0.5)
      img_id = img_files[index].split('/')[-1].split('.')[0]
      savename = osp.join(track_save_dir, '{}.w.{}.h.{}.jpg'.format(img_id,int(np.rint(new_sizes[0])),int(np.rint(new_sizes[1]))))
      #print(savename)
      if osp.exists(savename):
        continue
      imwrite(savename, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

def process_split(root_dir, save_dir, split):
  data_dir = osp.join(root_dir, split)
  video_names = os.listdir(data_dir)
  video_names.sort(reverse=False)

  for idx, video in enumerate(tqdm(video_names)):
    print('{split} ({idx}/{total}): Processing {video}...'.format(split=split,idx=idx, total=len(video_names),
                                                                           video=video))
    video_dir = osp.join(data_dir, video)
    txt_anno = osp.join(data_dir, video,"groundtruth.txt")

    track_save_dir = get_track_save_directory(save_dir, split, video)
    parser_txt_anno(video_dir, video, txt_anno, track_save_dir)


if __name__ == '__main__':
  source_data = '/157Dataset/data-xiong.jiangfeng/GOT10k/data'
  save_dir = '/157Dataset/data-xiong.jiangfeng/GOT10k/GOT10k-Curation'

  multi_thread = False
  if multi_thread:
    pool = ThreadPool(processes=16)
    one_work = lambda a: process_split(source_data, save_dir, a)
    results = []
    dirs = os.listdir(source_data)
    dirs.sort(reverse=True)
    for d in dirs:
      results.append(pool.apply_async(one_work, [d]))
    ans = [res.get() for res in results]
  else:
    dirs = os.listdir(source_data)
    dirs.sort()
    for d in dirs:
      #if d=='val': continue
      process_split(source_data, save_dir, d)
