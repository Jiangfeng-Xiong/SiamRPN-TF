import os
import os.path as osp
import sys
from glob import glob
from multiprocessing.pool import ThreadPool

import cv2
from cv2 import imread, imwrite
import numpy as np
from tqdm import tqdm
import pandas as pd

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
    saved_list = glob(track_save_dir+"/*.jpg")
    crop_imgs_size = len(saved_list)
    origin_img_size = len(glob(video_dir+"/*.jpg"))
    if((crop_imgs_size+count)==origin_img_size):
      print("video already croped, skip this video")
      return
    else:
      print("crop_imgs_size: %d, origin_img_size: %d, out-of-view: %d"%(crop_imgs_size, origin_img_size, count))
    #return
    have_croped_list = [ i.split('.')[0]+'.jpg' for i in saved_list]

  img_files = glob(video_dir+"/*.jpg")
  img_files.sort()

  with open(txt_anno,'r') as f:
    for index, line in enumerate(tqdm(f)):
      if img_files[index] in have_croped_list:
        print("img %s has been croped, skip"%(img_files[index]))
        continue
      img = None
      img = imread(img_files[index])
      if isinstance(img, type(None)):
        continue

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
    if os.path.exists(txt_anno):
        parser_txt_anno(video_dir, video, txt_anno, track_save_dir)
    else:
        print("%s not exists"%(txt_anno))

def generate_imgpath_and_gt(csv_file,d_set_dir='yt_bb_detection_validation'):
    df = pd.DataFrame.from_csv(csv_file, header=None, index_col=False)
    col_names = ['youtube_id', 'timestamp_ms','class_id','class_name',
             'object_id','object_presence','xmin','xmax','ymin','ymax']
    df.columns = col_names
    vids = df['youtube_id'].unique()
    
    for vid in vids:
        data = df[df['youtube_id']==vid]
        object_id = 0
        while len(data[data['object_id']==object_id])>0:
            target_data = data[data['object_id']==object_id]
            target_data = target_data[target_data['object_presence']=='present']
            
            video_dir = os.path.join(d_set_dir,vid+"_"+str(object_id))
            object_id = object_id + 1
            
            img_files = glob(video_dir+"/*.jpg")
            img_files = sorted(img_files, key=lambda x:int(x.split('/')[-1].split('.')[0]))
            num_items = (target_data.size)/len(col_names)
            if len(img_files)==0: 
                print("%s empty"%(video_dir))
                continue
            if len(img_files)!= num_items:
                print("image files num %d != csv gt nums %d"%(len(img_files), num_items))
                if os.path.exists(video_dir):
                    print("image files is not as expected, remove %s"%(video_dir))
                    #os.removedirs(video_dir)
                else:
                   print("image files is not as expected due to unable to download videos")
                   pass
                continue
            
            gt_path = os.path.join(video_dir,"groundtruth.txt")
            gt_file = open(gt_path,"w")
            frame_index = 0
            for index, row in target_data.iterrows():
                frame_path = img_files[frame_index]
                frame_index = frame_index + 1
                assert(os.path.exists(frame_path))
                
                img = cv2.imread(frame_path)
                if isinstance(img, type(None)):
                    gt_file.write("%d,%d,%d,%d\n"%(0,0,0,0))
                    continue
                h, w, c = img.shape 
                x1, x2, y1, y2 = row.values[6:10]
            
                x1 = int(x1*w)
                x2 = int(x2*w)
                y1 = int(y1*h)
                y2 = int(y2*h)
            
                target_w = (x2 - x1) + 1
                target_h = (y2 - y1) + 1
                gt_file.write("%d,%d,%d,%d\n"%(x1,y1,target_w, target_h))
            gt_file.close()
            print("generate gtfile %s "%(gt_path))


if __name__ == '__main__':
  source_data = '/157Dataset/data-xiong.jiangfeng/Projects/Dataset/YOUTUBE-BB'
  save_dir = '/157Dataset/data-xiong.jiangfeng/Projects/Dataset/YOUTUBE-BB-Curation'
  
  csv_val = "/157Dataset/data-xiong.jiangfeng/Projects/Dataset/YOUTUBE-BB/yt_bb_detection_validation.csv"
  csv_train = "/157Dataset/data-xiong.jiangfeng/Projects/Dataset/YOUTUBE-BB/yt_bb_detection_train.csv"
  
  crop_filter = "yt_bb_detection_validation"
  csv_file = csv_train if crop_filter=="yt_bb_detection_validation" else csv_val
  
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
      if d==crop_filter:
        generate_imgpath_and_gt(csv_file, source_data+'/'+crop_filter)
        process_split(source_data, save_dir, d)
