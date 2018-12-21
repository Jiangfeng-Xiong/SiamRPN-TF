from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
from glob import glob
from multiprocessing.pool import ThreadPool
import xml.etree.ElementTree as ET

import cv2
from cv2 import imread, imwrite
import numpy as np

CURRENT_DIR = osp.dirname(__file__)
ROOT_DIR = osp.join(CURRENT_DIR, '..')
sys.path.append(ROOT_DIR)

from utils.infer_utils import get_crops, Rectangle, convert_bbox_format


def get_track_save_directory(save_dir, split, video):
  return osp.join(save_dir, split, video)

def parser_xml_anno(img_file, xml_anno, track_save_dir):
  tree = ET.parse(xml_anno)
  root = tree.getroot()

  img = None

  # Get all object bounding boxes
  bboxs = []
  for object in root.iter('object'):
    bbox = object.find('bndbox')
    xmax = float(bbox.find('xmax').text)
    xmin = float(bbox.find('xmin').text)
    ymax = float(bbox.find('ymax').text)
    ymin = float(bbox.find('ymin').text)
    width = xmax - xmin + 1
    height = ymax - ymin + 1
    bboxs.append([xmin, ymin, width, height])

  for idx, object in enumerate(root.iter('object')):
      #id = object.find('trackid').text
      if img is None:
        img = cv2.imread(img_file)
      target_box = convert_bbox_format(Rectangle(*bboxs[idx]), 'center-based')

      crop, scale,new_sizes = get_crops(img, target_box, size_z=127, size_x=255, context_amount=0.5)

      index_sub = "_"+str(idx) if idx > 0 else ""
      save_dir = track_save_dir + index_sub
      if not os.path.exists(save_dir):
        os.makedirs(save_dir)
      savename = os.path.join(save_dir, '0.w.{}.h.{}.jpg'.format(int(np.rint(new_sizes[0])),int(np.rint(new_sizes[1]))))
      if osp.exists(savename):
        continue
      imwrite(savename, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

def process_split(root_dir, save_dir, split):
  data_dir = osp.join(root_dir, "images", split)
  anno_dir = osp.join(root_dir, "bboxes", split)
  images = os.listdir(data_dir)
  print("found %d images"%(len(images)))

  for idx, im in enumerate(images):
    print('{split} ({idx}/{total}): Processing {im}...'.format(split=split,idx=idx, total=len(images),
                                                                           im=im))
    xml_anno = osp.join(anno_dir, im.replace("JPEG","xml"))

    track_save_dir = get_track_save_directory(save_dir, split,split+"%05d"%(idx)) # a/
    #if not osp.exists(track_save_dir):
    #  os.makedirs(track_save_dir)

    parser_xml_anno(os.path.join(data_dir,im), xml_anno, track_save_dir)


if __name__ == '__main__':
  source_data = osp.join(ROOT_DIR, 'dataset/ILSVRC2014_DET')

  save_dir = 'dataset/ILSVRC2014_DET-Curation'

  pool = ThreadPool(processes=7)

  one_work = lambda x: process_split(source_data, save_dir, x)

  results = []
  results.append(pool.apply_async(one_work, ['a']))
  results.append(pool.apply_async(one_work, ['b']))
  results.append(pool.apply_async(one_work, ['c']))
  results.append(pool.apply_async(one_work, ['d']))
  results.append(pool.apply_async(one_work, ['e']))
  results.append(pool.apply_async(one_work, ['f']))
  results.append(pool.apply_async(one_work, ['g']))

  ans = [res.get() for res in results]
