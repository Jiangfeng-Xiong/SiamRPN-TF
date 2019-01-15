from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import os.path as osp
import sys
import xml.etree.ElementTree as ET
from glob import glob
from multiprocessing.pool import ThreadPool

import cv2
from cv2 import imread, imwrite

CURRENT_DIR = osp.dirname(__file__)
ROOT_DIR = osp.join(CURRENT_DIR, '..')
sys.path.append(ROOT_DIR)

from utils.infer_utils import get_crops, Rectangle, convert_bbox_format


def get_track_save_directory(save_dir, split, subdir, video):
    subdir_map = {'ILSVRC2015_VID_train_0000': 'vid_a',
                  'ILSVRC2015_VID_train_0001': 'vid_b',
                  'ILSVRC2015_VID_train_0002': 'vid_c',
                  'ILSVRC2015_VID_train_0003': 'vid_d',
                  '': 'vid_e'}
    return osp.join(save_dir, subdir_map[subdir], video)


def process_split(root_dir, save_dir, split, subdir='', ):
    data_dir = osp.join(root_dir, 'Data', 'VID', split)
    anno_dir = osp.join(root_dir, 'Annotations', 'VID', split, subdir)
    video_names = os.listdir(anno_dir)
    for idx, video in enumerate(video_names):
        print('{split}-{subdir} ({idx}/{total}): Processing {video}...'.format(split=split, subdir=subdir,
                                                                               idx=idx, total=len(video_names),
                                                                               video=video))
        video_path = osp.join(anno_dir, video)
        xml_files = glob(osp.join(video_path, '*.xml'))

        for xml in xml_files:
            tree = ET.parse(xml)
            root = tree.getroot()

            folder = root.find('folder').text
            filename = root.find('filename').text

            # Read image
            img_file = osp.join(data_dir, folder, filename + '.JPEG')
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
                
            track_save_dir = get_track_save_directory(save_dir, 'train', subdir, video)
            for idx, object in enumerate(root.iter('object')):
                id = object.find('trackid').text
                
                if img is None:
                   img = imread(img_file)
                 # Get crop
                target_box = convert_bbox_format(Rectangle(*bboxs[idx]), 'center-based')
                crop, scale,new_sizes = get_crops(img, target_box, size_z=127, size_x=255, context_amount=0.5)
                index_sub = "_"+str(idx) if idx > 0 else ""
                save_dir_object = track_save_dir + index_sub
                if not os.path.exists(save_dir_object):
                  os.makedirs(save_dir_object)
                savename = os.path.join(save_dir_object, '{}.w.{}.h.{}.jpg'.format(filename, int(np.rint(new_sizes[0])),int(np.rint(new_sizes[1]))))
                if osp.exists(savename):
                  continue
                imwrite(savename, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

if __name__ == '__main__':
    vid_dir = osp.join(ROOT_DIR, 'dataset/ILSVRC2015')
    save_dir = 'dataset/ILSVRC2015-VID-Curation'

    pool = ThreadPool(processes=5)

    one_work = lambda a, b: process_split(vid_dir, save_dir, a, b)

    results = []
    results.append(pool.apply_async(one_work, ['val', '']))
    results.append(pool.apply_async(one_work, ['train', 'ILSVRC2015_VID_train_0000']))
    results.append(pool.apply_async(one_work, ['train', 'ILSVRC2015_VID_train_0001']))
    results.append(pool.apply_async(one_work, ['train', 'ILSVRC2015_VID_train_0002']))
    results.append(pool.apply_async(one_work, ['train', 'ILSVRC2015_VID_train_0003']))
    ans = [res.get() for res in results]
