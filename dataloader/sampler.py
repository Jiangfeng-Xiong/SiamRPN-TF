from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle

import numpy as np

class ShuffleSample(object):
  def __init__(self, data_source, shuffle=True):
    self.data_source = data_source
    self.shuffle = shuffle

  def __iter__(self):
    data_idxs = np.arange(len(self.data_source))
    if self.shuffle:
      np.random.shuffle(data_idxs)

    for idx in data_idxs:
      yield idx

def downsample(n_in, n_out, max_frame_dist=1):
  # Get a list of frame distance between consecutive frames
  max_frame_dist = np.minimum(n_in, max_frame_dist)
  possible_frame_dist = range(1, max_frame_dist + 1)
  frame_dist = np.random.choice(possible_frame_dist, n_out - 1)
  end_to_start_frame_dist = np.sum(frame_dist)

  # Check frame dist boundary
  possible_max_start_idx = n_in - 1 - end_to_start_frame_dist
  if possible_max_start_idx < 0:
    n_extra = - possible_max_start_idx
    while n_extra > 0:
      for idx, dist in enumerate(frame_dist):
        if dist > 1:
          frame_dist[idx] = dist - 1
          n_extra -= 1
          if n_extra == 0: break

  # Get frame dist
  end_to_start_frame_dist = np.sum(frame_dist)
  possible_max_start_idx = n_in - 1 - end_to_start_frame_dist
  start_idx = np.random.choice(possible_max_start_idx + 1, 1)
  out_idxs = np.cumsum(np.concatenate((start_idx, frame_dist)))
  return out_idxs


def upsample(n_in, n_out):
  n_more = n_out - n_in
  in_idxs = range(n_in)
  more_idxs = np.random.choice(in_idxs, n_more)
  out_idxs = sorted(list(in_idxs) + list(more_idxs))
  return out_idxs


class Sampler:
  def __init__(self, imdb_path, max_frame_dist, epoch_size=None):
    with open(imdb_path, 'rb') as f:
      imdb = pickle.load(f)

    videos = imdb['videos']
    self.time_steps = 2
    self.video_image_ratio=10
    self.max_frame_dist = max_frame_dist
    
    # is use pesudo pairs with one image
    self.is_with_image = False
    for i in range(len(videos)):
      if len(videos[i]) == 1:
        self.is_with_image = True
        break
    #split videos into two parts
    pesudo_video_index = []
    video_index = []
    empty_folder_num=0
    for i in range(len(videos)):
      if len(videos[i])==1:
        pesudo_video_index.append(i)
      elif len(videos[i])>1:
        video_index.append(i)
      else:
        empty_folder_num = empty_folder_num + 1
    self.videos = [videos[j] for j in video_index]
    self.pesudo_videos = [videos[j] for j in pesudo_video_index]
    print("found %s with %d videos and %d pesudo videos and empty_folder %d"%(imdb_path, len(self.videos), len(self.pesudo_videos), empty_folder_num))

    if epoch_size is None:
      self.epoch_size = len(self.videos)
    else:
      self.epoch_size = int(epoch_size)

  def __getitem__(self, index):
    if self.is_with_image:
      if (index%self.video_image_ratio) != 0:
        video_index = (index//self.video_image_ratio)*self.video_image_ratio + (index-1)%(self.video_image_ratio)
        img_ids = self.videos[video_index % len(self.videos)]
      else:
        pesudo_video_index = index//self.video_image_ratio
        img_ids = self.pesudo_videos[pesudo_video_index % len(self.pesudo_videos)]
    else:
      img_ids = self.videos[index % len(self.videos)]

    n_frames = len(img_ids)
    #come across empty dir
    if n_frames < self.time_steps:
      out_idxs = upsample(n_frames, self.time_steps)
    elif n_frames == self.time_steps:
      out_idxs = range(n_frames)
    else:
      out_idxs = downsample(n_frames, self.time_steps, self.max_frame_dist)

    video = []
    for j, frame_idx in enumerate(out_idxs):
      img_path = img_ids[frame_idx]
      video.append(img_path.encode('utf-8'))
    return video

  def __len__(self):
    return self.epoch_size
