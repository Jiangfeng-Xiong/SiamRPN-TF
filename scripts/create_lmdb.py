import cv2
import lmdb
import argparse
import pickle
import functools
from multiprocessing import Pool
from tqdm import tqdm

encode = True


def check_saved_video_index(videos,txn):
	video_num = len(videos)
	for i in range(video_num):
		video_length = len(videos[i])
		for j in range(video_length):
			img_path = videos[i][j]
			key = img_path.encode('utf-8')
			if txn.get(key)!= None:
				print("key already in the database, skip key %s"%(img_path))
			else:
				image = cv2.imread(img_path,1)
				if isinstance(image, type(None)):
					continue
				else:
					return i
	return video_num-1

def worker(videos):
	kv = {}
	video_length = len(videos)
	for j in range(video_length):
		img_path = videos[j]
		key = img_path.encode('utf-8')
		image = cv2.imread(img_path,1)
		if isinstance(image, type(None)):
			print("image %s Error"%(img_path))
			continue
		image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
		if encode:
			_,image = cv2.imencode('.jpg', image)
		kv[key] = image
	return kv

def preprocess_lmdb(input_pickles, output_file):
	env = lmdb.open(output_file, map_size = 109951162777)
	with open(input_pickles, 'rb') as f:
		inputs = pickle.load(f)
	videos = inputs['videos']
	with env.begin() as temp:
		last_index = check_saved_video_index(videos, temp)
		videos = videos[last_index:]

	with Pool(processes=32) as pool:
		for ret in tqdm(pool.imap_unordered(functools.partial(worker), videos),total=len(videos)):
			with env.begin(write=True) as txn:
				for k, v in ret.items():
					if txn.put(k, v, overwrite=False)==False:
						print("key %s exist, skip this key"%(k))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()  
  parser.add_argument('--rootdir', type=str, nargs='?', default="LASOT_DET2014")
  parser.add_argument('--input', type=str, nargs='?', default='train.pickle')
  parser.add_argument('--output', type=str, nargs='?', default='train_lmdb_encode')
  args = parser.parse_args()
  print("start processing with Encode: ", encode)
  input = "dataset/%s/%s"%(args.rootdir,args.input)
  output = "dataset/%s/%s"%(args.rootdir,args.output)
  preprocess_lmdb(input, output)
