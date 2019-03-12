import cv2
import lmdb
import argparse
import pickle
import functools
from multiprocessing import Pool
from tqdm import tqdm

encode = True


def check_saved_video_indexs(videos,txn):
	video_num = len(videos)
	to_be_process_video_ids=[]
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
					to_be_process_video_ids.append(i)
					print("add video index %d "%(i))
					break
	return to_be_process_video_ids

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

def preprocess_lmdb(input_pickles, output_file, resume=False):
	env = lmdb.open(output_file, map_size = 109951162777*3)
	try: 
		with open(input_pickles, 'rb') as f:
			inputs = pickle.load(f)
	except:
		pass
		return
	videos = inputs['videos']
	if resume:
		with env.begin() as temp:
			to_be_process_video_ids = check_saved_video_indexs(videos, temp)
			print("Total unsaved video num: %d"%(len(to_be_process_video_ids)))
			videos = [videos[i] for i in to_be_process_video_ids]

	with Pool(processes=32) as pool:
		for ret in tqdm(pool.imap_unordered(functools.partial(worker), videos),total=len(videos)):
			with env.begin(write=True) as txn:
				for k, v in ret.items():
					if txn.put(k, v, overwrite=False)==False:
						print("key %s exist, skip this key"%(k))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()  
  parser.add_argument('--rootdir', type=str, nargs='?', default="DET2014_VID2015_LASOT_GOT10k")
  parser.add_argument('--resume', type=int, nargs='?', default=0)
  args = parser.parse_args()
  print("start processing with Encode: ", encode)
  
  print("Creating train_lmdb")
  preprocess_lmdb("dataset/%s/%s"%(args.rootdir,"train.pickle"), "dataset/%s/%s"%(args.rootdir,"train_lmdb_encode"),args.resume)

  #print("Creating validation_lmdb")
  #preprocess_lmdb("dataset/%s/%s"%(args.rootdir,"validation.pickle"), "dataset/%s/%s"%(args.rootdir,"validation_lmdb"),args.resume)
