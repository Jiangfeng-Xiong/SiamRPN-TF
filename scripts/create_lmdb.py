import cv2
import lmdb
import argparse
import pickle

def preprocess_lmdb(input_pickles, output_file):
	env = lmdb.open(output_file, map_size = 109951162777*20)
	with open(input_pickles, 'rb') as f:
		inputs = pickle.load(f)
	videos = inputs['videos']
	video_num = len(videos)
	total_frames = 0
	total_videos = 0
	total_pics = 0
	with env.begin(write=True) as txn:
		for i in range(video_num):
			video_length = len(videos[i])
			if video_length==1:
				total_pics = total_pics + 1
			else:
				total_videos = total_videos +1
			for j in range(video_length):
				img_path = videos[i][j]
				key = img_path.encode('utf-8')
				if txn.get(key)!= None:
					print("key already in the database, skip key %s"%(img_path))
				else:
					image = cv2.imread(img_path,1)
					if isinstance(image, type(None)):
						print("image %s Error"%(img_path))
						continue
					image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
					txn.put(key, image)
					if total_frames%1000==0 or (i==(video_num-1) and j== (video_length-1)):
						txn.commit()
						txn = env.begin(write=True)
				total_frames = total_frames + 1
				print("processing video %d/%d, %d/%d"%(i+1,video_num, j+1, video_length),key)
	print("total_videos: %d, total_pics: %d , total_frames: %d"%(total_videos,total_pics, total_frames))
	env.close()
	
if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--input_pickles', type=str, nargs='?', default='dataset/LASOT_DET2014/train.pickle')
  parser.add_argument('--output_file', type=str, nargs='?', default='dataset/LASOT_DET2014/train')
  args = parser.parse_args()
  print("start processing ...")
  preprocess_lmdb(args.input_pickles, args.output_file)
