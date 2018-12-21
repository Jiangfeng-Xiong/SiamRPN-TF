from utils.infer_utils import convert_bbox_format, Rectangle, get_crops,corrdinate_to_bbox,bbox_to_corrdinate
import cv2
import numpy as np
import os
from tqdm import tqdm
import time

class TargetState(object):
  def __init__(self, bbox):
    self.search_box = bbox #(cx, cy, w, h) in the original image
    self.target_box = bbox
    self.scale = np.sqrt(bbox.width*bbox.height)
    self.ratio = bbox.height*1.0/bbox.width

class Tracker(object):
  """
  step 1. use first frame to init the tracker
  step 2. tracke every frame based on the predicted location of previous frame
  """
  def __init__(self, sess, model, track_config):
    self.model = model
    self.sess = sess
    self.track_config = track_config
    self.x_image_size = track_config['x_image_size']  # Search image size
    self.log_level = track_config['log_level']
    self.conf_threshold = 0.01
    self.show_tracking=True
    self.auto_increase=False


  def track_init(self, first_bbox, first_frame_image_path):
    print(first_bbox,first_frame_image_path)

    first_frame_image = cv2.cvtColor(cv2.imread(first_frame_image_path), cv2.COLOR_BGR2RGB)
    self.first_frame_image = first_frame_image

    self.first_bbox = convert_bbox_format(Rectangle(first_bbox[0],first_bbox[1],first_bbox[2],first_bbox[3]), 'center-based')
    first_image_crop, _, target_size= get_crops(first_frame_image, self.first_bbox, 127, 255, 0.5)

    cx = (255-1)/2.0
    cy = (255-1)/2.0
    gt_examplar_box = np.array([cx - target_size[0]/2.0, cy - target_size[1]/2.0, cx + target_size[0]/2.0, cy + target_size[1]/2.0],np.float32)

    self.img_height,self.img_width,_ = first_frame_image.shape

    if self.show_tracking:
      video_name = first_frame_image_path.split('/')[-3]+'.mp4'
      fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
      result_dir = os.path.join("/home/lab-xiong.jiangfeng/temp",video_name)
      print("save video into %s"%(result_dir))
      self.video = cv2.VideoWriter(result_dir, fourcc, 30, (self.img_width,self.img_height))

    def center_crop(img):
      img_shape = np.shape(img)
      center_y = (img_shape[0]-1)//2
      center_x = (img_shape[1]-1)//2
      h = img_shape[0]//2
      w = img_shape[1]//2
      croped_img = img[center_y-h//2:center_y+h//2+1, center_x-w//2:center_x+w//2+1]
      assert(croped_img.shape[0]==img_shape[0]//2)
      return croped_img
    self.first_image_examplar = center_crop(first_image_crop)

    shift_y = (255 - 127)//2
    shift_x = shift_y
    x1 = gt_examplar_box[0] - shift_x
    y1 = gt_examplar_box[1] - shift_y
    x2 = gt_examplar_box[2] - shift_x
    y2 = gt_examplar_box[3] - shift_y
    self.gt_examplar_box = np.reshape(np.array([x1, y1 ,x2 ,y2]),[1,4])

    self.current_target_state = TargetState(bbox=self.first_bbox)
    self.window = np.tile(np.outer(np.hanning(17), np.hanning(17)).flatten(),5)

    #if self.show_tracking:
    #  show_target = self.first_image_examplar.copy()
    #  cv2.rectangle(show_target,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
      #cv2.imshow("target",show_target)
      #cv2.waitKey(1)

  def track(self, first_bbox,frames,bSaveImage=False,SavePath='/tmp'):
    #1. init the tracker
    self.track_init(first_bbox, frames[0])
    include_first = self.track_config['include_first']
    # Run tracking loop
    reported_bboxs = []
    examplar = np.reshape(self.first_image_examplar,[1,127,127,3])

    cost_time_dict={'load_img': 0.0, 'crop_img': 0.0, 'sess_run': 0.0, 'post_process':0.0}
    for i, filename in tqdm(enumerate(frames)):
      if i > 0 or include_first:  # We don't really want to process the first image unless intended to do so.
        load_img_start = time.time()
        bgr_img = cv2.imread(filename)
        load_img_end = time.time()
        cost_time_dict['load_img'] += load_img_end - load_img_start

        crop_img_start = time.time()
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        instance_img, scale_x, _= get_crops(rgb_img, self.current_target_state.search_box, 127, 255, 0.5)
        instance = np.reshape(instance_img, [1,255,255,3])
        crop_img_end = time.time()
        cost_time_dict['crop_img'] += crop_img_end - crop_img_start

        sess_run_start = time.time()
        boxes,scores = self.sess.run([self.model.topk_bboxes, self.model.topk_scores],
                              feed_dict={self.model.examplar_feed: examplar ,
                                         self.model.instance_feed: instance})
        sess_run_end = time.time()
        cost_time_dict['sess_run'] += sess_run_end - sess_run_start

        post_process_start = time.time()
        #boxes: 1*NA*4 score: 1*Na
        boxes = boxes[0] #NA*4
        scores = scores[0]
        scales = np.sqrt((boxes[:,2] - boxes[:,0])*(boxes[:,3]-boxes[:,1]))/scale_x #Na
        ratios = (boxes[:,3]-boxes[:,1])/(boxes[:,2] - boxes[:,0])

        scale_change = scales/self.current_target_state.scale
        scale_change = np.maximum(scale_change,1.0/scale_change)
        ratio_change = ratios/(self.current_target_state.ratio)
        ratio_change = np.maximum(ratio_change, 1.0/ratio_change)
        scale_penalty = np.exp(-(scale_change*ratio_change-1)*self.track_config['penalty_k'])
        pscores = scores * scale_penalty

        window_influence = self.track_config['window_influence']
        wpscores = pscores*(1-window_influence) + self.window * window_influence
        
        max_index = np.argmax(wpscores)
        corrdinates = boxes[max_index] #Top1
        #print("Tracking %d/%d with tracking score:%.2f, wpscore: %.2f"%(i+1, len(frames), scores[max_index],wpscores[max_index]))

        # Position within frame in frame coordinates
        res_box = Rectangle(*corrdinate_to_bbox(corrdinates))
        delta_x = (res_box.x - 127.0)/scale_x
        delta_y = (res_box.y - 127.0)/scale_x
        w = res_box.width/scale_x
        h = res_box.height/scale_x
        y = self.current_target_state.target_box.y + delta_y
        x = self.current_target_state.target_box.x + delta_x

        #update seach bbox
        alpha = self.track_config['search_scale_smooth_factor'] * pscores[max_index]
        belta = 0.0
        new_search_cx = max(min(self.img_width,self.current_target_state.target_box.x * belta + (1.0-belta)*x),0.0)
        new_search_cy = max(min(self.img_height,self.current_target_state.target_box.y * belta + (1.0-belta)*y),0.0)
        new_search_w = max(10.0,min(self.current_target_state.target_box.width * (1.0-alpha) + alpha*w, self.img_width))
        new_search_h = max(10.0,min(self.current_target_state.target_box.height * (1.0-alpha) + alpha*h, self.img_height))
        self.current_target_state.target_box = Rectangle(new_search_cx,new_search_cy,new_search_w,new_search_h)
        self.current_target_state.scale = np.sqrt(new_search_w*new_search_h)
        self.current_target_state.ratio = new_search_h*1.0/new_search_w
        
        #auto increase the search region if max score is lower than the conf_threshold
        if(scores[max_index] < self.conf_threshold and self.auto_increase):
          increase_w = min(new_search_w*1.5, self.img_width)
          increase_h = min(new_search_h*1.5, self.img_height)
          self.current_target_state.search_box = Rectangle(new_search_cx,new_search_cy,increase_w,increase_h)
        else:
          self.current_target_state.search_box = self.current_target_state.target_box

        #save and show tracking process
        if bSaveImage:
          cv2.imwrite(SavePath+"/"+os.path.basename(frames[i]), bgr_img)
        elif self.show_tracking:
          x1,y1,x2,y2 = bbox_to_corrdinate(self.current_target_state.search_box)
          cv2.rectangle(bgr_img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
          cv2.putText(bgr_img, "%.2f"%(scores[max_index]), (int(x1),int(y1)), 0, 1, (0,255,0),2)
          #cv2.imshow("Tracker",bgr_img)
          self.video.write(bgr_img)
        else:
          pass
        post_process_end = time.time()
        cost_time_dict['post_process'] += post_process_end - post_process_start

      else:
        x1,y1,x2,y2 = bbox_to_corrdinate(self.current_target_state.search_box)
        #cv2.rectangle(self.first_frame_image,(int(x1),int(y1)),(int(x2),int(y2)),(255,255,255),2)
        #cv2.imshow("Tracker",cv2.cvtColor(self.first_frame_image, cv2.COLOR_RGB2BGR))

      reported_bbox = convert_bbox_format(self.current_target_state.target_box, 'top-left-based')
      reported_bboxs.append(reported_bbox)

    for key in cost_time_dict:
      cost_time_dict[key]/=len(frames)
    print(cost_time_dict)
    return reported_bboxs
