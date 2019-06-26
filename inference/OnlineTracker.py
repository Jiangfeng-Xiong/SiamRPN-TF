from utils.infer_utils import convert_bbox_format,Rectangle, get_crops,corrdinate_to_bbox,bbox_to_corrdinate,overlap,NMS
from utils.infer_utils import safe_imread,vis_square,TargetState
import cv2
import numpy as np
import os
from tqdm import tqdm
import time
import copy
import tensorflow as tf
from tensorflow.contrib.layers import group_norm
slim = tf.contrib.slim
Project_root="/home/lab-xiong.jiangfeng/Projects/SiameseRPN"

#----------------------OnlineNet--------------------------------
class OnlineNet():
    def __init__(self,net_config={}, is_training=False, reuse=False):
        self.net_config = net_config
        self.is_training = is_training
        with tf.variable_scope("OnlineNet",reuse=reuse) as scope:
            self.build_input()
            self.build_net()
            if is_training:
                self.build_opt()
    def build_input(self):
        self.inps = tf.placeholder(shape=[None, self.net_config['template_size'], self.net_config['template_size'], 3],dtype=tf.float32,name='inps')
        self.label = tf.placeholder(shape=[None, self.net_config['output_size'], self.net_config['output_size'], 1],dtype=tf.float32,name='label')
    def build_net(self):
      bn_normalizer_params = {'decay': self.net_config['bn_decay'],'epsilon': 1e-6,'is_training': self.is_training,"trainable": True,'updates_collections': None, "scale":True}
      with slim.arg_scope([slim.conv2d],
          weights_regularizer=slim.l2_regularizer(self.net_config['weight_decay']),
          weights_initializer=slim.variance_scaling_initializer(factor=2.0, mode='FAN_OUT', uniform=False),
          biases_initializer=None,
          padding='VALID',
          trainable=True,
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,normalizer_params=bn_normalizer_params):
        
        trainable = not self.net_config['use_part_filter'] or self.net_config['finetune_part_filter']
        
        #multi scale filter 3x3 7x7 11x11
        self.conv1_s1= slim.conv2d(self.inps, self.net_config['conv_dims'][0]//3,[self.net_config['conv1_ksizes'][0]]*2, 1, scope='conv1_s1',trainable=trainable)
        self.conv1_s2= slim.conv2d(self.inps, self.net_config['conv_dims'][0]//3,[self.net_config['conv1_ksizes'][1]]*2, 1, scope='conv1_s2',trainable=trainable)
        self.conv1_s3= slim.conv2d(self.inps, self.net_config['conv_dims'][0]//3,[self.net_config['conv1_ksizes'][2]]*2, 1, scope='conv1_s3',trainable=trainable)
        
        featuremap_sizes = [self.net_config['template_size']-self.net_config['conv1_ksizes'][0]+1,
                            self.net_config['template_size']-self.net_config['conv1_ksizes'][1]+1,
                            self.net_config['template_size']-self.net_config['conv1_ksizes'][2]+1]
        
        self.net = tf.concat([tf.image.central_crop(self.conv1_s1,featuremap_sizes[2]*1.0/featuremap_sizes[0]),
                              tf.image.central_crop(self.conv1_s2, featuremap_sizes[2]*1.0/featuremap_sizes[1]),
                                                    self.conv1_s3], axis=3)
        #self.net = slim.dropout(self.net,keep_prob=self.net_config['dropout_keep_rate'],noise_shape=[1,1,1,tf.shape(self.net)[3]], is_training=self.is_training,scope='conv1_drop')
        for i in range(len(self.net_config['conv_dims'])-1):
            self.net = slim.conv2d(self.net, self.net_config['conv_dims'][i+1], [self.net_config['conv_ksizes'][i]]*2, 1, scope='conv%i'%(i+2))
            self.net = slim.dropout(self.net,keep_prob=self.net_config['dropout_keep_rate'],noise_shape=[1,1,1,tf.shape(self.net)[3]], is_training=self.is_training,scope='conv%i_drop'%(i+2))
            
        self.logits = slim.conv2d(self.net, 1, [1,1], 1, scope='logits',activation_fn=None,normalizer_fn=None)
        self.net = tf.nn.sigmoid(self.logits)
    def init_conv1(self, sess, Params):
        for i in range(3):
            var_in_model = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,"OnlineNet/conv1_s%d"%(i+1))
            op = tf.assign(var_in_model[0], Params[i])
            sess.run(op)
            
    def build_opt(self):
        windows = tf.matmul(tf.expand_dims(tf.contrib.signal.hann_window(self.net_config['output_size']),1),
                            tf.expand_dims(tf.contrib.signal.hann_window(self.net_config['output_size']),0))
        windows = tf.stop_gradient(tf.reshape(windows/tf.reduce_max(windows),[1, self.net_config['output_size'], self.net_config['output_size'], 1]))
            
        onlinetrain_global_step = tf.Variable(initial_value=0,name='onlinetrain_global_step', trainable=False,
                      collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])
        windows = tf.tile(windows, [tf.shape(self.label)[0], 1, 1, 1])
        
        windows = 1.0 + self.label*(windows - 1.0) # windows=1 if label ==0 else windows
                
        logistic_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.label, logits=self.logits)
        logistic_loss = tf.reduce_mean(logistic_loss*windows)
        self.total_loss = tf.losses.get_regularization_loss("OnlineNet") + logistic_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=self.net_config['online_lr'],epsilon=self.net_config['epsilon'])
        #optimizer = tf.train.MomentumOptimizer(self.net_config['online_lr'], 0.9)
        self.train_op = tf.contrib.layers.optimize_loss(loss=self.total_loss,learning_rate=self.net_config['online_lr'],optimizer=optimizer, global_step=onlinetrain_global_step,clip_gradients=1.0)

#--------------------------OnlineTracker---------------------------------
class OnlineTracker(object):
  """
  step 1. use first frame to init the tracker
  step 2. tracke every frame based on the predicted location of previous frame
  """
  def __init__(self, sess, model, track_config, online_config={}, show_video=False):
    self.model = model
    self.sess = sess
    self.track_config = track_config
    self.model_config = model.model_config
    self.x_image_size = track_config['x_image_size']  # Search image size
    self.z_image_size = track_config.get('z_image_size', 127)
    self.score_size = (self.x_image_size - self.z_image_size)//(model.model_config['embed_config']['stride']) + 1
    self.log_level = track_config['log_level']
    self.conf_threshold = 0.05
    self.image_use_rgb = True
    self.save_video=False
    self.show_video=show_video
    self.auto_increase=False
    self.collect=True
    self.online_update=True
    self.frame_id = 0
    self.debug = online_config['debug']
    #configs
    #online filter configs
    self.template_size = online_config['template_size']
    self.OnlineRankWeight = online_config['OnlineRankWeight']
    self.online_config = online_config
  def get_patches(self, template,patch_size):
      #stride = patch_size//2
      stride = patch_size if patch_size<(self.template_size)//4 else patch_size//4
      X = (self.template_size-patch_size)//stride + 1
      Y = (self.template_size-patch_size)//stride + 1
      N,D = X*Y, patch_size*patch_size*3 #RGB values
      P = np.zeros((N, D), dtype=np.uint8)
      for i in range(X):
          for j in range(Y):
              patch = template[stride*i:stride*i+patch_size,stride*j:stride*j+patch_size]
              P[i*Y + j] = np.reshape(patch, [-1])
      #print("Total get %d patches"%(N))
      return P
  def get_filter(self, P, n_pos, patch_size, filter_num):
      def normalize(NDArray,submean=False):
          if submean:
            NDArray = NDArray - np.mean(NDArray.astype(np.float32), axis=1, keepdims=True)
          norm = NDArray/(np.linalg.norm(NDArray,axis=1, keepdims=True)+1e-10)
          return norm
          
      P_norm = normalize(P, submean=True)
      scores = np.abs(np.matmul(P_norm, P_norm.T))
      mean_corrlation = (np.sum(scores[0:n_pos,:], axis=1)-1)/np.shape(P_norm)[0]
      
      sort_index_rgb = np.argsort(mean_corrlation)
      Posfilters=[]
      
      for i in range(filter_num):
          filter_reshape = np.reshape(P_norm[sort_index_rgb[i]], [patch_size,patch_size,3])
          Posfilters.append(np.transpose(filter_reshape,[2,0,1]))
          #print("Select filter with corrleation %.3f"%(mean_corrlation[sort_index_rgb[i]]))

      return np.transpose(np.array(Posfilters),[2,3,1,0])

  def getOnlineRankScore(self, template, search_img, candidate_box, pred_scores):
    resized_template = cv2.resize(template,(self.template_size, self.template_size))
    keep_indexs = NMS(candidate_box, pred_scores)
    
    #online traing example mining 
    if self.online_update:
        for i, index in enumerate(keep_indexs):
            if i == 0 and self.frame_id<5 :
                box = np.array(candidate_box[index],np.int32)
                crop_img = cv2.resize(search_img[box[1]:box[3],box[0]:box[2]], (self.template_size,self.template_size))
                if self.frame_id==0 : 
                    self.pos_collections.append(resized_template)
                self.pos_collections.append(crop_img)
            elif i!=0 and self.frame_id % 5==0:
                box = np.array(candidate_box[index],np.int32)
                crop_img = search_img[box[1]:box[3],box[0]:box[2]]
                if np.shape(crop_img)[0]:
                    self.online_neg_collections.append(cv2.resize(crop_img,(self.template_size,self.template_size)))
                if len(self.online_neg_collections)>10:
                    self.online_neg_collections.popleft()
            else:
                pass # Stop Update Postive Example
        if self.debug:
            cv2.imshow("pos collection", np.concatenate(self.pos_collections, axis=1))
            cv2.imshow("neg collection", np.concatenate(self.neg_collections, axis=1))
            if len(self.online_neg_collections): cv2.imshow("online neg collection", np.concatenate(self.online_neg_collections, axis=1))
            cv2.waitKey(0)
                
#-------------------------get filter and init the first conv layer
    if self.collect:
        self.collect=False
        if self.online_config['use_part_filter']:
            filter_sizes=self.online_config['conv1_ksizes']
            conv1_init_params=[]
            for i,filter_size in enumerate(filter_sizes):
                pos_matrix, neg_matrix=[],[]
                for j in range(len(self.pos_collections)):
                    if j==0: pos_matrix = self.get_patches(self.pos_collections[0], filter_size)
                    else: pos_matrix = np.concatenate((pos_matrix,self.get_patches(self.pos_collections[j],filter_size)), axis=0)
                for j in range(len(self.neg_collections)):
                    if j==0: neg_matrix = self.get_patches(self.neg_collections[0], filter_size)
                    else: neg_matrix = np.concatenate((neg_matrix,self.get_patches(self.neg_collections[j],filter_size)), axis=0)
                Matrix = np.concatenate((pos_matrix,neg_matrix),axis=0) if len(neg_matrix)!=0 else pos_matrix
                Posfilters = self.get_filter(Matrix,np.shape(pos_matrix)[0],filter_size,self.online_config['conv_dims'][0]//len(filter_sizes))
                conv1_init_params.append(Posfilters)
                if self.debug:
                    vis_square(np.transpose(Posfilters,[3,0,1,2]),"Posfilters_%dx%d"%(filter_size, filter_size))
                    cv2.waitKey(0)
            self.model.online_net.init_conv1(self.sess, conv1_init_params)

##==================================== online data collection ===========================================
    #update when frame_id<=10 and distractors exist
    update_flag = (self.frame_id>=10 and len(keep_indexs) > 1) or self.frame_id<=10
    if self.frame_id % 5==0 and update_flag:
        ## load dataset, POS and NEG
        N_pos,N_neg = len(self.pos_collections),len(self.neg_collections)+len(self.online_neg_collections)
        N_example=  N_pos+ N_neg
        labels = np.zeros((N_example, self.online_config['output_size'], self.online_config['output_size'], 1))
        labels[0:N_pos] = np.ones((N_pos, self.online_config['output_size'], self.online_config['output_size'],1))
        
        if len(self.online_neg_collections):
            inps = np.concatenate([self.pos_collections, self.neg_collections, self.online_neg_collections], axis=0)
        else:
            inps = np.concatenate([self.pos_collections, self.neg_collections], axis=0)
        total_train_step = 300 if self.frame_id<=10 else 30
        conf_th = 0.5 if self.frame_id<=10 else self.online_config['conf_th']
#========================================= Online Training ======================================
        conf,step,conf_avg = 0,0,0
        while conf_avg <= conf_th and step<total_train_step:
            _, loss, res,label = self.sess.run([self.model.online_net.train_op,self.model.online_net.total_loss, self.model.online_net.net, self.model.online_net.label],feed_dict={self.model.online_net.inps: inps, self.model.online_net.label: labels})
            mean_pos_value = np.mean(res[0:N_pos])
            mean_neg_value = np.mean(res[N_pos:])
            conf = mean_pos_value - mean_neg_value
            lambda_param = 0.9
            conf_avg = lambda_param*conf_avg + (1-lambda_param)*conf if conf_avg!=0 else conf
            print("step %d/%d with loss=%.3f and conf=%.3f, avg_conf=%.3f"%(step+1, total_train_step,loss, conf, conf_avg))
            if self.debug:
                vis_square(res,"train_resose")
                vis_square(label,"train_label")
                for i in range(3):
                    filter = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "OnlineNet/conv1_s%d"%(i+1))[0]
                    vis_square(np.transpose(self.sess.run(filter),[3,0,1,2]),"OnlineNet/conv1_s%d"%(i+1))
            step = step + 1
    ##=============================================================================================
    assert(len(keep_indexs) > 0)
    conv_inputs=np.zeros((len(keep_indexs),self.template_size,self.template_size,3))
    for i,index in enumerate(keep_indexs):
        box = np.array(candidate_box[index],np.int32)
        crop_img = cv2.resize(search_img[box[1]:box[3],box[0]:box[2]], (self.template_size,self.template_size))
        conv_inputs[i] = np.reshape(crop_img,[1,self.template_size,self.template_size,3])
    
    ###Inference and re-Rank
    respones = self.sess.run(self.model.online_valnet.net, feed_dict={self.model.online_valnet.inps: conv_inputs})
    if self.debug:
        vis_square(conv_inputs,"inference_Candidate")
        vis_square(respones,"inference_respones")
    
    alpha = np.zeros((len(pred_scores)))
    for i in range(len(keep_indexs)):
        alpha[keep_indexs[i]] = np.mean(respones,axis=(1,2,3))[i]
        #print("corrlation: %.3f"%(alpha[keep_indexs[i]])),
    #print("")
    
    alpha = np.maximum(alpha/(1e-6+np.abs(np.max(alpha))), 0)
    self.frame_id = self.frame_id + 1
    return alpha
       
  def track_init(self, first_bbox, first_frame_image_path):
    print(first_frame_image_path)
    first_frame_image = safe_imread(first_frame_image_path)
    self.first_frame_image = cv2.cvtColor(first_frame_image, cv2.COLOR_BGR2RGB) if self.image_use_rgb else first_frame_image
        
    self.first_bbox = convert_bbox_format(Rectangle(first_bbox[0],first_bbox[1],first_bbox[2],first_bbox[3]), 'center-based')
    first_image_crop, _, target_size= get_crops(self.first_frame_image, self.first_bbox, self.z_image_size, self.x_image_size, 0.5)

    cx = (self.x_image_size-1)/2.0
    cy = (self.x_image_size-1)/2.0
    gt_examplar_box = np.array([cx - target_size[0]/2.0, cy - target_size[1]/2.0, cx + target_size[0]/2.0, cy + target_size[1]/2.0],np.float32)

    self.img_height,self.img_width,_ = self.first_frame_image.shape

    if self.save_video:
      video_name = first_frame_image_path.split('/')[-3]+'.mp4'
      fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
      result_dir = os.path.join(Project_root,self.track_config['log_dir'])
      if not os.path.exists(result_dir):
        os.makedirs(result_dir)
      video_path = os.path.join(result_dir, video_name)
      print("save video into %s"%(video_path))
      self.video = cv2.VideoWriter(video_path, fourcc, 30, (self.img_width,self.img_height))

    def center_crop(img, crop_size=127):
      img_shape = np.shape(img)
      center_y = (img_shape[0]-1)//2
      center_x = (img_shape[1]-1)//2
      h = crop_size
      w = crop_size
      croped_img = img[center_y-h//2:center_y+h//2+1, center_x-w//2:center_x+w//2+1]
      assert(croped_img.shape[0]==crop_size)
      return croped_img
    self.first_image_examplar = center_crop(first_image_crop, self.z_image_size)

    shift_y = (self.x_image_size - self.z_image_size)//2
    shift_x = shift_y
    x1 = int(gt_examplar_box[0] - shift_x)
    y1 = int(gt_examplar_box[1] - shift_y)
    x2 = int(gt_examplar_box[2] - shift_x)
    y2 = int(gt_examplar_box[3] - shift_y)
    self.gt_examplar_boxes = np.reshape(np.array([x1, y1 ,x2 ,y2]),[1,4])

    self.template_img = self.first_image_examplar[y1:y2,x1:x2,:]

    self.current_target_state = TargetState(bbox=self.first_bbox)
    self.window = np.tile(np.outer(np.hanning(self.score_size), np.hanning(self.score_size)).flatten(),5) #5 is the number of aspect ratio anchors
    from collections import deque
    
    self.pos_collections=deque()
    self.neg_collections=deque()
    self.online_neg_collections=deque()
    self.frame_id = 0
    
    # add negtivate example aroud the template
    first_image_crop_copy = copy.copy(self.first_image_examplar)
    first_image_crop_copy[y1:y2,x1:x2,:] = np.mean(self.first_frame_image, axis=(0,1),keepdims=True)
    self.neg_collections.append(cv2.resize(first_image_crop_copy, (self.online_config['template_size'],self.online_config['template_size'])))
    
    
  def track(self, first_bbox,frames,bSaveImage=False,SavePath='/tmp'):
    #1. init the tracker
    self.track_init(first_bbox, frames[0])
    include_first = self.track_config['include_first']
    # Run tracking loop
    reported_bboxs = []
    examplar = np.reshape(self.first_image_examplar,[1,self.z_image_size,self.z_image_size,3])

    cost_time_dict={'load_img': 0.0, 'crop_img': 0.0, 'sess_run': 0.0, 'post_process':0.0}
    for i, filename in tqdm(enumerate(frames)):
      if i > 0 or include_first:  # We don't really want to process the first image unless intended to do so.
        load_img_start = time.time()
        bgr_img = safe_imread(filename)
        load_img_end = time.time()
        cost_time_dict['load_img'] += load_img_end - load_img_start
        crop_img_start = time.time()
        
        current_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB) if self.image_use_rgb else bgr_img
        instance_img, scale_x, _= get_crops(current_img, self.current_target_state.search_box, self.z_image_size, self.x_image_size, 0.5)
        instance = np.reshape(instance_img, [1,self.x_image_size, self.x_image_size,3])
        crop_img_end = time.time()
        cost_time_dict['crop_img'] += crop_img_end - crop_img_start

        sess_run_start = time.time()
        boxes,scores = self.sess.run([self.model.topk_bboxes, self.model.topk_scores],
                                  feed_dict={self.model.examplar_feed: examplar ,
                                             self.model.instance_feed: instance})
        sess_run_end = time.time()
        cost_time_dict['sess_run'] += sess_run_end - sess_run_start

        post_process_start = time.time()
        def padded_size(w, h):
            context = 0.5 * (w + h)
            return np.sqrt((w + context) * (h + context))
        def filter_box(boxes,w,h):
            boxes[:,0] = np.minimum(np.maximum(0, boxes[:,0]),w)
            boxes[:,1] = np.minimum(np.maximum(0, boxes[:,1]),h)
            boxes[:,2] = np.minimum(np.maximum(1, boxes[:,2]),w)
            boxes[:,3] = np.minimum(np.maximum(1, boxes[:,3]),h)
            return boxes
        
        #boxes: 1*NA*4 score: 1*Na
        boxes = boxes[0] #NA*4
        scores = scores[0] #NA
        
        boxes = filter_box(boxes, self.x_image_size,self.x_image_size)
        scales = padded_size((boxes[:,2] - boxes[:,0])/scale_x,(boxes[:,3]-boxes[:,1])/scale_x) #Na
        ratios = (boxes[:,3]-boxes[:,1])/(boxes[:,2] - boxes[:,0])

        scale_change = scales/self.current_target_state.scale
        scale_change = np.maximum(scale_change,1.0/scale_change)
        ratio_change = ratios/(self.current_target_state.ratio)
        ratio_change = np.maximum(ratio_change, 1.0/ratio_change)
        scale_penalty = np.exp(-(scale_change*ratio_change-1)*self.track_config['penalty_k'])
        pscores = scores * scale_penalty

        OnlineRankScore = self.getOnlineRankScore(self.template_img, instance_img, boxes, scores)
        wpscores = pscores*(1-self.OnlineRankWeight) +OnlineRankScore*self.OnlineRankWeight
        
        #window_influence = self.track_config['window_influence']
        #wpscores = wpscores*(1-window_influence) + self.window * window_influence
        #wpscores = pscores*(1-window_influence) + self.window * window_influence
        
        max_index = np.argmax(wpscores)
        
        corrdinates = boxes[max_index] #Top1
        #print("Tracking %d/%d with tracking score:%.2f, wpscore: %.2f, llscore: %.2f"%(i+1, len(frames), scores[max_index],wpscores[max_index],llscores[max_index]))
        # Position within frame in frame coordinates
        res_box = Rectangle(*corrdinate_to_bbox(corrdinates))
        center_x = (self.x_image_size -1.0)/2
        center_y = center_x

        delta_x = (res_box.x - center_x)/scale_x
        delta_y = (res_box.y - center_y)/scale_x

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
        self.current_target_state.scale = padded_size(new_search_w, new_search_h)
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
        elif self.save_video:
          x1,y1,x2,y2 = bbox_to_corrdinate(self.current_target_state.search_box)
          cv2.rectangle(bgr_img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
          cv2.putText(bgr_img, "%.2f,%.2f,%.2f"%(scores[max_index],pscores[max_index],wpscores[max_index] ), (int(x1),int(y1)), 0, 1, (0,255,0),2)
          self.video.write(bgr_img)
        elif self.show_video:
          x1,y1,x2,y2 = bbox_to_corrdinate(self.current_target_state.search_box)
          cv2.rectangle(bgr_img,(int(x1),int(y1)),(int(x2),int(y2)),(0,255,0),2)
          cv2.putText(bgr_img, "%.2f,%.2f,%.2f"%(scores[max_index],pscores[max_index],wpscores[max_index] ), (int(x1),int(y1)), 0, 1, (0,255,0),2)
          cv2.imshow("Tracker", bgr_img)
          cv2.waitKey(1)
        else:
          pass
        post_process_end = time.time()
        cost_time_dict['post_process'] += post_process_end - post_process_start

      else:
        x1,y1,x2,y2 = bbox_to_corrdinate(self.current_target_state.search_box)
        cv2.rectangle(self.first_frame_image,(int(x1),int(y1)),(int(x2),int(y2)),(255,255,255),2)
        #cv2.imshow("Tracker",cv2.cvtColor(self.first_frame_image, cv2.COLOR_RGB2BGR))
        #cv2.imshow("Target",self.first_frame_image)
        #cv2.waitKey(100)
      reported_bbox = convert_bbox_format(self.current_target_state.target_box, 'top-left-based')
      reported_bboxs.append(reported_bbox)

    #for key in cost_time_dict:
    #  cost_time_dict[key]/=len(frames)
    return reported_bboxs
