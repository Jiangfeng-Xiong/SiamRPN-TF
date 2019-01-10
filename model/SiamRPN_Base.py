import tensorflow as tf
import functools
import numpy as np
import os
slim = tf.contrib.slim

from utils.model_utils import HannWindows,GaussianWindows,BinWindows
from utils.bbox_transform_utils import bbox_transform_inv
from model.generate_anchors import generate_anchor_all,_make_anchors,get_rpn_label
from utils.train_utils import show_pred_bbox,load_pickle_model
from embeddings import get_scope_and_backbone

class SiamRPN():
    def __init__(self,model_config, train_config=None, mode='train', inputs=[]):
        self.model_config = model_config
        self.train_config = train_config
        self.mode = mode
        assert mode in ['train', 'validation', 'inference']
        ratios = [0.33, 0.5, 1.0, 2.0, 3.0]
        anchor_sizes = [64]
        self.embed_dim = 256
        self.anchor_nums_per_location = len(ratios) * len(anchor_sizes)
        self.score_size = (model_config['x_image_size'] - model_config['z_image_size'])//8 + 1
        self.anchors = generate_anchor_all(base_size=model_config['embed_config']['stride'],
                                           ratios=ratios,
                                           anchor_sizes=anchor_sizes,
                                           field_size=self.score_size, net_shift=model_config['net_shift'])
        
        self.anchors_tf = tf.reshape(tf.convert_to_tensor(self.anchors, tf.float32), [1, -1, 4])
        
        #centered_anchors = generate_centered_anchors(model_config['embed_config']['stride'],[8,],ratios, score_size)
        #self.anchors_bbox = tf.reshape(tf.convert_to_tensor(_make_anchors(centered_anchors[:,2], centered_anchors[:,3], centered_anchors[:,0], centered_anchors[:,1])),[1, -1, 4])
        #self.anchors_tf = tf.reshape(tf.convert_to_tensor(centered_anchors, tf.float32), [1, -1, 4])
        self.arg_scope,self.backbone_fn =  get_scope_and_backbone(self.model_config['embed_config'],self.is_training() )

        self.inputs=inputs
        self.examplars = None
        self.instances = None
        self.batch_loss = None
        self.total_loss = None
    # template build network process
    def build(self, reuse=False):
        with tf.name_scope(self.mode):
            self.build_inputs()
            self.build_image_embeddings(reuse=reuse)
            with slim.arg_scope(self.arg_scope):
                self.build_cls_branch(reuse=reuse)
                self.build_reg_branch(reuse=reuse)
            self.setup_embedding_init()

            if self.mode in ['train', 'validation']:
                self.get_topk_preds(top_num=1)
                self.build_loss()
                self.build_metric()
            else:
                self.get_topk_preds(top_num=-1)
    # Step 2. set up network backbone
    def build_image_embeddings(self, reuse=False):
        @functools.wraps(self.backbone_fn)
        def embedding_fn(images, reuse=False):
            with slim.arg_scope(self.arg_scope):
                return self.backbone_fn(images, reuse=reuse)

        self.examplar_embeds, self.endpoints_z = embedding_fn(self.examplars, reuse=reuse)
        self.instance_embeds, self.endpoints_x = embedding_fn(self.instances, reuse=True)
        
    def build_inputs(self):
        if len(self.inputs) > 0:
            with tf.device("/cpu:0"):  
                examplars, instances, gt_examplar_boxes, gt_instance_boxes = self.inputs[0],self.inputs[1],self.inputs[2],self.inputs[3]

                def single_img_gt(anchors_bbox, gt_box):
                    return tf.py_func(get_rpn_label, [anchors_bbox, tf.reshape(gt_box,[-1,4])],[tf.float32,tf.int32],name="single_img_gt")

                gpu_list = self.train_config['%s_data_config'%(self.mode)].get('gpu_ids','0')
                num_gpus = len(gpu_list.split(','))
                batch_size = self.train_config['%s_data_config'%(self.mode)]['batch_size'] // num_gpus
                
                bbox_gts, labels = tf.map_fn(lambda x: single_img_gt(x[0],x[1]),
                    [tf.tile(self.anchors_tf,[batch_size,1,1]),gt_instance_boxes],
                    dtype=[tf.float32, tf.int32])
                
                bbox_gts.set_shape([batch_size,None,4])
                labels.set_shape([batch_size,None])
                examplars.set_shape([batch_size, self.model_config['z_image_size'], self.model_config['z_image_size'], 3])
                instances.set_shape([batch_size, self.model_config['x_image_size'], self.model_config['x_image_size'], 3])

                examplars = tf.to_float(examplars)
                instances = tf.to_float(instances)
                self.bbox_gts, self.labels = bbox_gts, labels
                self.gt_instance_boxes = gt_instance_boxes
                self.gt_examplar_boxes = gt_examplar_boxes
        else:
            self.examplar_feed = tf.placeholder(shape=[1, self.model_config['z_image_size'], self.model_config['z_image_size'], 3],
                                            dtype=tf.uint8,
                                            name='examplar_input')
            self.instance_feed = tf.placeholder(shape=[1, self.model_config['x_image_size'], self.model_config['x_image_size'], 3],
                                            dtype=tf.uint8,
                                            name='instance_input')
            examplars = tf.to_float(self.examplar_feed)
            instances = tf.to_float(self.instance_feed)

        self.examplars = examplars
        self.instances = instances

        if self.model_config.get('Normalize', False):
            self.examplars = self.examplars - tf.reshape([123.68, 116.779, 103.939],[1,1,1,3])
            self.examplars = self.examplars/tf.reshape([58.393, 57.12, 57.375],[1,1,1,3])

            self.instances = self.instances - tf.reshape([123.68, 116.779, 103.939],[1,1,1,3])
            self.instances = self.instances/tf.reshape([58.393, 57.12, 57.375],[1,1,1,3])

        if self.model_config.get('BinWindow',False):
            self.examplars = BinWindows(self.examplars, self.gt_examplar_boxes)
            
    def build_cls_branch(self, reuse):
        with tf.variable_scope('cls', reuse=reuse):
            with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                cls_feats_z = slim.conv2d(self.examplar_embeds, self.embed_dim * self.anchor_nums_per_location * 2, [3, 3], padding='VALID', scope="conv_cls1")
                cls_feats_x = slim.conv2d(self.instance_embeds, self.embed_dim , [3, 3], padding='VALID', scope="conv_cls2")
            def _translation_match(x, z):
                x = tf.expand_dims(x, 0)
                filter_size = tf.shape(z)[0]
                z = tf.reshape(z, [filter_size, filter_size, self.embed_dim , 2 * self.anchor_nums_per_location])
                return tf.squeeze(tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match'))

        self.pred_probs = tf.map_fn(lambda x: _translation_match(x[0], x[1]),
                                    (cls_feats_x, cls_feats_z), dtype=cls_feats_x.dtype)

        feat_size = tf.shape(self.pred_probs)[1] * tf.shape(self.pred_probs)[2]
        num_of_anchors = self.anchor_nums_per_location * feat_size

        #To be consitant with pytorch version model        
        #self.pred_probs = tf.transpose(self.pred_probs, [3,1,2,0])
        #self.pred_scores = tf.transpose(tf.reshape(tf.nn.softmax(tf.reshape(self.pred_probs, [2,-1]),axis=0),[2, num_of_anchors, -1]),[2,1,0])
        
        self.pred_probs = tf.reshape(self.pred_probs, [-1, num_of_anchors, 2])  # N*Na*2
        self.pred_scores = tf.reshape(tf.nn.softmax(tf.reshape(self.pred_probs, [-1, 2])), tf.shape(self.pred_probs))
        
    def build_reg_branch(self, reuse):
        with tf.variable_scope('regssion', reuse=reuse):
            with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None):
                reg_feats_z = slim.conv2d(self.examplar_embeds, self.embed_dim * self.anchor_nums_per_location * 4, [3, 3], padding='VALID',scope='conv_r1')
                reg_feats_x = slim.conv2d(self.instance_embeds, self.embed_dim, [3, 3], padding='VALID', scope='conv_r2')

                def _translation_match(x, z):
                    x = tf.expand_dims(x, 0)
                    filter_size = tf.shape(z)[0]
                    z = tf.reshape(z, [filter_size, filter_size, self.embed_dim, 4 * self.anchor_nums_per_location])
                    return tf.squeeze(tf.nn.conv2d(x, z, strides=[1, 1, 1, 1], padding='VALID', name='translation_match'))

                self.pred_boxes = tf.map_fn(lambda x: _translation_match(x[0], x[1]),
                                    (reg_feats_x, reg_feats_z), dtype=reg_feats_x.dtype)

                batch_num = tf.shape(self.examplar_embeds)[0]
                feat_dim = self.anchor_nums_per_location * 4
                self.pred_boxes = tf.reshape(self.pred_boxes, [batch_num, self.score_size, self.score_size, feat_dim])
                self.pred_boxes = slim.conv2d(self.pred_boxes, self.anchor_nums_per_location * 4, [1, 1], padding='VALID', scope='regress_adjust')
                
        num_of_anchors = self.anchor_nums_per_location * self.score_size * self.score_size
        
        #self.pred_boxes = tf.transpose(self.pred_boxes, [3,1,2,0])        
        #self.pred_boxes = tf.reshape(self.pred_boxes, [4, num_of_anchors, -1])
        #self.pred_boxes = tf.transpose(self.pred_boxes, [2,1,0])# N*Na*4
        self.pred_boxes = tf.reshape(self.pred_boxes, [-1, num_of_anchors, 4])  # N*Na*4

    def _build_rpn_loss(self):
        """
        self.labels: NxNa(Na=32*32*k)
        self.pred_anchors: Nx32x32x4k
        self.pred_prob: Nx32x32x2k
        self.bbox_gts: NxNax4
        """
        with tf.name_scope('Loss'):
            valid_mask = tf.stop_gradient(tf.not_equal(self.labels, -1))  # N*Na

            valid_labels = tf.boolean_mask(self.labels, valid_mask)  # N*num_of_anchors_per_image(=64)
            valid_labels = tf.to_float(tf.reshape(valid_labels, [-1]))
            valid_labels = tf.stack([valid_labels, 1.0 - valid_labels], axis=1)  #[-1x2]

            valid_pred_probs = tf.boolean_mask(self.pred_probs, valid_mask)
            valid_pred_probs = tf.reshape(valid_pred_probs, [-1, 2])
            self.loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=valid_labels, logits=valid_pred_probs))  # N*?*2

            pos_mask = tf.stop_gradient(tf.equal(self.labels, 1))  # N*Na
            valid_bbox_gts = tf.boolean_mask(self.bbox_gts, pos_mask)
            valid_pred_boxes = tf.boolean_mask(self.pred_boxes, pos_mask)
            self.loss_reg = tf.reduce_mean(tf.losses.huber_loss(labels=valid_bbox_gts, predictions=valid_pred_boxes))  # N*?*4
    def build_metric(self):
        def get_iou(boxes1, boxes2):
            x11, y11, x12, y12 = tf.split(boxes1, 4, axis=1) #N*1
            x21, y21, x22, y22 = tf.split(boxes2, 4, axis=1)

            xA = tf.maximum(x11, x21)
            yA = tf.maximum(y11, y21)
            xB = tf.minimum(x12, x22)
            yB = tf.minimum(y12, y22)

            interArea = tf.maximum((xB - xA + 1), 0) * tf.maximum((yB-yA+1),0)

            boxAArea = (x12 - x11+1)*(y12 - y11 +1)
            boxBArea = (x22 - x21+1)*(y22 - y21 +1)

            iou = tf.reduce_mean(interArea/(boxAArea + boxBArea - interArea))
            return iou
        iou = get_iou(tf.squeeze(self.topk_bboxes), self.gt_instance_boxes)
        tf.summary.scalar('iou', iou, family=self.mode)        

    # Some commom help func
    def build_template(self):
        self.templates = self.examplar_embeds

    def restore_weights_from_checkpoint(self, sess, step=-1):
        checkpoint_path = self.model_config['checkpoint']
        if os.path.isdir(checkpoint_path):
            if step==-1:
                checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
            else:
                checkpoint_path = os.path.join(checkpoint_path, "model.ckpt-%d"%(step))
            if not checkpoint_path:
                print("No checkpoint file found in: {}".format(checkpoint_path))
            else:
                print("Loading model from checkpoint: %s" % checkpoint_path)
                saver = tf.train.Saver()
                saver.restore(sess, checkpoint_path)
                print("Successfully loaded checkpoint: %s" % os.path.basename(checkpoint_path))
        else:
            print("No checkpoint file found in: {}".format(checkpoint_path))
    def setup_embedding_init(self):
        embed_config = self.model_config['embed_config']
        if embed_config['embedding_checkpoint_file']:
          initialize = load_pickle_model()
          def restore_fn(sess):
            tf.logging.info("Restoring embedding variables from checkpoint file %s",
                            embed_config['embedding_checkpoint_file'])
            sess.run([initialize])
          self.init_fn = restore_fn

    def is_training(self):
        return self.mode == 'train'

    def get_topk_preds(self,top_num=1):
        ori_pred_boxes = tf.py_func(bbox_transform_inv, [tf.reshape(tf.tile(self.anchors_tf,[tf.shape(self.pred_boxes)[0],1,1]),[-1,4]),
                                          tf.reshape(self.pred_boxes,[-1,4])],tf.float32)
        ori_pred_boxes = tf.reshape(ori_pred_boxes,tf.shape(self.pred_boxes))

        pred_scores = self.pred_scores[:,:,0]

        def get_topk_pred(pred_boxes,pred_probs):
            if(top_num==-1):
                return pred_boxes, pred_probs
            topk = tf.minimum(top_num,tf.size(pred_probs))
            top_scores, top_indices = tf.nn.top_k(pred_probs,k=topk,sorted=False)
            topk_bboxes = tf.gather(pred_boxes, top_indices)
            return topk_bboxes,top_scores

        self.topk_bboxes, self.topk_scores = tf.map_fn(lambda x: get_topk_pred(x[0],x[1]),
                              (ori_pred_boxes, pred_scores),dtype=(tf.float32,tf.float32))
        return self.topk_bboxes, self.topk_scores #Retval shape: N*Topk*4, N*Topk

    def build_loss(self):
        self._build_rpn_loss()
        with tf.name_scope('Loss'):
            self.batch_loss = self.loss_cls + self.loss_reg*self.model_config['regloss_lambda']
            tf.losses.add_loss(self.batch_loss)
            self.total_loss = tf.losses.get_total_loss()
            mean_total_loss, update_op = tf.metrics.mean(self.total_loss)
            with tf.control_dependencies([update_op]):
                tf.summary.scalar('total_loss', mean_total_loss, family=self.mode)

            tf.summary.scalar('batch_loss', self.batch_loss, family=self.mode)
            tf.summary.scalar('loss_cls',   self.loss_cls, family=self.mode)
            tf.summary.scalar('loss_reg',   self.loss_reg, family=self.mode)

            track_instance = tf.py_func(show_pred_bbox,[self.instances, self.topk_bboxes, self.topk_scores, self.gt_instance_boxes],tf.float32)
            tf.summary.image('exemplar', self.examplars, family=self.mode)
            tf.summary.image('instance', track_instance, family=self.mode)

if __name__ == "__main__":
    from model.config import MODEL_CONFIG,TRAIN_CONFIG
    os.environ['CUDA_VISIBLE_DEVICES']="9"

    train_dataloader = DataLoader(TRAIN_CONFIG['train_data_config'], is_training=True)
    train_dataloader.build()
    inputs = train_dataloader.get_one_batch()

    m = SiamRPN(MODEL_CONFIG,TRAIN_CONFIG, inputs)
    m.build()
    global_variables_init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(global_variables_init_op)
        endpoint = sess.run(m.endpoints_z)
        for e in endpoint:
            print(endpoint[e].shape)
