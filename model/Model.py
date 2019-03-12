import tensorflow as tf
import functools
import numpy as np
import os
slim = tf.contrib.slim

from utils.model_utils import HannWindows,GaussianWindows
from utils.bbox_ops_utils import np_bbox_transform_inv
from utils.tf_bbox_ops_utils import tf_iou

from model.generate_anchors import generate_anchor_all
from utils.train_utils import load_pickle_model
from embeddings import get_scope_and_backbone


class Model:
    def __init__(self,model_config, train_config=None, mode='train', inputs=[]):
        self.model_config = model_config
        self.train_config = train_config
        self.mode = mode
        assert mode in ['train', 'validation', 'inference']
        ratios = [0.33, 0.5, 1.0, 2.0, 3.0]
        anchor_sizes = [64]
        self.embed_dim = 256
        self.anchor_nums_per_location = len(ratios) * len(anchor_sizes)
        self.score_size = (model_config['x_image_size'] - model_config['z_image_size'])//model_config['embed_config']['stride'] + 1
        self.anchors = generate_anchor_all(base_size=model_config['embed_config']['stride'],
                                           ratios=ratios,
                                           anchor_sizes=anchor_sizes,
                                           field_size=self.score_size, net_shift=model_config['net_shift'])
        
        self.anchors_tf = tf.reshape(tf.convert_to_tensor(self.anchors, tf.float32), [1, -1, 4])
        self.arg_scope,self.backbone_fn =  get_scope_and_backbone(self.model_config['embed_config'],self.is_training())
        if mode in ['train', 'validation']:
            gpu_list = self.train_config['%s_data_config'%(self.mode)].get('gpu_ids','0')
            num_gpus = len(gpu_list.split(','))
            self.batch_size = self.train_config['%s_data_config'%(self.mode)]['batch_size'] // num_gpus
        else:
            self.batch_size = 1

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
                self.build_metric()
                self.build_loss()
            else:
                self.get_topk_preds(top_num=-1)

    # Step 1.build input, this may varies due to differtent varient methods
    # Leave it blank
    def build_inputs(self):
        pass

    # Step 2. set up network backbone
    def build_image_embeddings(self, reuse=False):
        @functools.wraps(self.backbone_fn)
        def embedding_fn(images, reuse=False):
            with slim.arg_scope(self.arg_scope):
                return self.backbone_fn(images, reuse=reuse)

        self.examplar_embeds, self.endpoints_z = embedding_fn(self.examplars, reuse=reuse)
        self.instance_embeds, self.endpoints_x = embedding_fn(self.instances, reuse=True)
        
    def build_cls_branch(self, reuse):
        with tf.variable_scope('cls', reuse=reuse):
            with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None,trainable=True):
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

        self.pred_probs = tf.reshape(self.pred_probs, [-1, num_of_anchors, 2])  # N*Na*2
        self.pred_scores = tf.reshape(tf.nn.softmax(tf.reshape(self.pred_probs, [-1, 2])), tf.shape(self.pred_probs))
        
    def build_reg_branch(self, reuse):
        with tf.variable_scope('regssion', reuse=reuse):
            with slim.arg_scope([slim.conv2d], activation_fn=None, normalizer_fn=None,trainable=True):
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
            valid_labels = tf.reshape(valid_labels, [self.batch_size,-1])
            
            valid_labels_flatten_pos = tf.to_float(tf.reshape(valid_labels, [-1]))
            valid_labels_flatten = tf.stack([valid_labels_flatten_pos, 1.0 - valid_labels_flatten_pos], axis=1)  #[-1x2]

            valid_pred_probs = tf.boolean_mask(self.pred_probs, valid_mask)
            valid_pred_probs = tf.reshape(valid_pred_probs, [-1, 2])
            
            pos_mask = tf.stop_gradient(tf.equal(self.labels, 1))  # N*Na
            valid_bbox_gts = tf.boolean_mask(self.bbox_gts, pos_mask)
            valid_pred_boxes = tf.boolean_mask(self.pred_boxes, pos_mask)
            
            self.loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=valid_labels_flatten, logits=valid_pred_probs))  # N*?*2
            self.loss_reg = tf.reduce_mean(tf.losses.huber_loss(labels=valid_bbox_gts, predictions=valid_pred_boxes))  # N*?*4
            """
            if self.train_config['%s_data_config'%(self.mode)].get('time_decay'):
                weights = tf.exp(-tf.to_float(self.time_intervals)/100.0) #N
                batch_num = tf.shape(self.time_intervals)[0]
                cls_valid_anchor_num = tf.shape(valid_labels)[1]
                reg_valid_anchor_num = tf.shape(valid_pred_boxes)[1]
                
                cls_weight = tf.reshape(tf.tile(tf.reshape(weights,[self.batch_size, 1]), [1,cls_valid_anchor_num], 'cls_weight'),[-1])
                cls_weight = tf.to_float(self.batch_size*cls_valid_anchor_num)*cls_weight/tf.reduce_sum(cls_weight) #renormlize the weight
                def get_pos_weights(weights, pos_masks):
                    #weight(N) pos_mask(N*Npos)
                    N = np.shape(pos_masks)[0]
                    tiled_weights=[]
                    for i in range(N):
                        count = np.sum(pos_masks[i,:])
                        tiled_weights = tiled_weights + [weights[i]]*count
                    return np.array(tiled_weights, dtype=np.float32)
                    
                reg_weight = tf.py_func(get_pos_weights,[weights, pos_mask], tf.float32, name = "reg_weight")
                reg_weight.set_shape([None])
                
                reg_weights = tf.to_float(tf.shape(reg_weight)[0]) * reg_weight/tf.reduce_sum(reg_weight)
                reg_weights = tf.tile(tf.expand_dims(reg_weight,axis=1),[1,4])
                
                self.loss_cls = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=valid_labels_flatten, logits=valid_pred_probs, weights= cls_weight))
                self.loss_reg = tf.reduce_mean(tf.losses.huber_loss(labels=valid_bbox_gts, predictions=valid_pred_boxes, weights= reg_weights))
            else:
                self.loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=valid_labels_flatten, logits=valid_pred_probs))  # N*?*2
                self.loss_reg = tf.reduce_mean(tf.losses.huber_loss(labels=valid_bbox_gts, predictions=valid_pred_boxes))  # N*?*4
            """

    def build_loss(self):
        pass

    def build_metric(self):
        self.iou = tf.reduce_mean(tf_iou(tf.squeeze(self.topk_bboxes), self.gt_instance_boxes))
        tf.summary.scalar('iou', self.iou, family=self.mode)

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
                #saver = tf.train.Saver()
                #saver.restore(sess, checkpoint_path)
                restore_op = tf.contrib.slim.assign_from_checkpoint_fn(checkpoint_path, tf.global_variables(), ignore_missing_vars=True)
                restore_op(sess)
                print("Successfully loaded checkpoint: %s" % os.path.basename(checkpoint_path))
        else:
            print("No checkpoint file found in: {}".format(checkpoint_path))
    def setup_embedding_init(self):
        embed_config = self.model_config['embed_config']
        if embed_config['embedding_checkpoint_file']:
          def restore_fn(sess):
            initialize = load_pickle_model()
            tf.logging.info("Restoring embedding variables from checkpoint file %s",
                            embed_config['embedding_checkpoint_file'])
            sess.run([initialize])
          self.init_fn = restore_fn

    def is_training(self):
        return self.mode == 'train'

    def get_topk_preds(self,top_num=1):
        ori_pred_boxes = tf.py_func(np_bbox_transform_inv, [tf.reshape(tf.tile(self.anchors_tf,[tf.shape(self.pred_boxes)[0],1,1]),[-1,4]),
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
