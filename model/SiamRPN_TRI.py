import tensorflow as tf
import os,sys
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

from utils.train_utils import show_pred_bbox
from utils.model_utils import BinWindows
from model.generate_anchors import get_rpn_label
from utils.bbox_ops_utils import np_bbox_transform_inv

from model.Model import Model
from dataloader.data_augmentation import RandomMixUp

import numpy as np
slim = tf.contrib.slim


class SiamRPN_TRI(Model):
    def build_inputs(self):
        if len(self.inputs) > 0:
            with tf.device("/cpu:0"):  
                examplars, instances, gt_examplar_boxes, gt_instance_boxes = self.inputs[0],self.inputs[1],self.inputs[2],self.inputs[3]
                if self.train_config['%s_data_config'%(self.mode)].get('time_decay'):
                    self.time_intervals = self.inputs[4]
                    self.time_intervals.set_shape([self.batch_size])
                def single_img_gt(anchors_tf, gt_box):
                    # TODO(jfxiong) convert get_rpn_label from numpy implementation to tensorflow 
                    return tf.py_func(get_rpn_label, [anchors_tf, tf.reshape(gt_box,[-1,4])],[tf.float32,tf.int32],name="single_img_gt")

                bbox_gts, labels = tf.map_fn(lambda x: single_img_gt(x[0],x[1]),[tf.tile(self.anchors_tf,[self.batch_size,1,1]),gt_instance_boxes],dtype=[tf.float32, tf.int32])

                bbox_gts.set_shape([self.batch_size,None,4])
                labels.set_shape([self.batch_size,None])
                examplars.set_shape([self.batch_size, self.model_config['z_image_size'], self.model_config['z_image_size'], 3])
                instances.set_shape([self.batch_size, self.model_config['x_image_size'], self.model_config['x_image_size'], 3])

                examplars = tf.to_float(examplars)
                instances = tf.to_float(instances)
                
                if self.train_config['%s_data_config'%(self.mode)].get('RandomMixUp', False):
                    instances,self.random_mixup_rate = RandomMixUp(instances)
      
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
            self.gt_examplar_boxes = tf.placeholder(shape=[1, 4],
                                            dtype=tf.float32,
                                            name='gt_examplar_boxes')
            examplars = tf.to_float(self.examplar_feed)
            instances = tf.to_float(self.instance_feed)

        self.examplars = examplars
        self.instances = instances

        if self.model_config.get('Normalize', False):
            self.examplars = self.examplars - tf.reshape([123.68, 116.779, 103.939],[1,1,1,3])
            self.examplars = self.examplars/tf.reshape([58.393, 57.12, 57.375],[1,1,1,3])

            self.instances = self.instances - tf.reshape([123.68, 116.779, 103.939],[1,1,1,3])
            self.instances = self.instances/tf.reshape([58.393, 57.12, 57.375],[1,1,1,3])
   
    def _build_rpn_loss(self):
        """
        self.labels: NxNa(Na=32*32*k)
        self.pred_anchors: Nx32x32x4k
        self.pred_prob: Nx32x32x2k
        self.bbox_gts: NxNax4
        """
        eps = 1e-8
        with tf.name_scope('Loss'):
            def tri_probs_each_pair(scores, labels):
                #triplet loss
                #get pos probs 
                true_score,false_score = scores[:,0],scores[:,1]
                pos_mask = tf.equal(labels, 1)
                neg_mask = tf.equal(labels, 0)
                pos_scores = tf.boolean_mask(true_score, pos_mask)
                neg_scores = tf.boolean_mask(true_score, neg_mask)
                tri_scores = tf.concat([tf.tile(tf.expand_dims(neg_scores, axis=0),[tf.shape(pos_scores)[0],1]),
                                       tf.expand_dims(pos_scores, axis=1)], axis=1)
                tri_probs = tf.nn.softmax(tri_scores)
                
                neg_num = tf.shape(neg_scores)[0]
                norm_alpha = tf.log(tf.to_float(neg_num))/0.6931472 #ln2
                triplet_loss = tf.reduce_mean(-tf.log(tri_probs[:,-1]+eps))/(norm_alpha)
                return triplet_loss
                
            loss_cls_list = tf.map_fn(lambda x: tri_probs_each_pair(x[0],x[1]),[self.pred_probs, self.labels],dtype=tf.float32)
            
            valid_mask = tf.stop_gradient(tf.not_equal(self.labels, -1))  # N*Na
            valid_labels = tf.boolean_mask(self.labels, valid_mask)  # N*num_of_anchors_per_image(=64)
            valid_labels = tf.reshape(valid_labels, [self.batch_size,-1])
            
            valid_labels_flatten_pos = tf.to_float(tf.reshape(valid_labels, [-1]))
            valid_labels_flatten = tf.stack([valid_labels_flatten_pos, 1.0 - valid_labels_flatten_pos], axis=1)
            
            valid_pred_probs = tf.boolean_mask(self.pred_probs, valid_mask)
            valid_pred_probs = tf.reshape(valid_pred_probs, [-1, 2])
            
            pos_mask = tf.stop_gradient(tf.equal(self.labels, 1))  # N*Na
            valid_bbox_gts = tf.boolean_mask(self.bbox_gts, pos_mask)
            valid_pred_boxes = tf.boolean_mask(self.pred_boxes, pos_mask)
            
            if self.train_config['%s_data_config'%(self.mode)].get('time_decay'):
                loss_weights = tf.exp(-tf.to_float(self.time_intervals)/100.0) #N
            else:
                loss_weights = tf.ones([tf.shape(self.labels)[0]])
            is_weighted_mixup = self.train_config['%s_data_config'%(self.mode)].get('weighted_mixup', False)
            if self.train_config['%s_data_config'%(self.mode)].get('RandomMixUp', False) and is_weighted_mixup:
                loss_weights = loss_weights*self.random_mixup_rate
            logstic_weight = tf.boolean_mask(tf.tile(tf.reshape(loss_weights,[-1, 1]), [1,tf.shape(self.labels)[1]]), valid_mask)
            reg_weight = tf.boolean_mask(tf.tile(tf.reshape(loss_weights,[-1, 1, 1]), [1,tf.shape(self.labels)[1],4]), pos_mask)
                
            logstic_loss = tf.losses.softmax_cross_entropy(onehot_labels=valid_labels_flatten, logits=valid_pred_probs,weights=logstic_weight*0.5)
            tri_loss = tf.reduce_mean(loss_cls_list*tf.reshape(loss_weights,[-1]))*0.5
            tf.losses.add_loss(tri_loss)
            self.loss_cls =  tri_loss + logstic_loss
            self.loss_reg = tf.losses.huber_loss(labels=valid_bbox_gts, predictions=valid_pred_boxes,weights=reg_weight*self.model_config['regloss_lambda'])

    def build_loss(self):
        self._build_rpn_loss()
        with tf.name_scope('Loss'):
            self.batch_loss = self.loss_cls + self.loss_reg
            #tf.losses.add_loss(self.batch_loss)
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
