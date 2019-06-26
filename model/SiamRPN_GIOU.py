import tensorflow as tf
import os,sys
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))
from model.SiamRPN import SiamRPN
from utils.train_utils import show_pred_bbox
from utils.tf_bbox_ops_utils import tf_giou

slim = tf.contrib.slim

class SiamRPN_GIOU(SiamRPN):
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
           
            self.loss_cls = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=valid_labels_flatten, logits=valid_pred_probs))
            self.loss_giou =  tf.reduce_mean(-tf.log(0.5+0.5*tf_giou(tf.reshape(valid_bbox_gts,[-1,4]), tf.reshape(valid_pred_boxes,[-1,4]))))
            
    def build_loss(self):
        self._build_rpn_loss()
        with tf.name_scope('Loss'):
            self.batch_loss = self.loss_cls + self.loss_giou #Normlize 0-2 to be 0-1,the same as loss_iou
            tf.losses.add_loss(self.batch_loss)
            self.total_loss = tf.losses.get_total_loss()
            mean_total_loss, update_op = tf.metrics.mean(self.total_loss)
            with tf.control_dependencies([update_op]):
                tf.summary.scalar('total_loss', mean_total_loss, family=self.mode)

            tf.summary.scalar('batch_loss', self.batch_loss, family=self.mode)
            tf.summary.scalar('loss_cls',   self.loss_cls, family=self.mode)
            tf.summary.scalar('loss_giou',   self.loss_giou, family=self.mode)

            track_instance = tf.py_func(show_pred_bbox,[self.instances, self.topk_bboxes, self.topk_scores, self.gt_instance_boxes],tf.float32)
            tf.summary.image('exemplar', self.examplars, family=self.mode)
            tf.summary.image('instance', track_instance, family=self.mode)
            