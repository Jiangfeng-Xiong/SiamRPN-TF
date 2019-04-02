import tensorflow as tf
import os,sys
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

from utils.train_utils import show_pred_bbox
from utils.model_utils import BinWindows
from model.generate_anchors import get_rpn_label
from model.Model import Model
from dataloader.data_augmentation import RandomMixUp


class SiamRPN(Model):
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
