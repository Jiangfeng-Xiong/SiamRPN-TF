import logging
import os
import sys
import os.path as osp
import random
import time
from datetime import datetime

import numpy as np
import tensorflow as tf
from sacred import Experiment
from sacred.observers import FileStorageObserver
from utils.misc_utils import auto_select_gpu, mkdir_p, save_cfgs
from utils.train_utils import print_trainable, average_gradients
from configs import get_model,get_config

from dataloader.DataLoader import DataLoader


Debug=False

config_name = input("Input config name : ")

config = get_config(config_name)
ex = Experiment(config.RUN_NAME)
ex.observers.append(FileStorageObserver.create(osp.join(config.LOG_DIR, 'sacred')))

@ex.config
def configurations():
  model_config = config.MODEL_CONFIG
  train_config = config.TRAIN_CONFIG
  track_config = config.TRACK_CONFIG

def _configure_learning_rate(train_config, global_step):
  lr_config = train_config['lr_config']

  num_batches_per_epoch = \
    int(train_config['train_data_config']['num_examples_per_epoch'] / train_config['train_data_config']['batch_size'])

  lr_policy = lr_config['policy']
  if lr_policy == 'piecewise_constant':
    lr_boundaries = [int(e * num_batches_per_epoch) for e in lr_config['lr_boundaries']]
    return tf.train.piecewise_constant(global_step,
                                       lr_boundaries,
                                       lr_config['lr_values'])
  elif lr_policy == 'exponential':
    decay_steps = int(num_batches_per_epoch) * lr_config['num_epochs_per_decay']
    return tf.train.exponential_decay(lr_config['initial_lr'],
                                      global_step,
                                      decay_steps=decay_steps,
                                      decay_rate=lr_config['lr_decay_factor'],
                                      staircase=lr_config['staircase'])
  elif lr_policy == 'cosine':
    T_total = train_config['train_data_config']['epoch'] * num_batches_per_epoch
    return 0.5 * lr_config['initial_lr'] * (1 + tf.cos(np.pi * tf.to_float(global_step) / T_total))
  else:
    raise ValueError('Learning rate policy [%s] was not recognized', lr_policy)


def _configure_optimizer(train_config, learning_rate):
  optimizer_config = train_config['optimizer_config']
  optimizer_name = optimizer_config['optimizer'].upper()
  if optimizer_name == 'MOMENTUM':
    optimizer = tf.train.MomentumOptimizer(
      learning_rate,
      momentum=optimizer_config['momentum'],
      use_nesterov=optimizer_config['use_nesterov'],
      name='Momentum')
  elif optimizer_name == 'SGD':
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer_name == 'ADAM':
    optimizer = tf.train.AdamOptimizer(learning_rate,epsilon=0.1)
  else:
    raise ValueError('Optimizer [%s] was not recognized', optimizer_config['optimizer'])
  return optimizer

def tower_model(Model, inputs, model_config, train_config, mode='train'):
    model = Model(model_config, train_config, mode=mode, inputs=inputs)
    model.build()
    return model

@ex.automain
def main(model_config, train_config, track_config):

  # GPU Config 
  gpu_list = train_config['train_data_config'].get('gpu_ids','0')
  num_gpus = len(gpu_list.split(','))
  if num_gpus>1:
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
  else:
    os.environ['CUDA_VISIBLE_DEVICES'] = auto_select_gpu()

  # Create training directory which will be used to save: configurations, model files, TensorBoard logs
  train_dir = train_config['train_dir']
  if not osp.isdir(train_dir):
    logging.info('Creating training directory: %s', train_dir)
    mkdir_p(train_dir)

  g = tf.Graph()
  with g.as_default():
    # Set fixed seed for reproducible experiments
    random.seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    tf.set_random_seed(train_config['seed'])

    #Build global step
    with tf.name_scope('train/'):
        global_step = tf.Variable(initial_value=0,name='global_step', trainable=False,
          collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES])    
    
    Model = get_model(model_config['Model'])

    # build training dataloader and validation dataloader
    #---train
    train_dataloader = DataLoader(train_config['train_data_config'], is_training=True)
    train_dataloader.build()
    train_inputs = train_dataloader.get_one_batch()

    #---validation
    val_dataloader = DataLoader(train_config['validation_data_config'], is_training=False)
    val_dataloader.build()
    val_inputs = val_dataloader.get_one_batch()
    
    # Save configurations for future reference
    save_cfgs(train_dir, model_config, train_config, track_config)

    if train_config['lr_config'].get('lr_warmup', False):
      warmup_epoch_num = 10
      init_lr_ratio = 0.8
      warmup_steps = warmup_epoch_num * int(train_config['train_data_config']['num_examples_per_epoch'])//train_config['train_data_config']['batch_size']
      inc_per_step = (1-init_lr_ratio)*train_config['lr_config']['initial_lr']/warmup_steps
      warmup_lr = train_config['lr_config']['initial_lr']*init_lr_ratio + inc_per_step*tf.to_float(global_step)
      learning_rate = tf.cond(tf.less(global_step, warmup_steps),lambda: tf.identity(warmup_lr), lambda: _configure_learning_rate(train_config, global_step-warmup_steps))
    else:
      learning_rate = _configure_learning_rate(train_config, global_step)

    optimizer = _configure_optimizer(train_config, learning_rate)
    tf.summary.scalar('learning_rate', learning_rate)

    # Set up the training ops
    examplars, instances, gt_examplar_boxes, gt_instance_boxes = tf.split(train_inputs[0],num_gpus), \
                                                                 tf.split(train_inputs[1],num_gpus), \
                                                                 tf.split(train_inputs[2],num_gpus), \
                                                                 tf.split(train_inputs[3],num_gpus)
    tower_grads = []
    with tf.variable_scope(tf.get_variable_scope()):
      for i in range(num_gpus):
        with tf.device('/gpu:%d' % i):
          inputs = [examplars[i], instances[i], gt_examplar_boxes[i], gt_instance_boxes[i]]
          model = tower_model(Model, inputs, model_config, train_config, mode='train')
          # Reuse variables for the next tower.
          tf.get_variable_scope().reuse_variables()
          grads = optimizer.compute_gradients(model.total_loss)
          tower_grads.append(grads)
    grads = average_gradients(tower_grads)
    
    #Clip gradient
    gradients, tvars = zip(*grads)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, train_config['clip_gradients'])
    train_op = optimizer.apply_gradients(zip(clip_gradients, tvars),global_step=global_step)
    
    #Build validation model
    with tf.device('/gpu:0'): 
        model_va = Model(model_config, train_config, mode='validation', inputs=val_inputs)
        model_va.build(reuse=True)
    
    #Save Model setup
    saver = tf.train.Saver(tf.global_variables(),
                           max_to_keep=train_config['max_checkpoints_to_keep'])

    summary_writer = tf.summary.FileWriter(train_dir, g)
    summary_op = tf.summary.merge_all()

    global_variables_init_op = tf.global_variables_initializer()
    local_variables_init_op = tf.local_variables_initializer()
    

    # Dynamically allocate GPU memory
    gpu_options = tf.GPUOptions(allow_growth=True)
    sess_config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    #inter_op_parallelism_threads = 16, intra_op_parallelism_threads = 16, log_device_placement=True)

    ######Debug timeline
    if Debug:
        from tensorflow.python.client import timeline
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
    ######Debug timeline
    
    sess = tf.Session(config=sess_config)
    model_path = tf.train.latest_checkpoint(train_config['train_dir'])
    
    if not model_path:
      sess.run(global_variables_init_op)
      sess.run(local_variables_init_op)
      start_step = 0
      if model_config['embed_config']['embedding_checkpoint_file']:
        model.init_fn(sess)
    else:
      logging.info('Restore from last checkpoint: {}'.format(model_path))
      sess.run(local_variables_init_op)
      sess.run(global_variables_init_op)
      #saver.restore(sess, model_path)
      restore_op = tf.contrib.slim.assign_from_checkpoint_fn(model_path, tf.global_variables(), ignore_missing_vars=True)
      restore_op(sess)
      start_step = tf.train.global_step(sess, global_step.name) + 1

    print_trainable(sess) #help function, can be disenable
    g.finalize()  # Finalize graph to avoid adding ops by mistake

    # Training loop
    data_config = train_config['train_data_config']
    total_steps = int(data_config['epoch'] *
                      data_config['num_examples_per_epoch'] /
                      data_config['batch_size'])
    logging.info('Train for {} steps'.format(total_steps))
    for step in range(start_step, total_steps):
      try: 
        start_time = time.time()
        if Debug:
            _, loss, batch_loss, current_lr= sess.run([train_op, model.total_loss, model.batch_loss, learning_rate],run_metadata=run_metadata,options=run_options)
            t1 = timeline.Timeline(run_metadata.step_stats)
            ctf = t1.generate_chrome_trace_format()
            with open('timeline.json','w') as f:
                f.write(ctf)
        else:
            _, loss, batch_loss, current_lr= sess.run([train_op, model.total_loss, model.batch_loss, learning_rate])
        duration = time.time() - start_time
        
        if step % 10 == 0:
          examples_per_sec = data_config['batch_size'] / float(duration)
          time_remain = data_config['batch_size'] * (total_steps - step) / examples_per_sec
          current_epoch = (step*data_config['batch_size'])//data_config['num_examples_per_epoch'] + 1
          m, s = divmod(time_remain, 60)
          h, m = divmod(m, 60)
          format_str = ('%s: epoch %d-step %d,lr = %f, total loss = %.3f, batch loss = %.3f (%.1f examples/sec; %.3f '
                        'sec/batch; %dh:%02dm:%02ds remains)')
          logging.info(format_str % (datetime.now(), current_epoch, step, current_lr, loss, batch_loss,
                                     examples_per_sec, duration, h, m, s))

        if step % 200 == 0:
          summary_str = sess.run(summary_op)
          summary_writer.add_summary(summary_str, step)

        if step % train_config['save_model_every_n_step'] == 0 or (step + 1) == total_steps:
          checkpoint_path = osp.join(train_config['train_dir'], 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)
      except KeyboardInterrupt:
          checkpoint_path = osp.join(train_config['train_dir'], 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)
          print("save model.ckpt-%d"%(step))
          break
      except:
        print("Error found in current step, continue")
