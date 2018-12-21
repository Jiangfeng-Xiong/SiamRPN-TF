import os.path as osp

root_dir="/home/lab-xiong.jiangfeng/Projects/SiameseRPN"

RUN_NAME = "None"
Model = "SiamRPN"
batch_size=64
base_lr = 0.1*(batch_size/64.0)
embedding_checkpoint_file=None
adjust_regression=True
LOG_DIR = "Logs/%s"%(RUN_NAME)
regloss_lambda = 10.0

MODEL_CONFIG = {
  'z_image_size': 127,  # Exemplar image size
  'x_image_size': 255, #Instance image size
  'field_size': 17,
  'net_shift': 63,
  'regloss_lambda': regloss_lambda,
  'checkpoint': "%s/Logs/%s/track_model_checkpoints/%s"%(root_dir, RUN_NAME, RUN_NAME),
  'embed_config': {'embedding_name': 'convolutional_alexnet',
                   'embedding_checkpoint_file': embedding_checkpoint_file,
                   'train_embedding': True,
                   'init_method': 'kaiming_normal',
                   'use_bn': True,
                   'bn_scale': True,
                   'bn_momentum': 0.05,
                   'bn_epsilon': 1e-6,
                   'weight_decay': 1e-4,
                   'stride': 8, },
  'Model': Model,
  'adjust_regression': adjust_regression,
  'lr_warmup': False
}

TRAIN_CONFIG = {
  'train_dir': osp.join(LOG_DIR, 'track_model_checkpoints', RUN_NAME),
  'seed': 123,  # fix seed for reproducing experiments
  'npairs_loss_weight': 1e-2,
  'train_data_config': {'input_imdb': 'dataset/TrackingNet_DET2014/train.pickle',
                        'num_examples_per_epoch': 1e6, #
                        'epoch': 50,
                        'batch_size':batch_size,
                        'max_frame_dist': 100,  # Maximum distance between any two random frames draw from videos.
                        'prefetch_threads': 16,
                        'prefetch_capacity': 100 * batch_size,# The maximum elements number in the data loading queue
                        'augmentation_config':{'random_flip': True, 'random_color': True,'random_blur': True}
                        },  
  'validation_data_config': {'input_imdb': 'dataset/TrackingNet_DET2014/validation.pickle',
                             'batch_size': batch_size,
                             'max_frame_dist': 100,  # Maximum distance between any two random frames draw from videos.
                             'prefetch_threads': 16,
                             'prefetch_capacity': 100 * batch_size, },  # The maximum elements number in the data loading queue

  # Optimizer for training the model.
  'optimizer_config': {'optimizer': 'MOMENTUM',  # SGD and MOMENTUM and Adam are supported
                       'momentum': 0.9, #needed by MOMENTUM optimizer 
                       'use_nesterov': True, }, #needed by MOMENTUM optimizer 

  # Learning rate configs
  'lr_config': {'policy': 'exponential',
                'initial_lr': base_lr,
                'num_epochs_per_decay': 1,
                'lr_decay_factor': 0.8685, #0.001^(1/(num_epoch-1))=0.8685 for sgd or 0.1^((1/num_epoch-1_)=0.954 for adam
                'staircase': True, },

  # If not None, clip gradients to this value.
  'clip_gradients': 1.0,

  # Frequency at which loss and global step are logged
  'log_every_n_steps': 10,

  # Frequency to save model
  'save_model_every_n_step': 1e6 // batch_size// 2,  # save model every epoch

  # How many model checkpoints to keep. No limit if None.
  'max_checkpoints_to_keep': None,
}

TRACK_CONFIG = {
  # Directory for saving log files during tracking.
  'log_dir': osp.join(LOG_DIR, 'track_model_inference', RUN_NAME),

  # Logging level of inference, use 1 for detailed inspection. 0 for speed.
  'log_level': 0,

  'x_image_size': 255,  # Search image size during tracking

  'search_scale_smooth_factor': 0.3,
  'penalty_k': 0.22,
  'window_influence': 0.4,
  'include_first': False, # If track the first frame
}
