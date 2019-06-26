import os.path as osp

root_dir="/home/lab-xiong.jiangfeng/Projects/SiameseRPN"

RUN_NAME = "None"
Model = "SiamRPN"
batch_size=128
base_lr = 0.01
#embedding_checkpoint_file="/home/lab-xiong.jiangfeng/Projects/SiameseRPN/embeddings/pytorch_weights/siamrpn_model.pkl"
embedding_checkpoint_file=None
LOG_DIR = "Logs/%s"%(RUN_NAME)
regloss_lambda = 10.0
time_decay = True
RandomMixUp=False

MODEL_CONFIG = {
  'z_image_size': 127,  # Exemplar image size
  'x_image_size': 255, #Instance image size
  'field_size': 17,
  'net_shift': 63,
  'regloss_lambda': regloss_lambda,
  'finetuned_checkpoint_file': osp.join(root_dir,"Logs/SiamRPN_fixedEmbedding/track_model_checkpoints/SiamRPN_fixedEmbedding"),
  'checkpoint': "%s/Logs/%s/track_model_checkpoints/%s"%(root_dir, RUN_NAME, RUN_NAME),
  'embed_config': {'embedding_name': 'featureExtract_alexnet_fixedconv3',
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
}

TRAIN_CONFIG = {
  'train_dir': osp.join(LOG_DIR, 'track_model_checkpoints', RUN_NAME),
  'seed': 123,  # fix seed for reproducing experiments
  'train_data_config': {'input_imdb': 'dataset/YB/train.pickle',
						#'lmdb_path': 'dataset/YB/train_lmdb_encode',
						'lmdb_encode': True,
                        'time_decay':time_decay,
                        'num_examples_per_epoch': 2e5, #
                        'epoch': 200,
                        'batch_size':batch_size,
                        'max_frame_dist': 1000,  # Maximum distance between any two random frames draw from videos.
                        'prefetch_threads': 16,
                        'prefetch_capacity': 10 * batch_size,# The maximum elements number in the data loading queue
                        'augmentation_config':{'random_flip': True, 'random_color': True,'random_blur': True, 'random_downsample': False},
                        'RandomMixUp': RandomMixUp
                        },  
  'validation_data_config': {'input_imdb': 'dataset/YB/train.pickle',
							#'lmdb_path': 'dataset/YB/validation_lmdb_encode',
							'lmdb_encode': True,
                            'time_decay': False,
                             'batch_size': 64,
                             'max_frame_dist': 100,  # Maximum distance between any two random frames draw from videos.
                             'prefetch_threads': 8,
                             'prefetch_capacity': 64, },  # The maximum elements number in the data loading queue

  # Optimizer for training the model.
  'optimizer_config': {'optimizer': 'MOMENTUM',  # SGD and MOMENTUM and Adam are supported
                       'momentum': 0.9, #needed by MOMENTUM optimizer 
                       'use_nesterov': True, }, #needed by MOMENTUM optimizer 

  # Learning rate configs
  'lr_config': {'policy': 'exponential',
                'initial_lr': base_lr,
                'num_epochs_per_decay': 1,
                'lr_decay_factor': 0.1**(1.0/100.0), #0.001^(1/(num_epoch-1))=0.8685 for sgd or 0.1^((1/num_epoch-1_)=0.954 for adam
                'lr_warmup': False,
                'staircase': True, },

  # If not None, clip gradients to this value.
  'clip_gradients': 1.0,

  # Frequency at which loss and global step are logged
  'log_every_n_steps': 10,

  # Frequency to save model
  'save_model_every_n_step': 5000,  # save model every epoch

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
  'window_influence': 0.2,
  'include_first': False, # If track the first frame
}
