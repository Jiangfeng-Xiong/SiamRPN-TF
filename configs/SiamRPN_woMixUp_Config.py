import os,sys
import os.path as osp

CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

import configs.config as config
import copy 
MODEL_CONFIG = copy.deepcopy(config.MODEL_CONFIG)
TRAIN_CONFIG = copy.deepcopy(config.TRAIN_CONFIG)
TRACK_CONFIG = copy.deepcopy(config.TRACK_CONFIG)
root_dir = config.root_dir

RUN_NAME = "SiamRPN_woMixUp"

#LOGFILES
LOG_DIR = "Logs/%s"%(RUN_NAME)
MODEL_CONFIG['checkpoint'] = "%s/Logs/%s/track_model_checkpoints/%s"%(root_dir, RUN_NAME, RUN_NAME)
TRAIN_CONFIG['train_dir'] = osp.join(LOG_DIR, 'track_model_checkpoints', RUN_NAME)
TRACK_CONFIG['log_dir'] = osp.join(LOG_DIR, 'track_model_inference', RUN_NAME)

#Config
TRAIN_CONFIG['train_data_config']['gpu_ids']='4,5'
TRAIN_CONFIG['train_data_config']['RandomMixUp']=False