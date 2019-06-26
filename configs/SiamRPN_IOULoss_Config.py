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

RUN_NAME = "SiamRPN_IOULoss_stage2"

#LOGFILES
LOG_DIR = "Logs/%s"%(RUN_NAME)
MODEL_CONFIG['checkpoint'] = "%s/Logs/%s/track_model_checkpoints/%s"%(root_dir, RUN_NAME, RUN_NAME)
TRAIN_CONFIG['train_dir'] = osp.join(LOG_DIR, 'track_model_checkpoints', RUN_NAME)
TRACK_CONFIG['log_dir'] = osp.join(LOG_DIR, 'track_model_inference', RUN_NAME)

#Config
TRAIN_CONFIG['train_data_config']['gpu_ids']='2,3'
MODEL_CONFIG['Model'] = "SiamRPN_IOU"

#stage2
MODEL_CONFIG['finetuned_checkpoint_file']=osp.join(root_dir,"Logs/SiamRPN_IOULoss/track_model_checkpoints/SiamRPN_IOULoss")
MODEL_CONFIG['embed_config']['embedding_name']='featureExtract_alexnet'
TRAIN_CONFIG['lr_config']['initial_lr']=1e-4