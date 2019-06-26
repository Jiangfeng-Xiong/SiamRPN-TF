import os,sys
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))

#Exp Models 
from model.SiamRPN import SiamRPN as SiamRPN
from model.SiamRPN_IOU import SiamRPN_IOU as SiamRPN_IOU
from model.SiamRPN_TRI import SiamRPN_TRI as SiamRPN_TRI


#Exp Configs
import configs.SiamRPN_Config as SiamRPN_Config
import configs.SiamRPN_TRI_Config as SiamRPN_TRI_Config
import configs.SiamRPN_IOULoss_Config as SiamRPN_IOULoss_Config

import configs.SiamRPN_woMixUp_Config as SiamRPN_woMixUp_Config

def get_model(mode_name):
  return {'SiamRPN': SiamRPN,
          'SiamRPN_IOU':SiamRPN_IOU,
          'SiamRPN_TRI': SiamRPN_TRI,
  }[mode_name]

def get_config(config_name):
  return {
       "SiamRPN_Config":SiamRPN_Config,
       'SiamRPN_TRI_Config': SiamRPN_TRI_Config,
       'SiamRPN_IOULoss_Config': SiamRPN_IOULoss_Config,
       'SiamRPN_woMixUp_Config':SiamRPN_woMixUp_Config
   }[config_name]
