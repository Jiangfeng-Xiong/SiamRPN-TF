import os,sys
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))


#Exp Models 
from model.SiamRPN import SiamRPN as SiamRPN
from model.SiamRPNM import SiamRPNM as SiamRPNM
from model.SiamRPN_ml import SiamRPNML as SiamRPNML
from model.SiamRPN_npairs import SiamRPNNPairs as SiamRPNNPairs
from model.SiamRPN_decoder import SiamRPNDecoder as SiamRPNDecoder

#Exp Configs
import configs.SiamRPN_BIN as SiamRPN_BIN
import configs.SiamRPN_config as SiamRPN_config
import configs.SiamRPNM_config as SiamRPNM_config
import configs.SiamRPN_pre as SiamRPN_pre
import configs.SiamRPN_pre_lr2 as SiamRPN_pre_lr2
import configs.SiamRPN_scratch as SiamRPN_scratch


def get_model(mode_name):
  return {'SiamRPN': SiamRPN,
          'SiamRPNM': SiamRPNM,
          'SiamRPN_ml': SiamRPNML,
          'SiamRPNDecoder': SiamRPNDecoder,
          'SiamRPN_npairs': SiamRPNNPairs
  }[mode_name]

def get_config(config_name):
  return {
       "SiamRPN_BIN":SiamRPN_BIN,
       'SiamRPN': SiamRPN_config,
       'SiamRPN_pre': SiamRPN_pre,
       'SiamRPN_pre_lr2':SiamRPN_pre_lr2,
       'SiamRPN_scratch': SiamRPN_scratch,
       'SiamRPNM': SiamRPNM_config
   }[config_name]
