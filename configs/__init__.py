import os,sys
CURRENT_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(CURRENT_DIR, '..'))


#Exp Models 
from model.SiamRPN import SiamRPN as SiamRPN
from model.SiamRPN_Base import SiamRPN as SiamRPN_Base
from model.SiamRPNM import SiamRPNM as SiamRPNM
from model.SiamRPN_ml import SiamRPN as SiamRPN_ml
from model.SiamRPN_npairs import SiamRPN as SiamRPN_npairs
from model.SiamRPN_decoder import SiamRPN as SiamRPN_decoder

#Exp Configs
import configs.SiamRPN_bn_bz64_reg10 as SiamRPN_bn_bz64_reg10
import configs.SiamRPN_bn_bz8_reg10_step5e5 as SiamRPN_bn_bz8_reg10_step5e5


import configs.SiamRPN_bn_bz8_reg10 as SiamRPN_bn_bz8_reg10
import configs.SiamRPNM_bn_bz8_reg10 as SiamRPNM_bn_bz8_reg10


import configs.SiamRPN_bn_bz8_reg10_norm_bin as SiamRPN_bn_bz8_reg10_norm_bin
import configs.SiamRPN_bn_bz8_reg10_norm as SiamRPN_bn_bz8_reg10_norm
import configs.SiamRPN_bn_bz64_reg10_lr2 as SiamRPN_bn_bz64_reg10_lr2
import configs.SiamRPN_bn_bz128_reg10_lr2 as SiamRPN_bn_bz128_reg10_lr2
import configs.SiamRPN_Base_config as SiamRPN_Base_config
import configs.SiamRPN_AlexM as SiamRPN_AlexM


"""
import configs.SiamRPN_gn_bz8_reg10 as SiamRPN_gn_bz8_reg10
import configs.SiamRPN_gn_bz64_reg10 as SiamRPN_gn_bz64_reg10
import configs.SiamRPN_bn_bz8_reg10_normalize_tweak as SiamRPN_bn_bz8_reg10_normalize_tweak

#overfit
import configs.SiamRPN_bn_bz64_reg10_ldstep5e5 as SiamRPN_bn_bz64_reg10_ldstep5e5
import configs.SiamRPN_bn_bz8_reg10_ldstep5e5 as SiamRPN_bn_bz8_reg10_ldstep5e5
import configs.SiamRPN_bn_bz8_reg10_ldstep1e5 as SiamRPN_bn_bz8_reg10_ldstep1e5
"""

def get_model(mode_name):
  return {'SiamRPN': SiamRPN,
          'SiamRPN_Base': SiamRPN_Base,
          'SiamRPNM': SiamRPNM,
          'SiamRPN_ml': SiamRPN_ml,
          'SiamRPN_decoder': SiamRPN_decoder,
          'SiamRPN_npairs': SiamRPN_npairs
  }[mode_name]

def get_config(config_name):
  return {
       "SiamRPN_bn_bz64_reg10": SiamRPN_bn_bz64_reg10,
       "SiamRPN_bn_bz8_reg10":SiamRPN_bn_bz8_reg10,
       "SiamRPN_bn_bz8_reg10_norm_bin":SiamRPN_bn_bz8_reg10_norm_bin,
       "SiamRPN_bn_bz8_reg10_norm": SiamRPN_bn_bz8_reg10_norm,
       "SiamRPN_bn_bz8_reg10_step5e5":SiamRPN_bn_bz8_reg10_step5e5,
       "SiamRPNM_bn_bz8_reg10":SiamRPNM_bn_bz8_reg10,
       'SiamRPN_AlexM':SiamRPN_AlexM,
       'SiamRPN_bn_bz64_reg10_lr2':SiamRPN_bn_bz64_reg10_lr2,
       'SiamRPN_bn_bz128_reg10_lr2': SiamRPN_bn_bz128_reg10_lr2,
       'SiamRPN_Base':SiamRPN_Base_config
   }[config_name]
