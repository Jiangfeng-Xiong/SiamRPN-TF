import torch
from net import SiamRPNotb

net = SiamRPNotb()
net.load_state_dict(torch.load('SiamRPNOTB.model'))

dict={}
conv_layer_index=1
#gamma is scale, and beata is bias
for layer in net.featureExtract:
  if isinstance(layer,torch.nn.modules.conv.Conv2d ):
    dict['conv%d/weights'%(conv_layer_index)] = layer.weight.data.numpy().transpose(2,3,1,0)
    dict['conv%d/biases'%(conv_layer_index)] = layer.bias.data.numpy()
  elif isinstance(layer,torch.nn.modules.batchnorm.BatchNorm2d):
    dict['bn%d/moving_mean'%(conv_layer_index)] = layer.running_mean.numpy()
    dict['bn%d/moving_variance'%(conv_layer_index)] = layer.running_var.numpy()
    dict['bn%d/gamma'%(conv_layer_index)] = layer.weight.data.numpy()
    dict['bn%d/beta'%(conv_layer_index)] = layer.bias.data.numpy()
    conv_layer_index = conv_layer_index + 1
  else:
    print("Tere are no parameters in layer: ", layer)
  
dict['conv_r1/weights'] = net.conv_r1.weight.data.numpy().transpose(2,3,1,0)
dict['conv_r1/biases'] = net.conv_r1.bias.data.numpy()

dict['conv_r2/weights'] = net.conv_r2.weight.data.numpy().transpose(2,3,1,0)
dict['conv_r2/biases'] = net.conv_r2.bias.data.numpy()

dict['conv_cls1/weights'] = net.conv_cls1.weight.data.numpy().transpose(2,3,1,0)
dict['conv_cls1/biases'] = net.conv_cls1.bias.data.numpy()

dict['conv_cls2/weights'] = net.conv_cls2.weight.data.numpy().transpose(2,3,1,0)
dict['conv_cls2/biases'] = net.conv_cls2.bias.data.numpy()

dict['regress_adjust/weights'] = net.regress_adjust.weight.data.numpy().transpose(2,3,1,0)
dict['regress_adjust/biases'] = net.regress_adjust.bias.data.numpy()

import pickle
f = open('siamrpn_model.pkl', "wb")
pickle.dump(dict, f)
f.close()

import numpy as np
for k in dict:
    print(k, np.shape(dict[k]))