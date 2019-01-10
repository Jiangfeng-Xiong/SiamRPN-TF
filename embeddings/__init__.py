from embeddings.convolutional_alexnet import convolutional_alexnet_arg_scope, convolutional_alexnet
from embeddings.convolutional_alexnet_gn import convolutional_alexnet_gn_arg_scope
from embeddings.convolutional_alexnet_m import convolutional_alexnet_m
from embeddings.featureExtract_alexnet import featureExtract_alexnet,featureExtract_alexnet_arg_scope
from embeddings.alexnet_tweak import alexnet_tweak_arg_scope, alexnet_tweak

def get_scope_and_backbone(config, is_training):
    embedding_name = config['embedding_name']
    if embedding_name== 'convolutional_alexnet':
        arg_scope = convolutional_alexnet_arg_scope(config, trainable=config['train_embedding'], is_training=is_training)
        backbone_fn = convolutional_alexnet
    elif embedding_name== 'convolutional_alexnet_gn':
        arg_scope = convolutional_alexnet_gn_arg_scope(config, trainable=config['train_embedding'])
        backbone_fn = convolutional_alexnet
    elif embedding_name == 'convolutional_alexnet_m':
        backbone_fn = convolutional_alexnet_m
        arg_scope = convolutional_alexnet_arg_scope(config, trainable=config['train_embedding'], is_training=is_training)
    elif embedding_name == 'alexnet_tweak':
        arg_scope = alexnet_tweak_arg_scope(config, trainable=config['train_embedding'], is_training=is_training)
        backbone_fn = alexnet_tweak
    elif embedding_name == 'featureExtract_alexnet':
        arg_scope = featureExtract_alexnet_arg_scope(config, trainable=config['train_embedding'], is_training=is_training)
        backbone_fn = featureExtract_alexnet
    else:
        assert("support alexnet only now")
    return arg_scope,backbone_fn