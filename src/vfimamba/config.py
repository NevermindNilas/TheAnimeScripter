from functools import partial
import torch.nn as nn

from model import mamba_extractor
from model import mamba_estimation

LOG = 'VFIMamba_S'
LOCAL = 2

'''==========Model config=========='''
def init_model_config(F=32, W=7, depth=[2, 2, 2, 4, 4], M=False):
    '''This function should not be modified'''
    return { 
        'embed_dims':[(2**i)*F for i in range(len(depth))],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'num_heads':[8*(2**i)*F//32 for i in range(len(depth)-3)],
        'mlp_ratios':[4 for i in range(len(depth)-3)],
        'qkv_bias':True,
        'norm_layer':partial(nn.LayerNorm, eps=1e-6), 
        'depths':depth,
        'window_sizes':[W for i in range(len(depth)-3)],
        'conv_stages':3
    }, {
        'embed_dims':[(2**i)*F for i in range(len(depth))],
        'motion_dims':[0, 0, 0, 8*F//depth[-2], 16*F//depth[-1]],
        'depths':depth,
        'num_heads':[8*(2**i)*F//32 for i in range(len(depth)-3)],
        'window_sizes':[W, W],
        'scales':[4*(2**i) for i in range(len(depth)-2)],
        'hidden_dims':[4*F for i in range(len(depth)-3)],
        'c':F,
        'M':M,
        'local_hidden_dims':4*F,
        'local_num':2
    }


MODEL_CONFIG = {
    'LOGNAME': LOG,
    'MODEL_TYPE': (mamba_extractor, mamba_estimation),
    'MODEL_ARCH': init_model_config(
        F = 16,
        depth = [2, 2, 2, 3, 3],
        M = False
    )
}
