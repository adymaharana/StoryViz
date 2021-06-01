from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
from easydict import EasyDict as edict


__C = edict()
cfg = __C


__C.DATASET_NAME = 'pororo'
__C.EMBEDDING_TYPE = 'cnn-rnn'
__C.CONFIG_NAME = ''
__C.GPU_ID = '0'
__C.CUDA = True
__C.WORKERS = 6
__C.VIDEO_LEN = 5
__C.NET_G = ''
__C.NET_D = ''
__C.STAGE1_G = ''
__C.DATA_DIR = ''
__C.VIS_COUNT = 64

__C.Z_DIM = 100
__C.IMSIZE = 64
__C.STAGE = 1

__C.INITIALIZER_RANGE=0.02

__C.LABEL_NUM = 10
# Training options
__C.TRAIN = edict()
__C.TRAIN.FLAG = True
__C.TRAIN.IM_BATCH_SIZE = 64
__C.TRAIN.ST_BATCH_SIZE = 64
__C.TRAIN.MAX_EPOCH = 600
__C.TRAIN.SNAPSHOT_INTERVAL = 50
__C.TRAIN.PRETRAINED_MODEL = ''
__C.TRAIN.PRETRAINED_EPOCH = 600
__C.TRAIN.LR_DECAY_EPOCH = 600
__C.TRAIN.DISCRIMINATOR_LR = 2e-4
__C.TRAIN.GENERATOR_LR = 2e-4
__C.TRAIN.DAMSM_LR = 2e-4
__C.TRAIN.PERCEPTUAL_LOSS = False
__C.TRAIN.NET_E = ''

__C.TRAIN.SMOOTH = edict()
__C.TRAIN.SMOOTH.GAMMA1 = 5.0
__C.TRAIN.SMOOTH.GAMMA2 = 5.0
__C.TRAIN.SMOOTH.GAMMA3 = 10.0
__C.TRAIN.SMOOTH.GAMMA4 = 15.0
__C.TRAIN.RNN_GRAD_CLIP = 0.25

__C.TRAIN.COEFF = edict()
__C.TRAIN.COEFF.KL = 2.0
__C.TRAIN.UPDATE_RATIO = 2

# Modal options
__C.GAN = edict()
__C.GAN.CONDITION_DIM = 124
__C.GAN.Z_DIM = 100
__C.GAN.DF_DIM = 64
__C.GAN.GF_DIM = 128
__C.GAN.R_NUM = 4

__C.TEXT = edict()
__C.TEXT.DIMENSION = 128

# new options
__C.IMG_DUAL = False
__C.STORY_DUAL = False
__C.USE_MART = False
__C.IMG_DISC = False
__C.STORY_DISC = False
__C.RECURRENT = False
__C.USE_TRANSFORMER = False
__C.TWO_STG = False
__C.TWO_STG_ATTN = "stream"
__C.COPY_TRANSFORM = False

__C.JOINT_EMB_DIM = 256

__C.EMBED_SIZE = 300
__C.HIDDEN_SIZE = 512

__C.MART = edict()
__C.MART.hidden_size = 192
__C.MART.intermediate_size = 192
__C.MART.max_t_len = 30
__C.MART.n_memory_cells = 3
__C.MART.layer_norm_eps = 1e-12
__C.MART.hidden_dropout_prob = 0.1
__C.MART.num_hidden_layers = 2
__C.MART.attention_probs_dropout_prob = 0.1
__C.MART.num_attention_heads = 6
__C.MART.memory_dropout_prob = 0.1
__C.MART.pretrained_embeddings = './data/glove.840B.300d.txt'
__C.MART.freeze_embeddings = False
__C.MART.initializer_range = 0.2
__C.MART.vocab_size = 0
__C.MART.word_vec_size = 300
__C.MART.raw_glove_path = './data/glove.840B.300d.txt'
__C.MART.vocab_glove_path = ''
__C.MART.max_position_embeddings = ''
__C.MART.CKPT_PATH = ''

# Video Captioning configs
__C.MART.cls_word = '[CLS]'
__C.MART.vid_word = '[VID]'
__C.MART.sep_word = '[SEP]'

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b.keys():
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) '
                                  'for config key: {}').format(type(b[k]),
                                                               type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)
