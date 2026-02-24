# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.RESUME = ''
_C.MODEL.PRETRAINED = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True
_C.MODEL.SWIN.SCALE = False
_C.MODEL.SWIN.SHIFT = True

# Swin Transformer V2 parameters
_C.MODEL.SWINV2 = CN()
_C.MODEL.SWINV2.PATCH_SIZE = 4
_C.MODEL.SWINV2.IN_CHANS = 3
_C.MODEL.SWINV2.EMBED_DIM = 96
_C.MODEL.SWINV2.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWINV2.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWINV2.WINDOW_SIZE = 7
_C.MODEL.SWINV2.MLP_RATIO = 4.
_C.MODEL.SWINV2.QKV_BIAS = True
_C.MODEL.SWINV2.APE = False
_C.MODEL.SWINV2.PATCH_NORM = True
_C.MODEL.SWINV2.PRETRAINED_WINDOW_SIZES = [0, 0, 0, 0]
_C.MODEL.SWINV2.SCALE = False
_C.MODEL.SWINV2.SHIFT = True

# Swin MLP parameters
_C.MODEL.SWIN_MLP = CN()
_C.MODEL.SWIN_MLP.PATCH_SIZE = 4
_C.MODEL.SWIN_MLP.IN_CHANS = 3
_C.MODEL.SWIN_MLP.EMBED_DIM = 96
_C.MODEL.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MLP.WINDOW_SIZE = 7
_C.MODEL.SWIN_MLP.MLP_RATIO = 4.
_C.MODEL.SWIN_MLP.APE = False
_C.MODEL.SWIN_MLP.PATCH_NORM = True

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 1
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9
# Teacher model
_C.TRAIN.TEACHER_MODEL = ""
_C.TRAIN.TEACHER_TYPE = "hard"
_C.TRAIN.TEACHER_ALPHA = 0.5
_C.TRAIN.TEACHER_TAU = 3.0
_C.TRAIN.MODEL_EMA = False
_C.TRAIN.MODEL_EMA_DECAY = 0.99996

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Enable Pytorch automatic mixed precision (amp).
_C.AMP_ENABLE = True
# [Deprecated] Mixed precision opt level of apex, if O0, no apex amp is used ('O0', 'O1', 'O2')
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 50
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0

_C.MODE = ''

_C.SUPER = CN()
_C.SUPER.EMBED_DIM = [ 48, 72, 96 ]
_C.SUPER.DEPTHS = [ [ 1, 2 ], [ 1, 2 ], [ 4, 5, 6 ], [ 1, 2 ] ]
_C.SUPER.NUM_HEADS = [ [ 1, 2, 3 ], [ 1, 2, 3, 4, 6 ], [ 1, 2, 3, 4, 6, 8, 12 ], [ 1, 2, 3, 4, 6, 8, 12, 24 ] ]
_C.SUPER.WINDOW_SIZE = [ 3, 4, 5, 7 ]
_C.SUPER.MLP_RATIO = [ 3.5, 4.0 ]

_C.MIN_SUBNET = CN()
_C.MIN_SUBNET.EMBED_DIM = 48
_C.MIN_SUBNET.DEPTHS = [ 1, 1, 4, 1 ]
_C.MIN_SUBNET.NUM_HEADS = [ 1, 1, 1, 1, 1, 1, 1 ]
_C.MIN_SUBNET.WINDOW_SIZE = [ 3, 3, 3, 3, 3, 3, 3 ]
_C.MIN_SUBNET.MLP_RATIO = [ 3.5, 3.5, 3.5, 3.5, 3.5, 3.5, 3.5 ]

_C.MAX_SUBNET = CN()
_C.MAX_SUBNET.EMBED_DIM = 96
_C.MAX_SUBNET.DEPTHS = [ 2, 2, 6, 2 ]
_C.MAX_SUBNET.NUM_HEADS = [ 3, 3, 6, 6, 12, 12, 12, 12, 12, 12, 24, 24 ]
_C.MAX_SUBNET.WINDOW_SIZE = [ 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 ]
_C.MAX_SUBNET.MLP_RATIO = [ 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0 ]

_C.SUBNET = CN()
_C.SUBNET.EMBED_DIM = 96
_C.SUBNET.DEPTHS = [ 2, 2, 6, 2 ]
_C.SUBNET.NUM_HEADS = [ 3, 3, 6, 6, 12, 12, 12, 12, 12, 12, 24, 24 ]
_C.SUBNET.WINDOW_SIZE = [ 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7 ]
_C.SUBNET.MLP_RATIO = [ 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0 ]

# SEARCH
_C.SEARCH = CN()
_C.SEARCH.MAX_EPOCHS = 0
_C.SEARCH.SELECT_NUM = 0
_C.SEARCH.POPULATION_NUM = 0
_C.SEARCH.M_PROB = 0
_C.SEARCH.D_PROB = 0
_C.SEARCH.CROSSOVER_NUM = 0
_C.SEARCH.MUTATION_NUM = 0
_C.SEARCH.PARAM_LIMITS = 0.0
_C.SEARCH.MIN_PARAM_LIMITS = 0.0
_C.SEARCH.FLOPS_LIMITS = 0.0
_C.SEARCH.MIN_FLOPS_LIMITS = 0.0
_C.SEARCH.THROUGHPUT_LIMITS = None
_C.SEARCH.MIN_THROUGHPUT_LIMITS = None
_C.SEARCH.RESUME = ''


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")
        if args.amp_opt_level == 'O0':
            config.AMP_ENABLE = False
    if args.disable_amp:
        config.AMP_ENABLE = False
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    # search
    if hasattr(args, 'max_epochs') and args.max_epochs:
        config.SEARCH.MAX_EPOCHS = args.max_epochs
    if hasattr(args, 'select_num') and args.select_num:
        config.SEARCH.SELECT_NUM = args.select_num
    if hasattr(args, 'population_num') and args.population_num:
        config.SEARCH.POPULATION_NUM = args.population_num
    if hasattr(args, 'm_prob') and args.m_prob:
        config.SEARCH.M_PROB = args.m_prob
    if hasattr(args, 'd_prob') and args.d_prob:
        config.SEARCH.D_PROB = args.d_prob
    if hasattr(args, 'crossover_num') and args.crossover_num:
        config.SEARCH.CROSSOVER_NUM = args.crossover_num
    if hasattr(args, 'mutation_num') and args.mutation_num:
        config.SEARCH.MUTATION_NUM = args.mutation_num
    if hasattr(args, 'param_limits') and args.param_limits:
        config.SEARCH.PARAM_LIMITS = args.param_limits
    if hasattr(args, 'min_param_limits') and args.min_param_limits:
        config.SEARCH.MIN_PARAM_LIMITS = args.min_param_limits
    if hasattr(args, 'flops_limits') and args.flops_limits:
        config.SEARCH.FLOPS_LIMITS = args.flops_limits
    if hasattr(args, 'min_flops_limits') and args.min_flops_limits:
        config.SEARCH.MIN_FLOPS_LIMITS = args.min_flops_limits
    if hasattr(args, 'throughput_limits') and args.throughput_limits:
        config.SEARCH.THROUGHPUT_LIMITS = args.throughput_limits
    if hasattr(args, 'min_throughput_limits') and args.min_throughput_limits:
        config.SEARCH.MIN_THROUGHPUT_LIMITS = args.min_throughput_limits
    if hasattr(args, 'resume') and args.resume:
        config.SEARCH.RESUME = args.resume

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
