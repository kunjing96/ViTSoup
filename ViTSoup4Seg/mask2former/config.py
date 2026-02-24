# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_search_config(cfg):
    """
    Add config for Search.
    """
    cfg.SEARCH = CN()
    cfg.SEARCH.MAX_EPOCHS = 20
    cfg.SEARCH.SELECT_NUM = 10
    cfg.SEARCH.POPULATION_NUM = 50
    cfg.SEARCH.M_PROB = 0.2
    cfg.SEARCH.D_PROB = 0.4
    cfg.SEARCH.CROSSOVER_NUM = 25
    cfg.SEARCH.MUTATION_NUM = 25
    cfg.SEARCH.PARAM_LIMITS = 50.0
    cfg.SEARCH.MIN_PARAM_LIMITS = 0.0
    cfg.SEARCH.FLOPS_LIMITS = None # 100.0
    cfg.SEARCH.MIN_FLOPS_LIMITS = None # 0.0
    cfg.SEARCH.THROUGHPUT_LIMITS = None
    cfg.SEARCH.MIN_THROUGHPUT_LIMITS = None
    cfg.SEARCH.RESUME = ''


def add_maskformer2_super_config(cfg):
    """
    Add config for MASK_FORMER_SUPER.
    """
    # NOTE: configs from original maskformer
    # data config
    cfg.INPUT.FORMAT = "RGB"

    # test config
    cfg.TEST.DETECTIONS_PER_IMAGE = 100

    cfg.MODEL.PRETRAIN_IMG_SIZE = 224

    # backbone config
    cfg.MODEL.BACKBONE.PATCH_SIZE = 4
    cfg.MODEL.BACKBONE.EMBED_DIM = 128
    cfg.MODEL.BACKBONE.DEPTHS = [3, 3, 8, 3]
    cfg.MODEL.BACKBONE.NUM_HEADS = [4, 8, 16, 32]
    cfg.MODEL.BACKBONE.WINDOW_SIZE = 7
    cfg.MODEL.BACKBONE.MLP_RATIO = 4.0
    cfg.MODEL.BACKBONE.QKV_BIAS = True
    cfg.MODEL.BACKBONE.QK_SCALE = None
    cfg.MODEL.BACKBONE.DROP_RATE = 0.0
    cfg.MODEL.BACKBONE.ATTN_DROP_RATE = 0.0
    cfg.MODEL.BACKBONE.DROP_PATH_RATE = 0.3
    cfg.MODEL.BACKBONE.ABS_POS = False
    cfg.MODEL.BACKBONE.PATCH_NORM = True
    cfg.MODEL.BACKBONE.USE_CHECKPOINT = False
    cfg.MODEL.BACKBONE.SCALE = False
    cfg.MODEL.BACKBONE.SHIFT = False
    cfg.MODEL.BACKBONE.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    # ^ above config for super-swin
    cfg.MODEL.BACKBONE.ACT_LAYER = "gelu"
    cfg.MODEL.BACKBONE.REL_POS = True
    cfg.MODEL.BACKBONE.MAX_REL_POS = 56
    cfg.MODEL.BACKBONE.PRE_NORM = False
    cfg.MODEL.BACKBONE.NORM = "LN"
    cfg.MODEL.BACKBONE.CHANGE_QKV = True
    # cfg.MODEL.BACKBONE.MERGE_SIZE = 3

    # sem_seg_head config
    cfg.MODEL.SEM_SEG_HEAD.PRE_NORM = False
    cfg.MODEL.SEM_SEG_HEAD.ACT_LAYER = "relu"
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_NORM = "LN"
    cfg.MODEL.SEM_SEG_HEAD.ENC_N_POINTS = 8
    cfg.MODEL.SEM_SEG_HEAD.DROPOUT = 0.0
    cfg.MODEL.SEM_SEG_HEAD.ENC_N_HEADS = 12
    cfg.MODEL.SEM_SEG_HEAD.ENC_MLP_RATIO = 4.0
    cfg.MODEL.SEM_SEG_HEAD.ENC_DEPTHS = 8
    cfg.MODEL.SEM_SEG_HEAD.ENC_IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.SCALE = False
    cfg.MODEL.SEM_SEG_HEAD.USE_CHECKPOINT = False
    # ^ above config for super-sem_seg_head

    # mask_former config
    cfg.MODEL.MASK_FORMER.DEC_IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.MASK_FORMER.DEC_N_HEADS = 12
    cfg.MODEL.MASK_FORMER.DEC_MLP_RATIO = 8.0
    cfg.MODEL.MASK_FORMER.DEC_DEPTHS = 12
    cfg.MODEL.MASK_FORMER.ACT_LAYER = "relu"
    cfg.MODEL.MASK_FORMER.NORM = "LN"
    cfg.MODEL.MASK_FORMER.SCALE = False
    cfg.MODEL.MASK_FORMER.USE_CHECKPOINT = False
    # ^ above config for super-mask_former
    cfg.MODEL.MASK_FORMER.QKV_BIAS = False
    cfg.MODEL.MASK_FORMER.QK_SCALE = None
    cfg.MODEL.MASK_FORMER.CHANGE_QKV = True

    # forward mode config
    cfg.MODE = ""

    # search_space config
    cfg.SEARCH_SPACE = CN()
    cfg.SEARCH_SPACE.BACKBONE = CN()
    cfg.SEARCH_SPACE.BACKBONE.IS_USED = True
    cfg.SEARCH_SPACE.BACKBONE.EMBED_DIM = [64, 96, 128]
    cfg.SEARCH_SPACE.BACKBONE.DEPTHS = [ [ 1, 2 ], [ 1, 2 ], [ 4, 5, 6 ], [ 1, 2 ] ]
    cfg.SEARCH_SPACE.BACKBONE.NUM_HEADS = [ [ 1, 2, 3 ], [ 1, 2, 3, 4, 6 ], [ 1, 2, 3, 4, 6, 8, 12 ], [ 1, 2, 3, 4, 6, 8, 12, 24 ] ]
    cfg.SEARCH_SPACE.BACKBONE.WINDOW_SIZE = [ 3, 4, 5, 7 ]
    cfg.SEARCH_SPACE.BACKBONE.MLP_RATIO = [3.5, 4.0]
    # cfg.SEARCH_SPACE.BACKBONE.MERGE_SIZE = [2, 3]
    cfg.SEARCH_SPACE.SEM_SEG_HEAD = CN()
    cfg.SEARCH_SPACE.SEM_SEG_HEAD.IS_USED = True
    cfg.SEARCH_SPACE.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.SEARCH_SPACE.SEM_SEG_HEAD.ENC_IN_FEATURES = [["res2", "res3", "res4", "res5"], ["res3", "res4", "res5"], ["res4", "res5"], ["res5"]]
    cfg.SEARCH_SPACE.SEM_SEG_HEAD.CONVS_DIM = [128, 256, 384]
    cfg.SEARCH_SPACE.SEM_SEG_HEAD.MASK_DIM = [128, 256, 384]
    cfg.SEARCH_SPACE.SEM_SEG_HEAD.ENC_DEPTHS = [4, 6, 8]
    cfg.SEARCH_SPACE.SEM_SEG_HEAD.ENC_N_POINTS = [4, 6, 8]
    cfg.SEARCH_SPACE.SEM_SEG_HEAD.ENC_N_HEADS = [4, 8 ,12]
    cfg.SEARCH_SPACE.SEM_SEG_HEAD.ENC_MLP_RATIO = [3.5, 4.0]
    cfg.SEARCH_SPACE.MASK_FORMER = CN()
    cfg.SEARCH_SPACE.MASK_FORMER.IS_USED = True
    cfg.SEARCH_SPACE.MASK_FORMER.DEC_IN_FEATURES = [["res2", "res3", "res4", "res5"], ["res2", "res3", "res4"], ["res2", "res3"], ["res2"]]
    cfg.SEARCH_SPACE.MASK_FORMER.HIDDEN_DIM = [128, 256, 384]
    cfg.SEARCH_SPACE.MASK_FORMER.DEC_DEPTHS = [8, 10, 12]
    # cfg.SEARCH_SPACE.MASK_FORMER.DEC_MASK_RATIO = [0, 0.5, 0.9, 0.99, 0.999]
    cfg.SEARCH_SPACE.MASK_FORMER.DEC_N_HEADS = [4, 8, 12]
    cfg.SEARCH_SPACE.MASK_FORMER.DEC_MLP_RATIO = [7.0, 8.0]

    # subnet config
    cfg.SUBNET = CN()
    cfg.SUBNET.BACKBONE = CN()
    cfg.SUBNET.BACKBONE.EMBED_DIM = 64
    cfg.SUBNET.BACKBONE.DEPTHS = [1, 1, 4, 1]
    cfg.SUBNET.BACKBONE.NUM_HEADS = [2, 4, 8, 8, 8, 8, 16]
    # cfg.SUBNET.BACKBONE.MASK_RATIO = [0, 0, 0, 0, 0, 0, 0]
    cfg.SUBNET.BACKBONE.WINDOW_SIZE = [7, 7, 7, 7, 7, 7, 7]
    cfg.SUBNET.BACKBONE.MLP_RATIO = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    # cfg.SUBNET.BACKBONE.MERGE_SIZE = [2, 2, 2, None]
    cfg.SUBNET.SEM_SEG_HEAD = CN()
    cfg.SUBNET.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.SUBNET.SEM_SEG_HEAD.ENC_IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.SUBNET.SEM_SEG_HEAD.CONVS_DIM = 128
    cfg.SUBNET.SEM_SEG_HEAD.MASK_DIM = 128
    cfg.SUBNET.SEM_SEG_HEAD.ENC_DEPTHS = 4
    cfg.SUBNET.SEM_SEG_HEAD.ENC_N_POINTS = [4, 4, 4, 4]
    cfg.SUBNET.SEM_SEG_HEAD.ENC_N_HEADS = [4, 4, 4, 4]
    cfg.SUBNET.SEM_SEG_HEAD.ENC_MLP_RATIO = [4.0, 4.0, 4.0, 4.0]
    cfg.SUBNET.MASK_FORMER = CN()
    cfg.SUBNET.MASK_FORMER.DEC_IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.SUBNET.MASK_FORMER.HIDDEN_DIM = 128
    cfg.SUBNET.MASK_FORMER.DEC_DEPTHS = 8
    cfg.SUBNET.MASK_FORMER.DEC_CROSS_N_HEADS = [4, 4, 4, 4, 4, 4, 4, 4]
    # cfg.SUBNET.MASK_FORMER.DEC_SELF_MASK_RATIO = [0, 0, 0, 0, 0, 0, 0, 0]
    cfg.SUBNET.MASK_FORMER.DEC_SELF_N_HEADS = [4, 4, 4, 4, 4, 4, 4, 4]
    cfg.SUBNET.MASK_FORMER.DEC_FFN_MLP_RATIO = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
    cfg.SUBNET.MASK_FORMER.MULTI_SCALE_PER_LAYER = ['res2', 'res2', 'res2', 'res2', 'res2', 'res2', 'res2', 'res2']

    # min subnet config
    cfg.MIN_SUBNET = CN()
    cfg.MIN_SUBNET.BACKBONE = CN()
    cfg.MIN_SUBNET.BACKBONE.EMBED_DIM = 64
    cfg.MIN_SUBNET.BACKBONE.DEPTHS = [1, 1, 4, 1]
    cfg.MIN_SUBNET.BACKBONE.NUM_HEADS = [2, 4, 8, 8, 8, 8, 16]
    # cfg.MIN_SUBNET.BACKBONE.MASK_RATIO = [0, 0, 0, 0, 0, 0, 0]
    cfg.MIN_SUBNET.BACKBONE.WINDOW_SIZE = [7, 7, 7, 7, 7, 7, 7]
    cfg.MIN_SUBNET.BACKBONE.MLP_RATIO = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    # cfg.MIN_SUBNET.BACKBONE.MERGE_SIZE = [2, 2, 2, None]
    cfg.MIN_SUBNET.SEM_SEG_HEAD = CN()
    cfg.MIN_SUBNET.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MIN_SUBNET.SEM_SEG_HEAD.ENC_IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MIN_SUBNET.SEM_SEG_HEAD.CONVS_DIM = 128
    cfg.MIN_SUBNET.SEM_SEG_HEAD.MASK_DIM = 128
    cfg.MIN_SUBNET.SEM_SEG_HEAD.ENC_DEPTHS = 4
    cfg.MIN_SUBNET.SEM_SEG_HEAD.ENC_N_POINTS = [4, 4, 4, 4]
    cfg.MIN_SUBNET.SEM_SEG_HEAD.ENC_N_HEADS = [4, 4, 4, 4]
    cfg.MIN_SUBNET.SEM_SEG_HEAD.ENC_MLP_RATIO = [4.0, 4.0, 4.0, 4.0]
    cfg.MIN_SUBNET.MASK_FORMER = CN()
    cfg.MIN_SUBNET.MASK_FORMER.DEC_IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MIN_SUBNET.MASK_FORMER.HIDDEN_DIM = 128
    cfg.MIN_SUBNET.MASK_FORMER.DEC_DEPTHS = 8
    cfg.MIN_SUBNET.MASK_FORMER.DEC_CROSS_N_HEADS = [4, 4, 4, 4, 4, 4, 4, 4]
    # cfg.MIN_SUBNET.MASK_FORMER.DEC_SELF_MASK_RATIO = [0, 0, 0, 0, 0, 0, 0, 0]
    cfg.MIN_SUBNET.MASK_FORMER.DEC_SELF_N_HEADS = [4, 4, 4, 4, 4, 4, 4, 4]
    cfg.MIN_SUBNET.MASK_FORMER.DEC_FFN_MLP_RATIO = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
    cfg.MIN_SUBNET.MASK_FORMER.MULTI_SCALE_PER_LAYER = ['res2', 'res2', 'res2', 'res2', 'res2', 'res2', 'res2', 'res2']

    # subnet config
    cfg.MAX_SUBNET = CN()
    cfg.MAX_SUBNET.BACKBONE = CN()
    cfg.MAX_SUBNET.BACKBONE.EMBED_DIM = 64
    cfg.MAX_SUBNET.BACKBONE.DEPTHS = [1, 1, 4, 1]
    cfg.MAX_SUBNET.BACKBONE.NUM_HEADS = [2, 4, 8, 8, 8, 8, 16]
    # cfg.MAX_SUBNET.BACKBONE.MASK_RATIO = [0, 0, 0, 0, 0, 0, 0]
    cfg.MAX_SUBNET.BACKBONE.WINDOW_SIZE = [7, 7, 7, 7, 7, 7, 7]
    cfg.MAX_SUBNET.BACKBONE.MLP_RATIO = [4.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
    # cfg.MAX_SUBNET.BACKBONE.MERGE_SIZE = [2, 2, 2, None]
    cfg.MAX_SUBNET.SEM_SEG_HEAD = CN()
    cfg.MAX_SUBNET.SEM_SEG_HEAD.IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MAX_SUBNET.SEM_SEG_HEAD.ENC_IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MAX_SUBNET.SEM_SEG_HEAD.CONVS_DIM = 128
    cfg.MAX_SUBNET.SEM_SEG_HEAD.MASK_DIM = 128
    cfg.MAX_SUBNET.SEM_SEG_HEAD.ENC_DEPTHS = 4
    cfg.MAX_SUBNET.SEM_SEG_HEAD.ENC_N_POINTS = [4, 4, 4, 4]
    cfg.MAX_SUBNET.SEM_SEG_HEAD.ENC_N_HEADS = [4, 4, 4, 4]
    cfg.MAX_SUBNET.SEM_SEG_HEAD.ENC_MLP_RATIO = [4.0, 4.0, 4.0, 4.0]
    cfg.MAX_SUBNET.MASK_FORMER = CN()
    cfg.MAX_SUBNET.MASK_FORMER.DEC_IN_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MAX_SUBNET.MASK_FORMER.HIDDEN_DIM = 128
    cfg.MAX_SUBNET.MASK_FORMER.DEC_DEPTHS = 8
    cfg.MAX_SUBNET.MASK_FORMER.DEC_CROSS_N_HEADS = [4, 4, 4, 4, 4, 4, 4, 4]
    # cfg.MAX_SUBNET.MASK_FORMER.DEC_SELF_MASK_RATIO = [0, 0, 0, 0, 0, 0, 0, 0]
    cfg.MAX_SUBNET.MASK_FORMER.DEC_SELF_N_HEADS = [4, 4, 4, 4, 4, 4, 4, 4]
    cfg.MAX_SUBNET.MASK_FORMER.DEC_FFN_MLP_RATIO = [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]
    cfg.MAX_SUBNET.MASK_FORMER.MULTI_SCALE_PER_LAYER = ['res2', 'res2', 'res2', 'res2', 'res2', 'res2', 'res2', 'res2']


def add_maskformer2_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    # NOTE: configs from original maskformer
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # Sometimes `backbone.size_divisibility` is set to 0 for some backbone (e.g. ResNet)
    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75
