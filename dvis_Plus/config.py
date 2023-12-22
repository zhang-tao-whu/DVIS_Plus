# -*- coding: utf-8 -*-
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_minvis_config(cfg):
    cfg.INPUT.SAMPLING_FRAME_RATIO = 1.0
    cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE = False

    cfg.MODEL.MASK_FORMER.REID_BRANCH = True

def add_dvis_config(cfg):
    cfg.INPUT.REVERSE_AGU = False
    cfg.MODEL.SEM_SEG_HEAD.RETURN_TRANSFORMER_FEATURE = False
    cfg.MODEL.TRACKER = CN()
    cfg.MODEL.TRACKER.DECODER_LAYERS = 6
    cfg.MODEL.TRACKER.NOISE_MODE = 'none'
    cfg.MODEL.TRACKER.NOISE_RATIO = 0.5
    cfg.MODEL.TRACKER.USE_CL = True
    cfg.MODEL.REFINER = CN()
    cfg.MODEL.REFINER.DECODER_LAYERS = 6
    cfg.MODEL.REFINER.USE_CL = True

    cfg.MODEL.MASK_FORMER.TEST.WINDOW_SIZE = 3
    cfg.MODEL.MASK_FORMER.TEST.TASK = 'vis'

    cfg.MODEL.MASK_FORMER.TEST.MAX_NUM = 20

    cfg.DATASETS.DATASET_RATIO = [1.0, ]
    # Whether category ID mapping is needed
    cfg.DATASETS.DATASET_NEED_MAP = [False, ]
    # dataset type, selected from ['video_instance', 'video_panoptic', 'video_semantic',
    #                              'image_instance', 'image_panoptic', 'image_semantic']
    cfg.DATASETS.DATASET_TYPE = ['video_instance', ]
    cfg.DATASETS.DATASET_TYPE_TEST = ['video_instance', ]

    # Pseudo Data Use
    cfg.INPUT.PSEUDO = CN()
    cfg.INPUT.PSEUDO.AUGMENTATIONS = ['rotation']
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN = (480, 512, 544, 576, 608, 640, 672, 704, 736, 768)
    cfg.INPUT.PSEUDO.MAX_SIZE_TRAIN = 768
    cfg.INPUT.PSEUDO.MIN_SIZE_TRAIN_SAMPLING = "choice_by_clip"
    cfg.INPUT.PSEUDO.CROP = CN()
    cfg.INPUT.PSEUDO.CROP.ENABLED = False
    cfg.INPUT.PSEUDO.CROP.TYPE = "absolute_range"
    cfg.INPUT.PSEUDO.CROP.SIZE = (384, 600)

    # LSJ
    cfg.INPUT.LSJ_AUG = CN()
    cfg.INPUT.LSJ_AUG.ENABLED = False
    cfg.INPUT.LSJ_AUG.IMAGE_SIZE = 1024
    cfg.INPUT.LSJ_AUG.MIN_SCALE = 0.1
    cfg.INPUT.LSJ_AUG.MAX_SCALE = 2.0

    cfg.SEED = 42
    cfg.DATALOADER.NUM_WORKERS = 4

    # contrastive learning plugin
    cfg.MODEL.CL_PLUGIN = CN()
    cfg.MODEL.CL_PLUGIN.CL_PLUGIN_NAME = "CTCLPlugin"
    cfg.MODEL.CL_PLUGIN.REID_WEIGHT = 2.
    cfg.MODEL.CL_PLUGIN.AUX_REID_WEIGHT = 3.
    cfg.MODEL.CL_PLUGIN.NUM_NEGATIVES = 99
    cfg.MODEL.CL_PLUGIN.FUSION_LOSS = False
    cfg.MODEL.CL_PLUGIN.BIO_CL = False
    cfg.MODEL.CL_PLUGIN.ONE_DIRECTION = True
    cfg.MODEL.CL_PLUGIN.MOMENTUM_EMBED = True
    cfg.MODEL.CL_PLUGIN.NOISE_EMBED = False

def add_ctvis_config(cfg):
    cfg.MODEL.MASK_FORMER.REID_HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_REID_HEAD_LAYERS = 3


