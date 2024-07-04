# -*- coding: utf-8 -*-
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN

def add_ov_dvis_config(cfg):
    cfg.DATASETS.OV = True
    cfg.DATASETS.TEST2TRAIN = [None, ]

    # FC-CLIP model config
    cfg.MODEL.FC_CLIP = CN()
    cfg.MODEL.FC_CLIP.CLIP_MODEL_NAME = "convnext_large_d_320"
    cfg.MODEL.FC_CLIP.CLIP_PRETRAINED_WEIGHTS = "laion2b_s29b_b131k_ft_soup"
    cfg.MODEL.FC_CLIP.EMBED_DIM = 768
    cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_ALPHA = 0.4
    cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_BETA = 0.8
    cfg.MODEL.FC_CLIP.ENSEMBLE_ON_VALID_MASK = False


