# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_daq_config(cfg):
    cfg.MODEL.VIDEO_HEAD = CN()
    cfg.MODEL.VIDEO_HEAD.NUM_NEW_INS = 10
    cfg.MODEL.VIDEO_HEAD.NUM_SLOTS = 5
    cfg.MODEL.VIDEO_HEAD.OFFLINE_TOPK_NUM = 20

    # FOR TRAINING
    cfg.MODEL.VIDEO_HEAD.TRAINING_SELECT_THRESHOLD = 0.1
    cfg.MODEL.VIDEO_HEAD.USING_THR = False
    cfg.MODEL.VIDEO_HEAD.SKIP_PARAMS = []
    cfg.INPUT.USING_FRAME_NUM = None
    cfg.INPUT.STEPS = None
    cfg.MODEL.VIDEO_HEAD.CL_ON_SLOTS = False

    # FOR INFERENCE
    cfg.MODEL.VIDEO_HEAD.INFERENCE_SELECT_THRESHOLD = 0.1
    cfg.MODEL.VIDEO_HEAD.AUX_INFERENCE_SELECT_THRESHOLD = 0.01
    cfg.MODEL.VIDEO_HEAD.NOISE_FRAME_NUM = 1  # when sequence length less than this value, then filtering the sequence as noise
    cfg.MODEL.VIDEO_HEAD.TEMPORAL_SCORE_TYPE = "mean"
    cfg.MODEL.VIDEO_HEAD.DIS_FG_THRESHOLD = 0.01
    cfg.MODEL.VIDEO_HEAD.MASK_NMS_THR = 0.6
    cfg.MODEL.VIDEO_HEAD.OVIS_INFER = False
    cfg.MODEL.VIDEO_HEAD.USE_LOCAL_ATTN = False
