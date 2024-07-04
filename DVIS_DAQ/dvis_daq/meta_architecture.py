import copy
import logging
import os.path
import random
from typing import Tuple, List
import einops
import numpy as np
from collections import defaultdict
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
import pycocotools.mask as mask_util

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from mask2former_video.utils.memory import retry_if_cuda_oom
from dvis_Plus.meta_architecture import MinVIS
from .matcher import FrameMatcher, NewInsHungarianMatcher
from .criterion import DAQCriterion
from .track_module import VideoInstanceCutter
from .refiner import TemporalRefiner
from mask2former_video.modeling.criterion import VideoSetCriterion as DVIS_VideoSetCriterion
from mask2former_video.modeling.matcher import VideoHungarianMatcher as DVIS_VideoHungarianMatcher


@META_ARCH_REGISTRY.register()
class DVIS_DAQ_online(MinVIS):
    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            sem_seg_head: nn.Module,
            criterion: nn.Module,
            num_queries: int,
            object_mask_threshold: float,
            overlap_threshold: float,
            metadata,
            size_divisibility: int,
            sem_seg_postprocess_before_inference: bool,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            # video head
            tracker: nn.Module,
            num_frames: int,
            window_inference: bool,
            frame_matcher: nn.Module,
            new_ins_matcher: nn.Module,
            inference_select_thr: float,
            aux_inference_select_thr: float,
            daq_criterion: nn.Module,
            using_thr: bool,
            # inference
            task: str,
            max_num: int,
            max_iter_num: int,
            window_size: int,
            noise_frame_num: int = 2,
            temporal_score_type: str = 'mean',
            mask_nms_thr: float = 0.5,
            # training
            using_frame_num: List = None,
            increasing_step: List = None,
            cfg = None,
    ):
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            # video
            num_frames=num_frames,
            window_inference=window_inference,
        )
        # frozen the segmenter
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.sem_seg_head.parameters():
            p.requires_grad_(False)

        self.tracker = tracker

        self.frame_matcher = frame_matcher
        self.new_ins_matcher = new_ins_matcher
        self.inference_select_thr = inference_select_thr
        self.aux_inference_select_thr = aux_inference_select_thr
        self.daq_criterion = daq_criterion
        self.using_thr = using_thr

        self.max_num = max_num
        self.iter = 0
        self.max_iter_num = max_iter_num
        self.window_size = window_size
        self.task = task
        assert self.task in ['vis', 'vss', 'vps', 'vos'], "Only support vis, vss and vps !"
        inference_dict = {
            'vis': self.inference_video_vis,
            'vss': self.inference_video_vss,
            'vps': self.inference_video_vps,
            'vos': self.inference_video_vos,
        }
        self.inference_video_task = inference_dict[self.task]
        self.noise_frame_num = noise_frame_num
        self.temporal_score_type = temporal_score_type
        self.mask_nms_thr = mask_nms_thr

        self.using_frame_num = using_frame_num
        self.increasing_step = increasing_step

        self.cfg = cfg

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        frame_matcher = FrameMatcher(
            cost_class=class_weight,
            cost_dice=dice_weight,
            cost_mask=mask_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        new_ins_matcher = NewInsHungarianMatcher(
            cost_class=class_weight,
            cost_dice=dice_weight,
            cost_mask=mask_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            num_new_ins=cfg.MODEL.VIDEO_HEAD.NUM_NEW_INS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers * 10 - 1):  # more is harmless
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[-1])
        daq_criterion = DAQCriterion(
            sem_seg_head.num_classes,
            new_ins_matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            num_new_ins=cfg.MODEL.VIDEO_HEAD.NUM_NEW_INS,
        )

        tracker = VideoInstanceCutter(
            hidden_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            feedforward_dim=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            num_head=cfg.MODEL.MASK_FORMER.NHEADS,
            decoder_layer_num=cfg.MODEL.TRACKER.DECODER_LAYERS,
            mask_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            num_new_ins=cfg.MODEL.VIDEO_HEAD.NUM_NEW_INS,
            training_select_threshold=cfg.MODEL.VIDEO_HEAD.TRAINING_SELECT_THRESHOLD,
            inference_select_threshold=cfg.MODEL.VIDEO_HEAD.INFERENCE_SELECT_THRESHOLD,
            num_slots=cfg.MODEL.VIDEO_HEAD.NUM_SLOTS,
            keep_threshold=cfg.MODEL.VIDEO_HEAD.DIS_FG_THRESHOLD,
            task=cfg.MODEL.MASK_FORMER.TEST.TASK,
            ovis_infer=cfg.MODEL.VIDEO_HEAD.OVIS_INFER,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": None,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[-1]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "tracker": tracker,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE,
            "frame_matcher": frame_matcher,
            "new_ins_matcher": new_ins_matcher,
            "inference_select_thr": cfg.MODEL.VIDEO_HEAD.INFERENCE_SELECT_THRESHOLD,
            "daq_criterion": daq_criterion,
            "using_thr": cfg.MODEL.VIDEO_HEAD.USING_THR,
            # inference
            "task": cfg.MODEL.MASK_FORMER.TEST.TASK,
            "noise_frame_num": cfg.MODEL.VIDEO_HEAD.NOISE_FRAME_NUM,
            "temporal_score_type": cfg.MODEL.VIDEO_HEAD.TEMPORAL_SCORE_TYPE,
            "max_num": cfg.MODEL.MASK_FORMER.TEST.MAX_NUM,
            "max_iter_num": cfg.SOLVER.MAX_ITER,
            "window_size": cfg.MODEL.MASK_FORMER.TEST.WINDOW_SIZE,
            "mask_nms_thr": cfg.MODEL.VIDEO_HEAD.MASK_NMS_THR,
            # training
            "using_frame_num": cfg.INPUT.USING_FRAME_NUM,
            "increasing_step": cfg.INPUT.STEPS,
            "aux_inference_select_thr": cfg.MODEL.VIDEO_HEAD.AUX_INFERENCE_SELECT_THRESHOLD,
            "cfg": cfg,
        }

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        # for running demo on very long videos
        if 'keep' in batched_inputs[0].keys():
            self.keep = batched_inputs[0]['keep']
        else:
            self.keep = False

        if self.using_frame_num is None:
            images = []
            for video in batched_inputs:
                for frame in video["image"]:
                    images.append(frame.to(self.device))
            select_fi_set = [i for i in range(len(images))]
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            using_frame_num = self.num_frames
        else:
            if self.iter < self.increasing_step[0]:
                using_frame_num = self.using_frame_num[0]
                self.using_thr = False
            else:
                using_frame_num = self.using_frame_num[1]
                self.using_thr = True
            images = []

            video_length = len(batched_inputs[0]["image"])
            if using_frame_num <= 0 or using_frame_num > video_length:
                using_frame_num = video_length
            if using_frame_num == video_length:
                select_fi_set = np.arange(0, video_length)
            else:
                start_fi = random.randint(0, using_frame_num - 1)
                end_fi = start_fi + using_frame_num - 1
                if end_fi >= video_length:
                    start_fi = video_length - using_frame_num
                    end_fi = video_length - 1
                select_fi_set = np.arange(start_fi, end_fi + 1)
            assert len(select_fi_set) == using_frame_num

            for video in batched_inputs:
                for fi, frame in enumerate(video["image"]):
                    if fi in select_fi_set:
                        images.append(frame.to(self.device))
            images = [(x - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)
            self.num_frames = using_frame_num

        self.backbone.eval()
        self.sem_seg_head.eval()
        with torch.no_grad():
            features = self.backbone(images.tensor)
            image_outputs = self.sem_seg_head(features)
            frame_embeds = image_outputs['pred_embds'].clone().detach()  # (b, c, t, q)
            mask_features = image_outputs['mask_features'].clone().detach().unsqueeze(0)
            pred_logits, pred_masks = image_outputs["pred_logits"].flatten(0, 1), image_outputs["pred_masks"].transpose(
                1, 2).flatten(0, 1)
            del image_outputs['mask_features']
            torch.cuda.empty_cache()
            image_outputs = {"pred_logits": pred_logits, "pred_masks": pred_masks}
        B, _, T, Q = frame_embeds.shape
        video_targets = self.prepare_targets(batched_inputs, images, select_fi_set)
        video_targets = self.split_video_targets(video_targets, clip_len=1)

        frame_targets = []
        for b in range(B):
            frame_targets.extend([item[b] for item in video_targets])
        frame_indices, aux_frame_indices, valid_masks = self.frame_matcher(image_outputs, frame_targets,
                                                                           self.aux_inference_select_thr)
        (new_frame_indices, new_aux_frame_indices, new_valid_masks, new_pred_logits,
         new_pred_masks, image_feats, image_pos) = [], [], [], [], [], [], []
        for i in range(T):
            new_frame_indices.append([frame_indices[b * T + i] for b in range(B)])
            new_aux_frame_indices.append([aux_frame_indices[b * T + i] for b in range(B)])
            new_valid_masks.append([valid_masks[b * T + i] for b in range(B)])
            new_pred_logits.append([pred_logits[b * T + i] for b in range(B)])
            new_pred_masks.append([pred_masks[b * T + i] for b in range(B)])
        frames_info = {"indices": new_frame_indices, "aux_indices": new_aux_frame_indices, "valid": new_valid_masks,
                       "pred_logits": new_pred_logits, "pred_masks": new_pred_masks,
                       "seg_query_feat": self.sem_seg_head.predictor.query_feat,
                       "seg_query_embed": self.sem_seg_head.predictor.query_embed}

        # TODO, maybe the stage 2 new ins modeling causing the unstability
        stage = 2
        if self.iter >= self.increasing_step[0]:
            stage = 3
        self.iter += 1

        outputs, slot_outputs = self.tracker(frame_embeds, mask_features, video_targets, frames_info,
                                             self.new_ins_matcher, stage=stage)

        losses = self.daq_criterion(outputs, video_targets)

        for k in list(losses.keys()):
            if k in self.daq_criterion.weight_dict:
                losses[k] *= self.daq_criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)

        slot_losses = self.daq_criterion(slot_outputs, video_targets[1:])
        new_slot_losses = dict()
        for k in list(slot_losses.keys()):
            if k in self.daq_criterion.weight_dict:
                new_slot_losses["slot_" + k] = slot_losses[k] * self.daq_criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                slot_losses.pop(k)
        losses.update(new_slot_losses)

        return losses

    def inference(self, batched_inputs, window_size=5):
        # for running demo on very long videos
        if 'keep' in batched_inputs[0].keys():
            self.keep = batched_inputs[0]['keep']
        else:
            self.keep = False

        if 'long_video_start_fidx' in batched_inputs[0].keys():
            long_video_start_fidx = batched_inputs[0]['long_video_start_fidx']
        else:
            long_video_start_fidx = -1

        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        num_frames = len(images.tensor)
        outputs = self.run_window_inference(images.tensor, window_size=window_size, long_video_start_fidx=long_video_start_fidx)
        H, W = outputs["shape"]

        if len(outputs["pred_logits"]) == 0:
            video_output = {
                "image_size": images.image_sizes[0],
                "pred_scores": [],
                "pred_labels": [],
                "pred_masks": [] if self.task != "vps" else torch.zeros((num_frames, H, W), dtype=torch.int32),
                "pred_ids": [],
                "segments_infos": [],
                "task": self.task,
            }
            return video_output
        mask_cls_results = outputs["pred_logits"]  # b, n, k+1
        mask_pred_results = outputs["pred_masks"]  # b, n, t, h, w
        pred_ids = [torch.arange(0, outputs['pred_masks'].size(1))]

        mask_cls_result = mask_cls_results[0]
        mask_pred_result = mask_pred_results[0]

        pred_id = pred_ids[0]
        first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

        input_per_image = batched_inputs[0]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation

        height = input_per_image.get('height', image_size[0])  # raw image size before data augmentation
        width = input_per_image.get('width', image_size[1])

        if self.task == "vos":
            return retry_if_cuda_oom(self.inference_video_vos)(
                batched_inputs[0], outputs, first_resize_size, image_size
            )
        else:
            return retry_if_cuda_oom(self.inference_video_task)(
                mask_cls_result, mask_pred_result, image_size, height, width, first_resize_size, pred_id,
            )

    def prepare_targets(self, targets, images, select_fi_set):  # TODO, datamapper match with the function
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)
            fi2idx = {fi: idx for idx, fi in enumerate(select_fi_set)}

            gt_classes_per_video = targets_per_video["instances"][select_fi_set[0]].gt_classes.to(self.device)
            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                if f_i not in select_fi_set:
                    continue
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                _update_cls = gt_classes_per_video == -1
                gt_classes_per_video[_update_cls] = targets_per_frame.gt_classes[_update_cls]
                gt_ids_per_video.append(targets_per_frame.gt_ids)
                if isinstance(targets_per_frame.gt_masks, BitMasks):
                    gt_masks_per_video[:, fi2idx[f_i], :h, :w] = targets_per_frame.gt_masks.tensor
                else:
                    gt_masks_per_video[:, fi2idx[f_i], :h, :w] = targets_per_frame.gt_masks

            gt_ids_per_video = torch.stack(gt_ids_per_video, dim=1)  # ntgt, T
            gt_ids_per_video[gt_masks_per_video.sum(dim=(2, 3)) == 0] = -1
            valid_bool_frame = (gt_ids_per_video != -1)
            valid_bool_clip = valid_bool_frame.any(dim=-1)

            gt_classes_per_video = gt_classes_per_video[valid_bool_clip].long()  # N,
            gt_ids_per_video = gt_ids_per_video[valid_bool_clip].long()  # N, num_frames
            gt_masks_per_video = gt_masks_per_video[valid_bool_clip].float()  # N, num_frames, H, W
            valid_bool_frame = valid_bool_frame[valid_bool_clip]

            if len(gt_ids_per_video) > 0:
                min_id = max(gt_ids_per_video[valid_bool_frame].min(), 0)
                gt_ids_per_video[valid_bool_frame] -= min_id

            gt_instances.append(
                {
                    "labels": gt_classes_per_video, "ids": gt_ids_per_video, "masks": gt_masks_per_video,
                    "video_len": targets_per_video["video_len"], "frame_idx": targets_per_video["frame_idx"],
                }
            )
        return gt_instances

    def split_video_targets(self, clip_targets, clip_len=1):
        clip_target_splits = dict()
        for targets_per_video in clip_targets:
            labels = targets_per_video["labels"]  # Ni (number of instances)

            ids = targets_per_video["ids"]  # Ni, T
            masks = targets_per_video["masks"]  # Ni, T, H, W
            frame_idx = targets_per_video["frame_idx"]  # T

            masks_splits = masks.split(clip_len, dim=1)
            ids_splits = ids.split(clip_len, dim=1)

            prev_valid = torch.zeros_like(labels).bool()
            last_valid = torch.zeros_like(labels).bool()
            for clip_idx, (_masks, _ids) in enumerate(zip(masks_splits, ids_splits)):
                valid_inst = _masks.sum(dim=(1, 2, 3)) > 0.
                new_inst = (prev_valid == False) & (valid_inst == True)
                disappear_inst_ref2last = (last_valid == True) & (valid_inst == False)

                if not clip_idx in clip_target_splits:
                    clip_target_splits[clip_idx] = []

                clip_target_splits[clip_idx].append(
                    {
                        "labels": labels, "ids": _ids.squeeze(1), "masks": _masks.squeeze(1),
                        "video_len": targets_per_video["video_len"],
                        "frame_idx": frame_idx[clip_idx * clip_len:(clip_idx + 1) * clip_len],
                        "valid_inst": valid_inst,
                        "new_inst": new_inst,
                        "disappear_inst": disappear_inst_ref2last,
                    }
                )

                prev_valid = prev_valid | valid_inst
                last_valid = valid_inst

        return list(clip_target_splits.values())

    def run_window_inference(self, images_tensor, window_size=30, long_video_start_fidx=-1):
        video_start_idx = long_video_start_fidx if long_video_start_fidx >= 0 else 0

        window_size = 5
        num_frames = len(images_tensor)
        to_store = "cpu"
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size
            # segmenter inference
            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)
            # remove unnecessary variables to save GPU memory
            del features['res2'], features['res3'], features['res4'], features['res5']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            # referring tracker inference
            frame_embds = out['pred_embds']  # (b, c, t, q)
            mask_features = out['mask_features'].unsqueeze(0)  # as B == 1
            pred_logits, pred_masks = out["pred_logits"].flatten(0, 1), out["pred_masks"].transpose(1, 2).flatten(0, 1)

            B, _, T, Q = frame_embds.shape
            H, W = mask_features.shape[-2:]

            pred_scores = torch.max(pred_logits.softmax(dim=-1)[:, :, :-1], dim=-1)[0]
            valid_masks = pred_scores > self.aux_inference_select_thr
            new_pred_logits, new_pred_masks, new_valid_masks = [], [], []
            for t in range(T):
                new_pred_logits.append([pred_logits[b * T + t] for b in range(B)])
                new_pred_masks.append([pred_masks[b * T + t] for b in range(B)])
                new_valid_masks.append([valid_masks[b * T + t] for b in range(B)])
            frame_info = {"pred_logits": new_pred_logits, "pred_masks": new_pred_masks, "valid": new_valid_masks,
                          "seg_query_feat": self.sem_seg_head.predictor.query_feat,
                          "seg_query_embed": self.sem_seg_head.predictor.query_embed}

            if i != 0 or self.keep:
                self.tracker.inference(frame_embds, mask_features, frame_info,
                                       video_start_idx + start_idx, resume=True, to_store=to_store)
            else:
                self.tracker.inference(frame_embds, mask_features, frame_info,
                                       video_start_idx + start_idx, to_store=to_store)

        logits_list = []
        full_logits_list = []
        masks_list = []
        seq_id_list = []
        dead_seq_id_list = []
        padding_mask_list = []
        for seq_id, ins_seq in self.tracker.video_ins_hub.items():
            if len(ins_seq.pred_masks) < self.noise_frame_num:
                # if ins_seq.sT + len(ins_seq.pred_masks) == num_frames, which means this object appeared at the end of
                # this clip and cloud be exists in the next clip.
                if ins_seq.sT + len(ins_seq.pred_masks) < video_start_idx + num_frames:
                    continue
            full_masks = torch.ones(num_frames, H, W).to(torch.float32).to(to_store) * -1e4
            full_logits = torch.ones(num_frames, self.sem_seg_head.num_classes + 1).to(torch.float32).to("cuda") * -1e4
            full_logits[:, -1] = 1.
            padding_mask = torch.ones(size=(num_frames, )) > 0
            seq_logits = []
            seq_start_t = ins_seq.sT
            for j in range(len(ins_seq.pred_masks)):
                if seq_start_t + j < video_start_idx:
                    continue
                re_j = seq_start_t + j - video_start_idx
                full_masks[re_j, :, :] = ins_seq.pred_masks[j]
                full_logits[re_j, :] = ins_seq.pred_logits[j]
                padding_mask[re_j] = False
                seq_logits.append(ins_seq.pred_logits[j])
            if len(seq_logits) == 0:
                continue
            seq_logits = torch.stack(seq_logits, dim=0).mean(0)  # n, c -> c
            logits_list.append(seq_logits)
            masks_list.append(full_masks)
            full_logits_list.append(full_logits)
            padding_mask_list.append(padding_mask)
            assert ins_seq.gt_id == seq_id
            seq_id_list.append(seq_id)
            if ins_seq.dead:
                dead_seq_id_list.append(seq_id)

        for seq_id in dead_seq_id_list:  # for handling large long videos, saving memory
            self.tracker.video_ins_hub.pop(seq_id)

        if len(logits_list) > 0:
            pred_cls = torch.stack(logits_list, dim=0)[None, ...]  # b, n, c
            pred_masks = torch.stack(masks_list, dim=0)[None, ...]  # b, n, t, h, w
            pred_ids = torch.as_tensor(seq_id_list).to(torch.int64)[None, :]  # b, n
            padding_masks = torch.stack(padding_mask_list, dim=0)[None, ...]  # b, n, t
            full_logits_tensor = torch.stack(full_logits_list)[None, ...]  # b, n, t, c
        else:
            pred_cls = []
            pred_masks = []
            pred_ids = []
            padding_masks = []
            full_logits_tensor = []

        outputs = {
            "pred_logits": pred_cls,
            "pred_masks": pred_masks,
            "pred_ids": pred_ids,
            "shape": (H, W),
            "padding_masks": padding_masks,
            "full_logits": full_logits_tensor,
        }

        return outputs

    def inference_video_vos(self, targets, outputs, img_pad_size, image_size):
        mask_cls_results = outputs["pred_logits"]  # b, n, k+1
        mask_pred_results = outputs["pred_masks"]  # b, n, t, h, w
        s4_h, s4_w = mask_pred_results.shape[-2:]

        scores = torch.max(F.softmax(mask_cls_results[0], dim=-1)[:, :-1], dim=1)[0]  # n,
        # max_num = mask_pred_results.shape[1]
        max_num = self.max_num if self.max_num < mask_pred_results.shape[1] else mask_pred_results.shape[1]
        topk_scores, topk_indices = scores.topk(max_num, sorted=False)  # 20,
        topk_masks = mask_pred_results[0, topk_indices.to(mask_pred_results.device), ...]  # 20, t, h, w
        # topk_masks = topk_masks.to(self.device)

        video_len = len(targets["image"])
        img_h_pad, img_w_pad = img_pad_size

        mask_dict = {}
        for fidx in range(video_len):
            targets_per_frame = targets["instances"][fidx]
            targets_per_frame = targets_per_frame.to(self.device)
            ori_ids_list = targets_per_frame.ori_id
            if len(ori_ids_list) == 0 or fidx > 0:
                continue
            has_new_obj = False
            for ori_id in ori_ids_list:
                if ori_id not in mask_dict:
                    has_new_obj = True
            if not has_new_obj:
                continue
            h, w = targets_per_frame.image_size
            _num_instance = len(targets_per_frame)
            mask_shape = [_num_instance, img_h_pad, img_w_pad]
            gt_masks_per_frame = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)
            if isinstance(targets_per_frame.gt_masks, BitMasks):
                gt_masks_per_frame[:, :h, :w] = targets_per_frame.gt_masks.tensor
            else:
                gt_masks_per_frame[:, :h, :w] = targets_per_frame.gt_masks

            gt_masks_per_frame = gt_masks_per_frame.to(torch.float32)
            gt_masks_s4 = F.interpolate(gt_masks_per_frame.unsqueeze(0), size=(s4_h, s4_w), mode="nearest")
            gt_masks_s4 = gt_masks_s4[0]  # n', h, w
            pred_masks_s4 = topk_masks[:, fidx, :, :]  # 20, h, w
            # _mask_iou = mask_iou(gt_masks_s4, pred_masks_s4)
            _mask_iou = mask_iou(pred_masks_s4, gt_masks_s4.to(pred_masks_s4.device))
            cost = 1 - _mask_iou
            src_i, tgt_i = linear_sum_assignment(cost)
            sorted_idx = tgt_i.argsort()
            src_i, tgt_i = src_i[sorted_idx], tgt_i[sorted_idx]
            tgt2src_idx = {_tgt_i: _src_i for _src_i, _tgt_i in zip(src_i, tgt_i)}

            for obj_idx, ori_id in enumerate(ori_ids_list):
                if ori_id in mask_dict:
                    continue
                mask_dict[ori_id] = [tgt2src_idx[obj_idx], fidx]

        # image_size: before padding
        output_height = targets.get('height', image_size[0])  # raw image size before data augmentation
        output_width = targets.get('width', image_size[1])

        def resize_to_out(mask):
            mask = F.interpolate(mask[None, None], size=img_pad_size, mode="bilinear", align_corners=False)
            mask = mask[:, :, :image_size[0], :image_size[1]]
            mask = F.interpolate(mask, size=(output_height, output_width), mode="bilinear", align_corners=False)
            mask = mask[0, 0] > 0.
            return mask

        # pred_masks = F.interpolate(
        #     topk_masks, size=img_pad_size, mode="bilinear", align_corners=False
        # )
        # pred_masks = pred_masks[:, :, : image_size[0], : image_size[1]]
        # pred_masks = F.interpolate(
        #     pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
        # )
        # masks = pred_masks > 0.  # 20, t, h, w
        video_id = targets["file_names"][0].split('/')[-2]
        save_dir = os.path.join(self.cfg.OUTPUT_DIR, "inference", video_id)
        os.makedirs(save_dir, exist_ok=True)
        print(f"processing {save_dir}...")

        cur_obj_ids = None
        for fidx in range(video_len):
            targets_per_frame = targets["instances"][fidx]
            if fidx == 0:
                cur_obj_ids = targets_per_frame.ori_id

            cur_obj_ids_int = [int(x) for x in cur_obj_ids]  # 1, 2, 3...
            if len(cur_obj_ids_int) != 0:
                mask_merge = np.zeros((output_height, output_width, max(cur_obj_ids_int)+1))  # (H, W, N+1)
            else:
                mask_merge = np.zeros((output_height, output_width, 1))
            tmp_list = []
            for cur_id in cur_obj_ids:
                mask_merge[:, :, int(cur_id)] = resize_to_out(topk_masks[mask_dict[cur_id][0], fidx, :, :]).numpy()
                tmp_list.append(resize_to_out(topk_masks[mask_dict[cur_id][0], fidx, :, :]).numpy())
            if len(tmp_list) != 0:
                back_prob = np.prod(1 - np.stack(tmp_list, axis=-1), axis=-1, keepdims=False)
                mask_merge[:, :, 0] = back_prob
            mask_merge_final = np.argmax(mask_merge, axis=-1).astype(np.uint8)  # (H, W)
            mask_merge_final = Image.fromarray(mask_merge_final).convert('P')
            mask_merge_final.putpalette(targets["mask_palette"])
            file_name = targets["file_names"][fidx].split('/')[-1]
            save_img_dir = os.path.join(save_dir, file_name.replace('.jpg', '.png'))
            mask_merge_final.save(save_img_dir)
            mask_merge_final.close()

        return

    def inference_video_vis(
            self, pred_cls, pred_masks, img_size, output_height, output_width,
            first_resize_size, pred_id, aux_pred_cls=None
    ):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            if aux_pred_cls is not None:
                aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
                scores = torch.maximum(scores, aux_pred_cls.to(scores))
            labels = torch.arange(
                self.sem_seg_head.num_classes, device=self.device
            ).unsqueeze(0).repeat(scores.shape[0], 1).flatten(0, 1)
            # keep top-K predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.max_num, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices.to(pred_masks.device)]
            pred_ids = pred_id[topk_indices.to(pred_id.device)]

            # interpolation to original image size
            pred_masks = F.interpolate(
                pred_masks, size=first_resize_size, mode="bilinear", align_corners=False
            )
            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )
            masks = pred_masks > 0.
            del pred_masks

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_ids = pred_ids.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []
            out_ids = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
            "pred_ids": out_ids,
            "task": "vis",
        }

        return video_output

    def inference_video_vps(
            self, pred_cls, pred_masks, img_size, output_height, output_width,
            first_resize_size, pred_id, aux_pred_cls=None
    ):
        pred_cls = F.softmax(pred_cls, dim=-1)
        if aux_pred_cls is not None:
            aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
            pred_cls[:, :-1] = torch.maximum(pred_cls[:, :-1], aux_pred_cls.to(pred_cls))
        mask_pred = pred_masks
        scores, labels = pred_cls.max(-1)

        # filter out the background prediction
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep.to(scores.device)]
        cur_classes = labels[keep.to(labels.device)]
        cur_ids = pred_id[keep.to(pred_id.device)]
        cur_masks = mask_pred[keep.to(mask_pred.device)]

        # interpolation to original image size
        cur_masks = F.interpolate(
            cur_masks, size=first_resize_size, mode="bilinear", align_corners=False
        )
        cur_masks = cur_masks[:, :, :img_size[0], :img_size[1]].sigmoid()
        cur_masks = F.interpolate(
            cur_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
        )
        cur_prob_masks = cur_scores.view(-1, 1, 1, 1).to(cur_masks.device) * cur_masks

        # initial panoptic_seg and segments infos
        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((cur_masks.size(1), h, w), dtype=torch.int32, device=cur_masks.device)
        segments_infos = []
        out_ids = []
        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask
            return {
                "image_size": (output_height, output_width),
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "pred_ids": out_ids,
                "task": "vps",
            }
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)  # (t, h, w)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class < len(self.metadata.thing_dataset_id_to_contiguous_id)
                # filter out the unstable segmentation results
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue
                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_infos.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
                    out_ids.append(cur_ids[k])

            return {
                "image_size": (output_height, output_width),
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "pred_ids": out_ids,
                "task": "vps",
            }

    def inference_video_vss(
            self, pred_cls, pred_masks, img_size, output_height, output_width,
            first_resize_size, pred_id, aux_pred_cls=None, **kwargs
    ):
        mask_cls = F.softmax(pred_cls, dim=-1)[..., :-1]
        if aux_pred_cls is not None:
            aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
            mask_cls[..., :-1] = torch.maximum(mask_cls[..., :-1], aux_pred_cls.to(mask_cls))
        mask_pred = pred_masks
        # interpolation to original image size
        cur_masks = F.interpolate(
            mask_pred, size=first_resize_size, mode="bilinear", align_corners=False
        )
        cur_masks = cur_masks[:, :, :img_size[0], :img_size[1]].sigmoid()
        cur_masks = F.interpolate(
            cur_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
        )

        semseg = torch.einsum("qc,qthw->cthw", mask_cls, cur_masks)
        sem_score, sem_mask = semseg.max(0)
        sem_mask = sem_mask
        return {
            "image_size": (output_height, output_width),
            "pred_masks": sem_mask.cpu(),
            "task": "vss",
        }


@META_ARCH_REGISTRY.register()
class DVIS_DAQ_offline(DVIS_DAQ_online):
    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            sem_seg_head: nn.Module,
            criterion: nn.Module,
            num_queries: int,
            object_mask_threshold: float,
            overlap_threshold: float,
            metadata,
            size_divisibility: int,
            sem_seg_postprocess_before_inference: bool,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            # video head
            tracker: nn.Module,
            num_frames: int,
            window_inference: bool,
            frame_matcher: nn.Module,
            new_ins_matcher: nn.Module,
            inference_select_thr: float,
            aux_inference_select_thr: float,
            daq_criterion: nn.Module,
            using_thr: bool,
            offline_topk_ins: int = 20,
            # inference
            task: str,
            max_num: int,
            max_iter_num: int,
            window_size: int,
            noise_frame_num: int = 2,
            temporal_score_type: str = 'mean',
            mask_nms_thr: float = 0.5,
            # training
            using_frame_num: List = None,
            increasing_step: List = None,
            # offline
            refiner: nn.Module,
            cfg = None,
    ):
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            metadata=metadata,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            # video head
            tracker=tracker,
            num_frames=num_frames,
            window_inference=window_inference,
            frame_matcher=frame_matcher,
            new_ins_matcher=new_ins_matcher,
            inference_select_thr=inference_select_thr,
            aux_inference_select_thr=aux_inference_select_thr,
            daq_criterion=daq_criterion,
            using_thr=using_thr,
            # inference
            task=task,
            max_num=max_num,
            max_iter_num=max_iter_num,
            window_size=window_size,
            noise_frame_num=noise_frame_num,
            temporal_score_type=temporal_score_type,
            mask_nms_thr=mask_nms_thr,
            # training
            using_frame_num=using_frame_num,
            increasing_step=increasing_step,
        )

        self.offline_topk_ins = offline_topk_ins
        self.cfg = cfg

        # frozen the referring tracker
        for p in self.tracker.parameters():
            p.requires_grad_(False)

        self.refiner = refiner

        dis_tracker_params = 0
        for n, p in self.tracker.named_parameters():
            if "slot" in n:
                dis_tracker_params += p.numel()

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        frame_matcher = FrameMatcher(
            cost_class=class_weight,
            cost_dice=dice_weight,
            cost_mask=mask_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        new_ins_matcher = NewInsHungarianMatcher(
            cost_class=class_weight,
            cost_dice=dice_weight,
            cost_mask=mask_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            num_new_ins=cfg.MODEL.VIDEO_HEAD.NUM_NEW_INS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight, }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers * 10 - 1):  # more is harmless
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        dvis_matcher = DVIS_VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            # since when calculating the loss, the t frames of a video are flattened into a image with size of (th, w),
            # the number of sampling points is increased t times accordingly.
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS * cfg.INPUT.SAMPLING_FRAME_NUM,
        )

        daq_criterion = DVIS_VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=dvis_matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS * cfg.INPUT.SAMPLING_FRAME_NUM,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[-1])
        tracker = VideoInstanceCutter(
            hidden_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            feedforward_dim=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            num_head=cfg.MODEL.MASK_FORMER.NHEADS,
            decoder_layer_num=cfg.MODEL.TRACKER.DECODER_LAYERS,
            mask_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            num_classes=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            num_new_ins=cfg.MODEL.VIDEO_HEAD.NUM_NEW_INS,
            training_select_threshold=cfg.MODEL.VIDEO_HEAD.TRAINING_SELECT_THRESHOLD,
            inference_select_threshold=cfg.MODEL.VIDEO_HEAD.INFERENCE_SELECT_THRESHOLD,
            num_slots=cfg.MODEL.VIDEO_HEAD.NUM_SLOTS,
            keep_threshold=cfg.MODEL.VIDEO_HEAD.DIS_FG_THRESHOLD,
            ovis_infer=cfg.MODEL.VIDEO_HEAD.OVIS_INFER,
        )

        refiner = TemporalRefiner(
            hidden_channel=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            feedforward_channel=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            num_head=cfg.MODEL.MASK_FORMER.NHEADS,
            decoder_layer_num=cfg.MODEL.REFINER.DECODER_LAYERS,
            mask_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            class_num=cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            windows=cfg.MODEL.MASK_FORMER.TEST.WINDOW_SIZE,
            use_local_attn=cfg.MODEL.VIDEO_HEAD.USE_LOCAL_ATTN
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": None,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[-1]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "tracker": tracker,
            "refiner": refiner,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE,
            "frame_matcher": frame_matcher,
            "new_ins_matcher": new_ins_matcher,
            "inference_select_thr": cfg.MODEL.VIDEO_HEAD.INFERENCE_SELECT_THRESHOLD,
            "daq_criterion": daq_criterion,
            "using_thr": cfg.MODEL.VIDEO_HEAD.USING_THR,
            "offline_topk_ins": cfg.MODEL.VIDEO_HEAD.OFFLINE_TOPK_NUM,
            # inference
            "task": cfg.MODEL.MASK_FORMER.TEST.TASK,
            "noise_frame_num": cfg.MODEL.VIDEO_HEAD.NOISE_FRAME_NUM,
            "temporal_score_type": cfg.MODEL.VIDEO_HEAD.TEMPORAL_SCORE_TYPE,
            "max_num": cfg.MODEL.MASK_FORMER.TEST.MAX_NUM,
            "max_iter_num": cfg.SOLVER.MAX_ITER,
            "window_size": cfg.MODEL.MASK_FORMER.TEST.WINDOW_SIZE,
            "mask_nms_thr": cfg.MODEL.VIDEO_HEAD.MASK_NMS_THR,
            # training
            "using_frame_num": cfg.INPUT.USING_FRAME_NUM,
            "increasing_step": cfg.INPUT.STEPS,
            "aux_inference_select_thr": cfg.MODEL.VIDEO_HEAD.AUX_INFERENCE_SELECT_THRESHOLD,
            "cfg": cfg
        }

    def forward(self, batched_inputs):
        if not self.training:
            return self.inference(batched_inputs)

        if 'keep' in batched_inputs[0].keys():
            self.keep = batched_inputs[0]['keep']
        else:
            self.keep = False

        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        video_length = len(batched_inputs[0]["image"])
        self.backbone.eval()
        self.sem_seg_head.eval()
        self.tracker.eval()

        with torch.no_grad():
            common_out = self.common_inference(images.tensor, window_size=5, to_store="cuda")
        frame_embeds = common_out["frame_embeds"].clone().detach()
        mask_features = common_out["mask_features"].clone().detach()
        B, C, T, Q = frame_embeds.shape
        H, W = mask_features.shape[-2:]

        instance_embeds = common_out["instance_embeds"].clone().detach()  # (b, c, t, q)
        padding_masks = common_out["padding_masks"].clone().detach()  # (b, q, t)
        online_out = common_out["online_out"]
        dvis_targets = self.dvis_prepare_targets(batched_inputs, images)
        refined_out = self.refiner(instance_embeds, padding_masks, frame_embeds, mask_features, None)
        if self.iter < self.max_iter_num // 2:
            online_out, refined_out, dvis_targets = self.frame_decoder_loss_reshape(
                refined_out, dvis_targets, image_outputs=online_out
            )
        else:
            online_out, refined_out, dvis_targets = self.frame_decoder_loss_reshape(
                refined_out, dvis_targets, image_outputs=None
            )
        self.iter += 1

        losses, matching_result = self.daq_criterion(refined_out, dvis_targets,
                                                     matcher_outputs=online_out,
                                                     ret_match_result=True)

        for k in list(losses.keys()):
            if k in self.daq_criterion.weight_dict:
                losses[k] *= self.daq_criterion.weight_dict[k]
            else:
                # remove this loss if not specified in `weight_dict`
                losses.pop(k)
        return losses

    def segmenter_windows_inference(self, images_tensor, window_size=5, long_video_start_fidx=-1, to_store="cpu"):
        image_outputs = {}
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1

        outs_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size

            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)

            del features['res2'], features['res3'], features['res4'], features['res5']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']

            outs_list.append(out)

        image_outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in outs_list], dim=2).detach()
        image_outputs['mask_features'] = torch.cat([x['mask_features'] for x in outs_list], dim=0).detach()
        image_outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in outs_list], dim=1).detach()
        if to_store == "cpu":
            image_outputs['pred_masks'] = torch.cat([x['pred_masks'].to("cpu").to(torch.float32) for x in outs_list],
                                                    dim=2).detach()
        else:
            image_outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in outs_list], dim=2).detach()
        return image_outputs

    def common_inference(self, images_tensor, window_size=30, long_video_start_fidx=-1, to_store="cpu"):
        video_start_idx = long_video_start_fidx if long_video_start_fidx >= 0 else 0

        with torch.no_grad():
            image_outputs = self.segmenter_windows_inference(images_tensor, window_size=window_size, to_store=to_store)
            seg_out = {"pred_logits": image_outputs["pred_logits"].clone().detach(),
                       "pred_masks": image_outputs["pred_masks"].clone().detach(),
                       "pred_embds": image_outputs['pred_embds'].clone().detach()}  # .clone().detach() is Important!!!
            frame_embeds = image_outputs['pred_embds'].clone().detach()  # (b, c, t, q)
            mask_features = image_outputs['mask_features'].clone().detach().unsqueeze(0)
            pred_logits, pred_masks = image_outputs["pred_logits"].clone().detach().flatten(0, 1), image_outputs[
                "pred_masks"].clone().detach().transpose(1, 2).flatten(0, 1)
            del image_outputs['mask_features'], image_outputs['pred_embds'], image_outputs['pred_logits'], \
            image_outputs[
                'pred_masks']
            torch.cuda.empty_cache()
            B, C, T, Q = frame_embeds.shape
            H, W = mask_features.shape[-2:]

            new_valid_masks, new_pred_logits, new_pred_masks = [], [], []
            pred_scores = torch.max(pred_logits.softmax(dim=-1)[:, :, :-1], dim=-1)[0]
            valid_masks = pred_scores > self.aux_inference_select_thr
            for i in range(T):
                new_valid_masks.append([valid_masks[b * T + i] for b in range(B)])
                new_pred_logits.append([pred_logits[b * T + i] for b in range(B)])
                new_pred_masks.append([pred_masks[b * T + i] for b in range(B)])

            num_frames = len(images_tensor)
            iters = len(images_tensor) // window_size
            infer_func = self.tracker.forward_offline_mode if self.training else self.tracker.inference
            if len(images_tensor) % window_size != 0:
                iters += 1
            for i in range(iters):
                start_idx = i * window_size
                end_idx = (i + 1) * window_size if (i + 1) * window_size < num_frames else num_frames

                frame_embeds_i = frame_embeds[:, :, start_idx:end_idx, :]
                mask_features_i = mask_features[:, start_idx:end_idx, ...]
                frames_info_i = {
                    "valid": new_valid_masks[start_idx:end_idx], "pred_logits": new_pred_logits[start_idx:end_idx],
                    "pred_masks": new_pred_masks[start_idx:end_idx],
                    "seg_query_feat": self.sem_seg_head.predictor.query_feat,
                    "seg_query_embed": self.sem_seg_head.predictor.query_embed
                }

                if i != 0 or self.keep:
                    infer_func(frame_embeds_i, mask_features_i, frames_info_i,
                               video_start_idx + start_idx, resume=True, to_store=to_store)
                else:
                    infer_func(frame_embeds_i, mask_features_i, frames_info_i,
                               video_start_idx + start_idx, to_store=to_store)
        logits_list = []
        masks_list = []
        pos_embed_list = []
        trc_queries_list = []
        padding_mask_list = []
        seq_id_list = []
        dead_seq_id_list = []
        for seq_id, ins_seq in self.tracker.video_ins_hub.items():
            if len(ins_seq.pred_masks) < self.noise_frame_num:
                # if ins_seq.sT + len(ins_seq.pred_masks) == num_frames, which means this object appeared at the end of
                # this clip and cloud be exists in the next clip.
                if ins_seq.sT + len(ins_seq.pred_masks) < video_start_idx + num_frames:
                    continue
            full_masks = torch.ones(num_frames, H, W).to(torch.float32).to(to_store) * -1e4
            seq_logits = []
            seq_start_t = ins_seq.sT
            for j in range(len(ins_seq.pred_masks)):
                if seq_start_t + j < video_start_idx:
                    continue
                re_j = seq_start_t + j - video_start_idx
                full_masks[re_j, :, :] = ins_seq.pred_masks[j]
                seq_logits.append(ins_seq.pred_logits[j])
            if len(seq_logits) == 0:
                continue
            seq_logits = torch.stack(seq_logits, dim=0).mean(0)  # n, c -> c
            logits_list.append(seq_logits)
            masks_list.append(full_masks)
            pos_embed_list.append(ins_seq.similarity_guided_pos_embed)  # c

            front_padding_length = seq_start_t - video_start_idx  # for handling very long video
            tail_padding_length = num_frames - len(ins_seq.embeds) - front_padding_length
            padding_embed = self.refiner.padding_embed(ins_seq.similarity_guided_pos_embed)
            # padding_embed = ins_seq.similarity_guided_pos_embed
            valid_trc_queries = []
            for j in range(len(ins_seq.embeds)):
                valid_trc_queries.append(ins_seq.embeds[j])
            trc_queries_list.append(torch.cat(
                # [torch.zeros(size=(front_padding_length, C)).to(valid_trc_queries[-1]),
                [padding_embed[None].repeat(front_padding_length, 1),
                 torch.stack(valid_trc_queries),
                 padding_embed[None].repeat(tail_padding_length, 1)]
                # torch.zeros(size=(tail_padding_length, C)).to(valid_trc_queries[-1])]
            ))
            padding_mask_list.append(
                torch.BoolTensor(
                    [True] * front_padding_length + [False] * len(valid_trc_queries) + [True] * tail_padding_length).to(
                    "cuda")
            )
            seq_id_list.append(seq_id)

            if ins_seq.dead:
                dead_seq_id_list.append(seq_id)
        if len(logits_list) == 0:
            online_logits = torch.zeros(size=(1, 0, pred_logits.shape[-1])).to(pred_logits)
            online_masks = torch.zeros(size=(1, 0, T, H, W)).to(pred_masks)
            trc_queries = torch.zeros(size=(1, 0, T, C)).to(pred_logits)
            padding_masks = torch.zeros(size=(1, 0, T), dtype=torch.bool).to("cuda")
            seq_id_tensor = torch.IntTensor([]).to("cuda")
        else:
            online_logits = torch.stack(logits_list).unsqueeze(0)  # b, q, k+1
            online_masks = torch.stack(masks_list).unsqueeze(0)  # b, q, t, h, w
            trc_queries = torch.stack(trc_queries_list).unsqueeze(0)  # b, q, t, c
            padding_masks = torch.stack(padding_mask_list).unsqueeze(0)  # b, q, t
            seq_id_tensor = torch.IntTensor(seq_id_list).to("cuda")

        # topk video thing-instance sequence
        scores = torch.max(F.softmax(online_logits[0, :, :], dim=-1)[:, :-1], dim=-1)[0]
        if self.offline_topk_ins > scores.shape[0]:
            topk_indices = torch.arange(scores.shape[0]).to("cuda")
        else:
            _, topk_indices = scores.topk(self.offline_topk_ins, sorted=False)
        online_logits = online_logits[:, topk_indices, :]
        online_masks = online_masks[:, topk_indices.to(to_store), ...]
        trc_queries = trc_queries[:, topk_indices, ...]
        padding_masks = padding_masks[:, topk_indices, :]
        seq_id_tensor = seq_id_tensor[topk_indices]
        seq_id_list = seq_id_tensor.cpu().numpy().tolist()

        num_left = self.tracker.num_new_ins - online_logits.shape[1]
        if num_left > 0:
            naive_link_out = self.minvis_post_processing(seg_out)
            naive_scores = torch.max(F.softmax(naive_link_out["pred_logits"], dim=-1)[0, :, :-1], dim=-1)[0]
            _, topk_indices = naive_scores.topk(num_left, sorted=False)
            online_logits = torch.cat([online_logits, naive_link_out["pred_logits"][:, topk_indices, :]], dim=1)
            online_masks = torch.cat([online_masks, naive_link_out["pred_masks"][:, topk_indices.to(to_store), ...]],
                                     dim=1)
            trc_queries = torch.cat([trc_queries, naive_link_out["pred_embds"][:, topk_indices, ...]], dim=1)
            naive_padding_masks = torch.ones(size=(num_left, num_frames)).to("cuda") < 0
            padding_masks = torch.cat([padding_masks, naive_padding_masks.unsqueeze(0)], dim=1)
            for ii in range(1, num_left + 1):
                seq_id_list.append((10000 + video_start_idx) * 10000 + ii * 1000)

        # logit_padding_tensor = torch.zeros(size=(self.tracker.num_classes + 1,), dtype=torch.float32, device="cuda")
        # logit_padding_tensor[-1] = 100.
        # online_logits = torch.cat([online_logits, logit_padding_tensor[None, None, :]], dim=1)
        # online_masks = torch.cat([online_masks, torch.ones(1, 1, num_frames, H, W).to(masks_list[-1]) * -1e4], dim=1)
        # trc_queries = torch.cat([trc_queries, torch.zeros(size=(1, 1, num_frames, C), dtype=torch.float32, device="cuda")], dim=1)
        # padding_masks = torch.cat([padding_masks, torch.zeros(size=(1, 1, num_frames), dtype=torch.bool, device="cuda")], dim=1)

        online_out = {"pred_logits": online_logits, "pred_masks": online_masks}

        common_out = {
            "frame_embeds": frame_embeds,
            "mask_features": mask_features,
            "online_out": online_out,
            "instance_embeds": trc_queries.permute(0, 3, 2, 1),  # (b, c, t, q)
            "padding_masks": padding_masks,
            "seq_id_list": seq_id_list,
            "dead_seq_id_list": dead_seq_id_list,
        }
        return common_out

    def run_window_inference(self, images_tensor, window_size=30, long_video_start_fidx=-1):
        common_out = self.common_inference(images_tensor, window_size, long_video_start_fidx, to_store="cpu")

        dead_seq_id_list = common_out["dead_seq_id_list"]
        for seq_id in dead_seq_id_list:  # for handling very long videos, saving memory
            self.tracker.video_ins_hub.pop(seq_id)

        frame_embeds = common_out["frame_embeds"]
        mask_features = common_out["mask_features"]
        B, C, T, Q = frame_embeds.shape
        H, W = mask_features.shape[-2:]

        instance_embeds = common_out["instance_embeds"]  # (b, c, t, q)
        padding_masks = common_out["padding_masks"]  # (b, q, t)
        seq_id_list = common_out["seq_id_list"]

        if instance_embeds.shape[-1] == 1:
            pred_cls = []
            pred_masks = []
            pred_ids = []
        else:
            out_dict = self.refiner(instance_embeds, padding_masks, frame_embeds, mask_features, None)
            pred_cls = out_dict["pred_logits"][:, 0, :, :]  # b, n, c
            pred_masks = out_dict["pred_masks"]
            pred_ids = torch.as_tensor(seq_id_list).to(torch.int64)[None, :]  # b, n
        outputs = {
            "pred_logits": pred_cls,
            "pred_masks": pred_masks,
            "pred_ids": pred_ids,
            "shape": (H, W),
        }

        return outputs

    def dvis_prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        gt_instances = []
        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_ids_per_video = []
            gt_classes_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                gt_ids_per_video.append(targets_per_frame.gt_ids[:, None])
                gt_classes_per_video.append(targets_per_frame.gt_classes[:, None])
                if isinstance(targets_per_frame.gt_masks, BitMasks):
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                else:  # polygon
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks

            gt_ids_per_video = torch.cat(gt_ids_per_video, dim=1)
            gt_classes_per_video = torch.cat(gt_classes_per_video, dim=1).max(dim=1)[0]
            valid_idx = (gt_ids_per_video != -1).any(dim=-1)

            gt_classes_per_video = gt_classes_per_video[valid_idx]  # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]  # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()  # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances

    def minvis_post_processing(self, outputs):
        pred_logits, pred_masks, pred_embds = outputs['pred_logits'], outputs['pred_masks'], outputs['pred_embds']

        pred_logits = pred_logits[0]
        pred_masks = einops.rearrange(pred_masks[0], 'q t h w -> t q h w')
        pred_embds = einops.rearrange(pred_embds[0], 'c t q -> t q c')

        pred_logits = list(torch.unbind(pred_logits))
        pred_masks = list(torch.unbind(pred_masks))
        pred_embds = list(torch.unbind(pred_embds))

        out_logits = []
        out_masks = []
        out_embds = []
        out_logits.append(pred_logits[0])
        out_masks.append(pred_masks[0])
        out_embds.append(pred_embds[0])

        # match the instances frame by frame
        for i in range(1, len(pred_logits)):
            indices = self.match_from_embds(out_embds[-1], pred_embds[i])

            out_logits.append(pred_logits[i][indices, :])
            out_masks.append(pred_masks[i][indices, :, :])
            out_embds.append(pred_embds[i][indices, :])

        out_logits = sum(out_logits) / len(out_logits)
        out_masks = torch.stack(out_masks, dim=1)  # q h w -> q t h w

        out_logits = out_logits.unsqueeze(0)
        out_masks = out_masks.unsqueeze(0)
        out_embds = torch.stack(out_embds, dim=1).unsqueeze(0)  # b, q, t, c

        outputs['pred_logits'] = out_logits
        outputs['pred_masks'] = out_masks
        outputs['pred_embds'] = out_embds

        return outputs

    def frame_decoder_loss_reshape(self, outputs, targets, image_outputs=None):
        # flatten the t frames as an image with size of (th, w)
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> b q () (t h) w')
        outputs['pred_logits'] = outputs['pred_logits'][:, 0, :, :]
        if image_outputs is not None:
            image_outputs['pred_masks'] = einops.rearrange(image_outputs['pred_masks'], 'b q t h w -> b q () (t h) w')

        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_masks'], 'b q t h w -> b q () (t h) w'
                )
                outputs['aux_outputs'][i]['pred_logits'] = outputs['aux_outputs'][i]['pred_logits'][:, 0, :, :]

        gt_instances = []
        for targets_per_video in targets:
            targets_per_video['masks'] = einops.rearrange(
                targets_per_video['masks'], 'q t h w -> q () (t h) w'
            )
            gt_instances.append(targets_per_video)
        return image_outputs, outputs, gt_instances


def mask_iou(mask1, mask2):
    mask1 = mask1.unsqueeze(1).char()  # n', 1, h, w
    mask2 = mask2.unsqueeze(0).char()  # 1, 20, h, w

    intersection = (mask1[:,:,:,:] * mask2[:,:,:,:]).sum(-1).sum(-1)  # n', 20
    union = (mask1[:,:,:,:] + mask2[:,:,:,:] - mask1[:,:,:,:] * mask2[:,:,:,:]).sum(-1).sum(-1)
    return (intersection + 1e-6) / (union + 1e-6)

