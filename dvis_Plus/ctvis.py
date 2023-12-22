import logging
from typing import Tuple
import einops

from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from mask2former_video.modeling.criterion import VideoSetCriterion
from mask2former_video.modeling.matcher import VideoHungarianMatcher
from mask2former.modeling.matcher import HungarianMatcher
from mask2former_video.utils.memory import retry_if_cuda_oom
from scipy.optimize import linear_sum_assignment
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from detectron2.config import configurable
from detectron2.structures import BitMasks
from detectron2.utils.registry import Registry


logger = logging.getLogger(__name__)

@META_ARCH_REGISTRY.register()
class CTMinVIS(nn.Module):
    """
    Copied from "https://github.com/NVlabs/MinVIS".
    """

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
        # video
        num_frames,
        window_inference,
        # ctvis
        image_matcher,
        cl_plugin,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames
        self.window_inference = window_inference

        self.image_matcher = image_matcher
        self.cl_plugin = cl_plugin

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = VideoHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        # for cl loss
        image_matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        cl_plugin = build_cl_plugin(cfg)  # train

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = VideoSetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE,
            # ctvis
            "image_matcher": image_matcher,
            "cl_plugin": cl_plugin,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def prepare_for_cl_plugin(self, outputs, targets):
        del outputs['aux_outputs'], outputs['pred_embds'], outputs['pred_embds_without_norm'], outputs['mask_features']
        for item in targets:
            item["masks"] = item["masks"].squeeze(1)
            item["ids"] = item["ids"].squeeze(1)
        outputs['pred_masks'] = outputs['pred_masks'].squeeze(2)
        outputs['pred_reid_embed'] = einops.rearrange(outputs['pred_reid_embed'], 'b c t q -> (b t) q c')
        return outputs, targets

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor, window_size=3)
        else:
            features = self.backbone(images.tensor)
            outputs = self.sem_seg_head(features)

        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)
            outputs, targets = self.frame_decoder_loss_reshape(outputs, targets)

            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            # for cl loss
            det_outputs, gt_instances = self.prepare_for_cl_plugin(outputs, targets)
            losses.update(self.cl_plugin.train_loss(
                det_outputs, gt_instances, self.image_matcher))
            return losses
        else:
            outputs = self.post_processing(outputs)

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]

            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video)(
                mask_cls_result,
                mask_pred_result,
                image_size,
                height,
                width,
                first_resize_size)

    def frame_decoder_loss_reshape(self, outputs, targets):
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
        outputs['pred_logits'] = einops.rearrange(outputs['pred_logits'], 'b t q c -> (b t) q c')
        if 'aux_outputs' in outputs:
            for i in range(len(outputs['aux_outputs'])):
                outputs['aux_outputs'][i]['pred_masks'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_masks'], 'b q t h w -> (b t) q () h w'
                )
                outputs['aux_outputs'][i]['pred_logits'] = einops.rearrange(
                    outputs['aux_outputs'][i]['pred_logits'], 'b t q c -> (b t) q c'
                )
        gt_instances = []
        for targets_per_video in targets:
            num_labeled_frames = targets_per_video['ids'].shape[1]
            for f in range(num_labeled_frames):
                labels = targets_per_video['labels']
                ids = targets_per_video['ids'][:, [f]]
                masks = targets_per_video['masks'][:, [f], :, :]
                gt_instances.append({"labels": labels, "ids": ids, "masks": masks})
        return outputs, gt_instances

    def match_from_embds(self, tgt_embds, cur_embds):
        cur_embds = cur_embds / cur_embds.norm(dim=1)[:, None]
        tgt_embds = tgt_embds / tgt_embds.norm(dim=1)[:, None]
        cos_sim = torch.mm(cur_embds, tgt_embds.transpose(0, 1))
        cost_embd = 1 - cos_sim
        C = 1.0 * cost_embd
        C = C.cpu()
        indices = linear_sum_assignment(C.transpose(0, 1))  # target x current
        indices = indices[1]  # permutation that makes current aligns to target
        return indices

    def post_processing(self, outputs):
        # pred_logits, pred_masks, pred_embds = outputs['pred_logits'], outputs['pred_masks'], outputs['pred_embds']
        pred_logits, pred_masks, pred_embds = outputs['pred_logits'], outputs['pred_masks'], outputs['pred_reid_embed']

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

        out_logits = sum(out_logits)/len(out_logits)
        out_masks = torch.stack(out_masks, dim=1)  # q h w -> q t h w

        out_logits = out_logits.unsqueeze(0)
        out_masks = out_masks.unsqueeze(0)

        outputs['pred_logits'] = out_logits
        outputs['pred_masks'] = out_masks

        return outputs

    def run_window_inference(self, images_tensor, window_size=30):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size

            features = self.backbone(images_tensor[start_idx:end_idx])
            out = self.sem_seg_head(features)
            del features['res2'], features['res3'], features['res4'], features['res5']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            del out['mask_features']
            out['pred_masks'] = out['pred_masks'].detach().cpu().to(torch.float32)
            out_list.append(out)

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1).to(torch.float32).detach().cpu()
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2)
        outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in out_list], dim=2).to(torch.float32).detach().cpu()
        outputs['pred_reid_embed'] = torch.cat([x['pred_reid_embed'] for x in out_list], dim=2).to(torch.float32).detach().cpu()

        return outputs

    def prepare_targets(self, targets, images):
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

    def inference_video(self, pred_cls, pred_masks, img_size, output_height, output_width, first_resize_size):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(
                self.sem_seg_head.num_classes,
                device=self.device
            ).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-10 predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // self.sem_seg_head.num_classes
            pred_masks = pred_masks[topk_indices]

            pred_masks = F.interpolate(
                pred_masks, size=first_resize_size, mode="bilinear", align_corners=False
            )

            pred_masks = pred_masks[:, :, : img_size[0], : img_size[1]]
            pred_masks = F.interpolate(
                pred_masks, size=(output_height, output_width), mode="bilinear", align_corners=False
            )

            masks = pred_masks > 0.

            out_scores = scores_per_image.tolist()
            out_labels = labels_per_image.tolist()
            out_masks = [m for m in masks.cpu()]
        else:
            out_scores = []
            out_labels = []
            out_masks = []

        video_output = {
            "image_size": (output_height, output_width),
            "pred_scores": out_scores,
            "pred_labels": out_labels,
            "pred_masks": out_masks,
        }

        return video_output

CL_PLUGIN_REGISTRY = Registry("CL_PLUGIN")
CL_PLUGIN_REGISTRY.__doc__ = """Registry for CL PLUGIN for Discriminative Representation Learning."""

def build_cl_plugin(cfg):
    name = cfg.MODEL.CL_PLUGIN.CL_PLUGIN_NAME
    return CL_PLUGIN_REGISTRY.get(name)(cfg)

class TrainTracklet(object):
    def __init__(self, instance_id, maximum_cache=10, momentum_embed=True, noise_embed=False):
        self.instance_id = instance_id
        self.logits = []
        self.masks = []
        self.reid_embeds = []
        self.negative_embeds = []
        self.long_scores = []
        self.frame_ids = []
        self.last_reid_embed = torch.zeros((256,), device='cuda')
        self.similarity_guided_reid_embed = None
        self.similarity_guided_reid_embed_list = []
        self.positive_embed_list = []
        self.exist_frames = 0
        self.maximum_cache = maximum_cache
        self.momentum = 0.75
        self.momentum_embed = momentum_embed
        self.noise_embed = noise_embed

    def update(self, positive_embed, negative_embed, frame_id=0):
        # update with noise
        if self.noise_embed and positive_embed is None:
            if np.random.rand() < 0.999:
                positive_embed = None
            else:
                index = random.randint(0, negative_embed.shape[0] - 1)
                positive_embed = negative_embed[index][None, ...]
        else:
            pass

        # if self.noise_embed and noise_embed is not None:
        #     positive_embed = noise_embed

        self.reid_embeds.append(positive_embed)
        self.negative_embeds.append(negative_embed)

        if positive_embed is not None:
            self.positive_embed_list.append(positive_embed)
            if self.exist_frames == 0:
                self.similarity_guided_reid_embed = positive_embed
                self.similarity_guided_reid_embed_list.append(self.similarity_guided_reid_embed)
            else:
                # Similarity-Guided Feature Fusion
                # https://arxiv.org/abs/2203.14208v1
                all_reid_embed = []
                for embedding in self.reid_embeds[:-1]:
                    if embedding is not None:
                        all_reid_embed.append(embedding)
                all_reid_embed = torch.cat(all_reid_embed, dim=0)

                similarity = torch.sum(torch.einsum("bc,c->b",
                                                    F.normalize(all_reid_embed, dim=-1),
                                                    F.normalize(positive_embed.squeeze(),
                                                                dim=-1))) / self.exist_frames  # noqa
                beta = max(0, similarity)
                self.similarity_guided_reid_embed = (
                                                                1 - beta) * self.similarity_guided_reid_embed + beta * positive_embed  # noqa
                self.similarity_guided_reid_embed_list.append(
                    self.similarity_guided_reid_embed)
            self.exist_frames += 1
        else:
            # no instance in the current frame
            self.similarity_guided_reid_embed_list.append(self.similarity_guided_reid_embed)

    def exist_before(self, frame_id):
        return frame_id != sum([1 if _ is None else 0 for _ in self.reid_embeds[:frame_id]])

    def exist_after(self, frame_id):
        return frame_id != sum([1 if _ is None else 0 for _ in self.reid_embeds[frame_id + 1:]])

    def get_positive_negative_embeddings(self, frame_id):
        anchor_embedding = self.reid_embeds[frame_id]
        positive_embedding = None
        if self.exist_before(frame_id):
            if self.momentum_embed and np.random.rand() > 0.5:
                positive_embedding = self.similarity_guided_reid_embed_list[frame_id - 1]
            else:
                for embedding in self.reid_embeds[:frame_id][::-1]:
                    if embedding is not None:
                        positive_embedding = embedding
                        break
        else:
            if self.exist_after(frame_id):
                for embedding in self.reid_embeds[frame_id + 1:]:
                    if embedding is not None:
                        positive_embedding = embedding
                        break
        negative_embedding = self.negative_embeds[frame_id - 1]

        return anchor_embedding, positive_embedding, negative_embedding

class SimpleTrainMemory:
    def __init__(self,
                 embed_type='temporally_weighted_softmax',
                 num_dead_frames=10,
                 maximum_cache=10,
                 momentum_embed=True,
                 noise_embed=False):
        self.tracklets = dict()
        self.num_tracklets = 0

        # last | temporally_weighted_softmax | momentum | similarity_guided
        self.embed_type = embed_type
        self.num_dead_frames = num_dead_frames
        self.maximum_cache = maximum_cache
        self.momentum_embed = momentum_embed
        self.noise_embed = noise_embed

    def update(self, instance_id, reid_embed, negative_embed):
        if instance_id not in self.exist_ids:
            self.tracklets[instance_id] = TrainTracklet(
                instance_id, self.maximum_cache, momentum_embed=self.momentum_embed, noise_embed=self.noise_embed)
            self.num_tracklets += 1
        self[instance_id].update(reid_embed, negative_embed)

    def __getitem__(self, instance_id):
        return self.tracklets[instance_id]

    def __len__(self):
        return self.num_tracklets

    def empty(self):
        self.tracklets = dict()
        self.num_tracklets = 0

    @property
    def exist_ids(self):
        return self.tracklets.keys()

    def valid(self, instance_id, frame_id):
        return self[instance_id].reid_embeds[frame_id] is not None

    def exist_reid_embeds(self):
        memory_bank_embeds = []
        memory_bank_ids = []

        for instance_id, tracklet in self.tracklets.items():
            memory_bank_embeds.append(tracklet.similarity_guided_reid_embed)

            memory_bank_ids.append(instance_id)

        memory_bank_embeds = torch.stack(memory_bank_embeds, dim=0)
        memory_bank_ids = memory_bank_embeds.new_tensor(memory_bank_ids).to(dtype=torch.long)

        return memory_bank_ids, memory_bank_embeds

    def get_training_samples(self, instance_id, frame_id):
        anchor_embedding, positive_embedding, negative_embedding = self[instance_id].get_positive_negative_embeddings(
            frame_id)  # noqa

        return anchor_embedding, positive_embedding, negative_embedding


@CL_PLUGIN_REGISTRY.register()
class CTCLPlugin(nn.Module):
    @configurable
    def __init__(self,
                 *,
                 weight_dict,
                 num_negatives,
                 sampling_frame_num,
                 bio_cl,
                 momentum_embed,
                 noise_embed):
        super().__init__()
        self.weight_dict = weight_dict
        self.num_negatives = num_negatives
        self.sampling_frame_num = sampling_frame_num

        self.bio_cl = bio_cl
        self.momentum_embed = momentum_embed
        self.noise_embed = noise_embed
        self.train_memory_bank = SimpleTrainMemory(momentum_embed=self.momentum_embed, noise_embed=self.noise_embed)

    @classmethod
    def from_config(cls, cfg):
        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM

        reid_weight = cfg.MODEL.CL_PLUGIN.REID_WEIGHT
        aux_reid_weight = cfg.MODEL.CL_PLUGIN.AUX_REID_WEIGHT

        weight_dict = {"loss_reid": reid_weight,
                       "loss_aux_reid": aux_reid_weight}

        num_negatives = cfg.MODEL.CL_PLUGIN.NUM_NEGATIVES

        bio_cl = cfg.MODEL.CL_PLUGIN.BIO_CL

        momentum_embed = cfg.MODEL.CL_PLUGIN.MOMENTUM_EMBED
        noise_embed = cfg.MODEL.CL_PLUGIN.NOISE_EMBED

        ret = {"weight_dict": weight_dict,
               "num_negatives": num_negatives,
               "sampling_frame_num": sampling_frame_num,
               "bio_cl": bio_cl,
               "momentum_embed": momentum_embed,
               "noise_embed": noise_embed}
        return ret

    @property
    def device(self):
        return torch.device('cuda')

    def get_key_ref_outputs(self, det_outputs):
        outputs_keys = det_outputs.keys()  # noqa
        outputs_list = [dict() for _ in range(self.sampling_frame_num)]

        num_images = det_outputs['pred_logits'].shape[0]
        index_list = []
        for i in range(self.sampling_frame_num):
            index_list.append(torch.arange(
                i, num_images, step=self.sampling_frame_num, device=self.device))

        for key in outputs_keys:
            if key in ['aux_outputs', 'interm_outputs']:
                pass
            else:
                for i in range(self.sampling_frame_num):
                    outputs_list[i][key] = det_outputs[key][index_list[i]]
        # outputs_list, [per frame bs output, ...], len is sampling frames
        # per frame bs output, dict, e.g. 'mask' is (b, q, h, w)
        return outputs_list

    def train_loss(self, det_outputs, gt_instances, matcher):
        targets_list = self.prepare_targets(gt_instances)
        outputs_list = self.get_key_ref_outputs(det_outputs)

        indices_list = []
        for i in range(self.sampling_frame_num):
            # perform per frame matching indices
            outputs = outputs_list[i]
            targets = targets_list[i]
            outputs_without_aux = {k: v for k,
                                            v in outputs.items() if k != "aux_outputs"}
            # [matched_row, matched_colum]
            indices = matcher(outputs_without_aux, targets)
            indices_list.append(indices)

        losses = dict()

        losses.update(self.get_reid_loss(targets_list, outputs_list, indices_list))

        for k in list(losses.keys()):
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]
            else:
                losses.pop(k)
        return losses

    def get_reid_loss(self, targets_list, outputs_list, indices_list):
        contrastive_items = []

        batch_size = len(targets_list[0])
        for i in range(batch_size):  # per batch
            # empty memory bank
            self.train_memory_bank.empty()
            indice_list = [indices[i] for indices in indices_list]
            target_list = [targets[i] for targets in targets_list]

            gt2query_id_list = [indice[0][torch.sort(
                indice[1])[1]] for indice in indice_list]

            reid_embedding_list = [outputs[f'pred_reid_embed'][i]
                                   for outputs in outputs_list]
            num_instances = target_list[0]['valid'].shape[0]  # num of instances in this frame

            # Step 1: Store and update the embeddings into memory bank first
            for j in range(self.sampling_frame_num):
                anchor_embeddings = reid_embedding_list[j]
                anchor_target = target_list[j]

                for instance_i in range(num_instances):
                    if anchor_target['valid'][instance_i]:  # instance exists
                        anchor_query_id = gt2query_id_list[j][instance_i]  # query id
                        anchor_embedding = anchor_embeddings[anchor_query_id][None, ...]

                        negative_query_id = sorted(
                            random.sample(set(range(self.num_negatives + 1)) - set([anchor_query_id.item()]),
                                          self.num_negatives))  # noqa
                        negative_embedding = anchor_embeddings[negative_query_id]
                    else:  # not exists
                        anchor_embedding = None
                        negative_embedding = anchor_embeddings

                    self.train_memory_bank.update(
                        instance_i, anchor_embedding, negative_embedding)  # update the memory bank

            # Step 2: build contrastive items frame by frame
            for frame_id in range(self.sampling_frame_num):
                if frame_id == 0:
                    continue
                else:
                    # query -> memory_bank
                    for instance_i in range(num_instances):
                        if self.train_memory_bank.valid(instance_i, frame_id):
                            anchor_embedding, positive_embedding, negative_embedding = self.train_memory_bank.get_training_samples(
                                instance_i, frame_id)  # noqa
                            if positive_embedding is None:
                                # No valid positive embedding
                                continue
                            num_positive = positive_embedding.shape[0]

                            pos_neg_embedding = torch.cat([positive_embedding, negative_embedding], dim=0)

                            pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                                                                        dtype=torch.int64)  # noqa
                            pos_neg_label[:num_positive] = 1.

                            # dot product
                            dot_product = torch.einsum('ac,kc->ak', [pos_neg_embedding, anchor_embedding])
                            aux_normalize_pos_neg_embedding = nn.functional.normalize(pos_neg_embedding, dim=1)
                            aux_normalize_anchor_embedding = nn.functional.normalize(anchor_embedding, dim=1)

                            aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                               aux_normalize_anchor_embedding])
                            contrastive_items.append({
                                'dot_product': dot_product,
                                'cosine_similarity': aux_cosine_similarity,
                                'label': pos_neg_label})

        # we follow the losses in IDOL
        losses = loss_reid(contrastive_items, outputs_list[0])

        return losses

    def prepare_targets(self, targets):
        # prepare for track part
        # process per image targets
        for targets_per_image in targets:

            inst_ids = targets_per_image["ids"]
            valid_id = inst_ids != -1  # if an object is disappeared，its gt_ids is -1
            targets_per_image.update({'inst_id': inst_ids, 'valid': valid_id})

        new_targets = targets
        bz = len(new_targets) // self.sampling_frame_num
        ids_list = []
        # get image ids for per time frame, (bz, )
        for i in range(self.sampling_frame_num):
            ids_list.append(
                list(range(i, bz * self.sampling_frame_num, self.sampling_frame_num)))

        targets_list = []
        for i in range(self.sampling_frame_num):
            targets_list.append([new_targets[j] for j in ids_list[i]])

        # [per bz frame gt, ...], len is sampling feames
        # per bz frame gt, [per image gt, ...], len is bz
        return targets_list


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size() -> int:
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def loss_reid(qd_items, outputs, reduce=False):
    contras_loss = 0
    aux_loss = 0

    num_qd_items = len(qd_items)
    if reduce:  # it seems worse when reduce is True
        num_qd_items = torch.as_tensor(
            [num_qd_items], dtype=torch.float, device=outputs['pred_reid_embed'].device)

        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_qd_items)
        num_qd_items = torch.clamp(
            num_qd_items / get_world_size(), min=1).item()

    if len(qd_items) == 0:
        losses = {'loss_reid': outputs['pred_reid_embed'].sum() * 0,
                  'loss_aux_reid': outputs['pred_reid_embed'].sum() * 0}
        return losses

    for qd_item in qd_items:
        pred = qd_item['dot_product'].permute(1, 0)
        label = qd_item['label'].unsqueeze(0)
        # contrastive loss
        pos_inds = (label == 1)
        neg_inds = (label == 0)
        pred_pos = pred * pos_inds.float()
        pred_neg = pred * neg_inds.float()
        # use -inf to mask out unwanted elements.
        pred_pos[neg_inds] = pred_pos[neg_inds] + float('inf')
        pred_neg[pos_inds] = pred_neg[pos_inds] + float('-inf')

        _pos_expand = torch.repeat_interleave(pred_pos, pred.shape[1], dim=1)
        _neg_expand = pred_neg.repeat(1, pred.shape[1])
        # [bz,N], N is all pos and negative samples on reference frame, label indicate it's pos or negative
        x = torch.nn.functional.pad((_neg_expand - _pos_expand), (0, 1), "constant", 0)
        contras_loss += torch.logsumexp(x, dim=1)

        aux_pred = qd_item['cosine_similarity'].permute(1, 0)
        aux_label = qd_item['label'].unsqueeze(0)

        aux_loss += (torch.abs(aux_pred - aux_label) ** 2).mean()

    losses = {'loss_reid': contras_loss.sum() / num_qd_items,
              'loss_aux_reid': aux_loss / num_qd_items}
    return losses