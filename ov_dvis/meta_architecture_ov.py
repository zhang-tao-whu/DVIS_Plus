import logging
from typing import Tuple
import einops

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks

from mask2former_video.modeling.criterion import VideoSetCriterion_ov
from mask2former_video.modeling.matcher import VideoHungarianMatcher, VideoHungarianMatcher_Consistent
from mask2former_video.utils.memory import retry_if_cuda_oom

from scipy.optimize import linear_sum_assignment

from .video_dvis_modules_ov import ReferringTracker_noiser_OV, TemporalRefiner_OV
from .video_mask2former_transformer_decoder_ov import MaskPooling
from dvis_Plus.utils import loss_reid

logger = logging.getLogger(__name__)

VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]

def get_classification_logits(x, text_classifier, logit_scale, num_templates=None):
    x = F.normalize(x, dim=-1)
    logit_scale = torch.clamp(logit_scale.exp(), max=100)
    pred_logits = logit_scale * x @ text_classifier.T # B, *, N + 1
    # max ensembel as in OpenSeg/ODISE
    final_pred_logits = []
    cur_idx = 0
    for num_t in num_templates[:-1]:
        final_pred_logits.append(pred_logits[:, :, cur_idx: cur_idx + num_t].max(-1).values)
        cur_idx += num_t
    # final_pred_logits.append(pred_logits[:, :, -1]) # the last classifier is for void
    final_pred_logits.append(pred_logits[:, :, -num_templates[-1]:].max(-1).values)
    final_pred_logits = torch.stack(final_pred_logits, dim=-1)
    return final_pred_logits

@META_ARCH_REGISTRY.register()
class MinVIS_OV(nn.Module):
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
        train_metadatas: dict,
        test_metadatas: dict,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        num_frames,
        window_inference,
        # fc-clip
        geometric_ensemble_alpha: float,
        geometric_ensemble_beta: float,
        ensemble_on_valid_mask: bool,
        # multi datasets
        test2train={},
        task='vis',
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
            test2train: dict, which void embedding to use
        """
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = train_metadatas
        self.test_metadata = test_metadatas
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.num_frames = num_frames
        self.window_inference = window_inference

        # FC-CLIP args
        self.mask_pooling = MaskPooling()
        self.geometric_ensemble_alpha = geometric_ensemble_alpha
        self.geometric_ensemble_beta = geometric_ensemble_beta
        self.ensemble_on_valid_mask = ensemble_on_valid_mask

        self.train_text_classifier = None
        self.test_text_classifier = None
        self.train_num_templates = None
        self.test_num_templates = None
        self.category_overlapping_mask = None
        self.train_text_classifier_dict = {}
        self.test_text_classifier_dict = {}
        self.train_num_templates_dict = {}
        self.test_num_templates_dict = {}
        self.test_num_templates_dict = {}

        self.void_embedding = nn.Embedding(1, backbone.dim_latent)  # use this for void
        # init private void embedding for each dataset
        if len(train_metadatas) - 1 > 0:
            self.additional_void_embedding = nn.Embedding(len(train_metadatas) - 1, backbone.dim_latent)
        else:
            self.additional_void_embedding = None

        self.train_class_prepares = {}
        self.train_names2id = {}
        self.test_class_prepares = {}
        for i, name in enumerate(train_metadatas.keys()):
            self.train_names2id[name] = i
            train_metadata = train_metadatas[name]
            _, train_num_templates, train_class_names = self.prepare_class_names_from_metadata(train_metadata,
                                                                                               train_metadata)
            self.train_class_prepares.update({name: {'num_templates': train_num_templates,
                                                     'class_names': train_class_names}})
        all_train_metadatas = [train_metadatas[key] for key in train_metadatas.keys()]
        self.all_train_metadatas = all_train_metadatas
        for name in test_metadatas.keys():
            test_metadata = test_metadatas[name]
            category_overlapping_mask, test_num_templates, test_class_names = self.prepare_class_names_from_metadata(
                test_metadata, all_train_metadatas)
            self.test_class_prepares.update({name: {'overlapping': category_overlapping_mask,
                                                    'num_templates': test_num_templates,
                                                    'class_names': test_class_names}})

        self.test2train = test2train
        self.test_use_all_vocabulary = False
        self.void_embedding_merge_mode = 'coco'  # ['mean', 'max', 'coco']

        self.task = task
        assert self.task in ['vis', 'vss', 'vps'], "Only support vis, vss and vps !"
        inference_dict = {
            'vis': self.inference_video_vis,
            'vss': self.inference_video_vss,
            'vps': self.inference_video_vps,
        }
        self.inference_video = inference_dict[self.task]

    def get_text_classifier_with_void(self, text_classifier, num_templates, name):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',')  # there can be multiple synonyms for single class
                res.append(x_)
            return res
        if self.training or not self.test_use_all_vocabulary:
            if self.additional_void_embedding is None:
                _zero = self.void_embedding.weight.sum() * 0.0
            else:
                _zero = self.void_embedding.weight.sum() * 0.0 + self.additional_void_embedding.weight.sum() * 0.0
            if name in self.train_names2id.keys():
                i = self.train_names2id[name]
                if i == 0:
                    void_embed = self.void_embedding.weight
                else:
                    void_embed = self.additional_void_embedding.weight[i - 1: i]
                void_embed = F.normalize(void_embed, dim=-1) + _zero
            else:
                if self.additional_void_embedding is None:
                    void_embed = self.void_embedding.weight
                    void_embed = F.normalize(void_embed, dim=-1)
                else:
                    void_embed = torch.cat([self.void_embedding.weight, self.additional_void_embedding.weight], dim=0)
                    void_embed = F.normalize(void_embed, dim=-1).detach()
                    if self.void_embedding_merge_mode == 'mean':
                        void_embed = torch.mean(void_embed, dim=0, keepdim=True)
                    elif self.void_embedding_merge_mode == 'max':
                        pass
                    elif self.void_embedding_merge_mode == 'coco':
                        void_embed = void_embed[:1]
                    else:
                        raise NotImplementedError
            text_classifier = torch.cat([text_classifier, void_embed], dim=0)
            num_templates = num_templates + [void_embed.shape[0]]
            return text_classifier, num_templates
        else:
            # print("using additional vocabulary !!!")
            class_names = split_labels(self.test_metadata[name].classes_ov)  # it includes both thing and stuff
            if isinstance(self.all_train_metadatas, list):
                train_classes = []
                for item in self.all_train_metadatas:
                    train_classes += item.classes_ov
                if len(train_classes) != 0:
                    train_class_names = split_labels(train_classes)
                else:
                    raise NotImplementedError
            else:
                train_class_names = split_labels(self.all_train_metadatas.classes_ov)
            test_class_names = {l for label in class_names for l in label}
            # train_class_names = {l for label in train_class_names for l in label}
            train2test_category_overlapping_list = []
            for train_class_name in train_class_names:
                not_overlapping = set(train_class_name).isdisjoint(set(test_class_names))
                train2test_category_overlapping_list.extend([not_overlapping] * len(train_class_name))
            train2test_category_overlapping_list = torch.tensor(
                train2test_category_overlapping_list, dtype=torch.bool)

            train_classifiers = []
            for key in self.metadata.keys():
                if key not in self.train_text_classifier_dict.keys():
                    self._set_class_information(key, train=True)
                train_classifiers.append(self.train_text_classifier_dict[key])

            train_classifiers = torch.cat(train_classifiers, dim=0)[train2test_category_overlapping_list]

            if name in self.test2train.keys():
                i = self.train_names2id[self.test2train[name]]
                if i == 0:
                    void_embed = self.void_embedding.weight
                else:
                    void_embed = self.additional_void_embedding.weight[i - 1: i]
                void_embed = F.normalize(void_embed, dim=-1)
            else:
                if self.additional_void_embedding is not None:
                    void_embed = torch.cat([self.void_embedding.weight, self.additional_void_embedding.weight], dim=0)
                    void_embed = F.normalize(void_embed, dim=-1).detach()
                    if self.void_embedding_merge_mode == 'mean':
                        void_embed = torch.mean(void_embed, dim=0, keepdim=True)
                    elif self.void_embedding_merge_mode == 'max':
                        pass
                    elif self.void_embedding_merge_mode == 'coco':
                        void_embed = void_embed[:1]
                    else:
                        raise NotImplementedError
                else:
                    void_embed = self.void_embedding.weight
                    void_embed = F.normalize(void_embed, dim=-1)
            text_classifier = torch.cat([text_classifier, void_embed, train_classifiers], dim=0)
            num_templates = num_templates + [len(void_embed) + len(train_classifiers)]
            return text_classifier, num_templates

    def _set_class_information(self, name, train=True):
        self.name = name
        if train:
            if name in self.train_text_classifier_dict.keys():
                return self.train_text_classifier_dict[name], self.train_num_templates_dict[name]
            else:
                infos = self.train_class_prepares[name]
                self.train_num_templates = infos['num_templates']
                self.train_class_names = infos['class_names']
                self.train_text_classifier = None
                self.train_text_classifier, self.train_num_templates = self.get_text_classifier(train=train)
                self.train_text_classifier_dict[name] = self.train_text_classifier
                self.train_num_templates_dict[name] = self.train_num_templates
                return self.train_text_classifier, self.train_num_templates
        else:
            self.category_overlapping_mask = self.test_class_prepares[name]['overlapping']
            if name in self.test_text_classifier_dict.keys():
                return self.test_text_classifier_dict[name], self.test_num_templates_dict[name]
            infos = self.test_class_prepares[name]
            self.category_overlapping_mask = infos['overlapping']
            self.test_num_templates = infos['num_templates']
            self.test_class_names = infos['class_names']
            self.test_text_classifier = None
            self.test_text_classifier, self.test_num_templates = self.get_text_classifier(train=train)
            self.test_text_classifier_dict[name] = self.test_text_classifier
            self.test_num_templates_dict[name] = self.test_num_templates
            return self.test_text_classifier, self.test_num_templates

    def set_metadata(self, name, metadata):
        print(metadata.classes_ov)
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = \
            self.prepare_class_names_from_metadata(metadata, self.all_train_metadatas)
        self.test_class_prepares.update({name: {'overlapping': self.category_overlapping_mask,
                                                'num_templates': self.test_num_templates,
                                                'class_names': self.test_class_names}})
        if name in self.test_text_classifier_dict.keys():
            del self.test_text_classifier_dict[name]
        self.test_text_classifier = None
        return

    def get_text_classifier(self, train=False):
        if self.training or train:
            if self.train_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier.append(self.backbone.get_text_classifier(self.train_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)
                # get per text embedding for per class template

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.train_text_classifier = text_classifier
            # self.train_text_classifier, per component templates
            # self.train_num_templates, per class have how many components
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.test_class_names), bs):
                    text_classifier.append(self.backbone.get_text_classifier(self.test_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.test_text_classifier = text_classifier
            return self.test_text_classifier, self.test_num_templates

    def prepare_class_names_from_metadata(self, metadata, train_metadata):
        def split_labels(x):
            res = []
            for x_ in x:
                x_ = x_.replace(', ', ',')
                x_ = x_.split(',')  # there can be multiple synonyms for single class
                res.append(x_)
            return res

        # get text classifier
        try:
            class_names = split_labels(metadata.classes_ov)  # it includes both thing and stuff
            if isinstance(train_metadata, list):
                train_classes = []
                for item in train_metadata:
                    train_classes += item.classes_ov
                if len(train_classes) != 0:
                    train_class_names = split_labels(train_classes)
                else:
                    raise NotImplementedError
            else:
                train_class_names = split_labels(train_metadata.classes_ov)
        except:
            # this could be for insseg, where only thing_classes are available
            class_names = split_labels(metadata.thing_classes_ov)
            if isinstance(train_metadata, list):
                train_thing_classes = []
                for item in train_metadata:
                    train_thing_classes += item.thing_classes_ov
                train_class_names = split_labels(train_thing_classes)
            else:
                train_class_names = split_labels(train_metadata.thing_classes_ov)
        train_class_names = {l for label in train_class_names for l in label}
        category_overlapping_list = []
        for test_class_names in class_names:
            is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names))
            category_overlapping_list.append(is_overlapping)
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long)

        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)

        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num)  # how many templates for current classes
        class_names = templated_class_names
        # category_overlapping_mask (N_train, )
        # num_templates, [num_per_class_name, ], num of cur class is splited to how many components
        # class_names, [per_class_template, ], per_class_template [N_comp * N_template]
        return category_overlapping_mask, num_templates, class_names

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

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = VideoSetCriterion_ov(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        train_metadatas = {}
        test_metadatas = {}
        for name in cfg.DATASETS.TRAIN:
            train_metadatas[name] = MetadataCatalog.get(name)
        for name in cfg.DATASETS.TEST:
            test_metadatas[name] = MetadataCatalog.get(name)
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "train_metadatas": train_metadatas,
            "test_metadatas": test_metadatas,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE,
            "task": cfg.MODEL.MASK_FORMER.TEST.TASK,
            # fc clip
            "geometric_ensemble_alpha": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_ALPHA,
            "geometric_ensemble_beta": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_BETA,
            "ensemble_on_valid_mask": cfg.MODEL.FC_CLIP.ENSEMBLE_ON_VALID_MASK,
            # multi datasets
            "test2train": {x: y for x, y in zip(cfg.DATASETS.TEST, cfg.DATASETS.TEST2TRAIN)},
        }

    @property
    def device(self):
        return self.pixel_mean.device

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
        name = batched_inputs[0]['name']
        for batched_input in batched_inputs:
            assert name == batched_input['name']

        # print(batched_inputs)
        images = []
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)

        text_classifier, num_templates = self._set_class_information(batched_inputs[0]['name'], self.training)
        # Append void class weight
        text_classifier, num_templates = self.get_text_classifier_with_void(text_classifier, num_templates,
                                                                            name=batched_inputs[0]['name'])

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor, window_size=3,
                                                text_classifier=text_classifier, num_templates=num_templates)
        else:
            features = self.backbone(images.tensor)
            features['text_classifier'] = text_classifier
            features['num_templates'] = num_templates
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
            return losses
        else:
            # when inference, bs must be 1
            mask_cls_results = outputs["pred_logits"][0]  # t q c
            mask_pred_results = outputs["pred_masks"][0].transpose(0, 1)  # t q h w

            # We ensemble the pred logits of in-vocab and out-vocab
            if "clip_vis_dense" in outputs.keys():
                clip_feature = outputs["clip_vis_dense"]
            else:
                clip_feature = features["clip_vis_dense"]
            mask_for_pooling = F.interpolate(mask_pred_results, size=clip_feature.shape[-2:],
                                             mode='bilinear', align_corners=False)

            if "convnext" in self.backbone.model_name.lower():
                pooled_clip_feature = self.mask_pooling(clip_feature, mask_for_pooling)
                pooled_clip_feature = self.backbone.visual_prediction_forward(pooled_clip_feature)
            elif "rn" in self.backbone.model_name.lower():
                try:
                    pooled_clip_feature = self.backbone.visual_prediction_forward(clip_feature,
                                                                                  mask_for_pooling)  # (t, q, c)
                except:
                    pooled_clip_feature = []
                    _windows_size = 16
                    iters = len(mask_for_pooling) // _windows_size
                    if len(mask_for_pooling) % _windows_size != 0:
                        iters += 1
                    for i in range(iters):
                        start_idx = i * _windows_size
                        end_idx = (i + 1) * _windows_size
                        pooled_clip_feature.append(self.backbone.visual_prediction_forward(
                            clip_feature[start_idx:end_idx].to(self.device),
                            mask_for_pooling[start_idx:end_idx].to(self.device)))
                    pooled_clip_feature = torch.cat(pooled_clip_feature, dim=0)

            else:
                raise NotImplementedError

            out_vocab_cls_results = get_classification_logits(pooled_clip_feature, text_classifier,
                                                              self.backbone.clip_model.logit_scale, num_templates)
            in_vocab_cls_results = mask_cls_results[..., :-1]  # remove void
            out_vocab_cls_results = out_vocab_cls_results[..., :-1]  # remove void

            # Reference: https://github.com/NVlabs/ODISE/blob/main/odise/modeling/meta_arch/odise.py#L1506
            out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
            in_vocab_cls_results = in_vocab_cls_results.softmax(-1)
            category_overlapping_mask = self.category_overlapping_mask.to(self.device)

            if self.ensemble_on_valid_mask:
                # Only include out_vocab cls results on masks with valid pixels
                # We empirically find that this is important to obtain reasonable AP/mIOU score with ResNet CLIP models
                valid_masking = (mask_for_pooling > 0).to(mask_for_pooling).sum(-1).sum(-1) > 0
                valid_masking = valid_masking.to(in_vocab_cls_results.dtype).unsqueeze(-1)
                alpha = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_alpha
                beta = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_beta
                alpha = alpha * valid_masking
                beta = beta * valid_masking
            else:
                alpha = self.geometric_ensemble_alpha
                beta = self.geometric_ensemble_beta

            cls_logits_seen = (
                    (in_vocab_cls_results ** (1 - alpha) * out_vocab_cls_probs ** alpha).log()
                    * category_overlapping_mask
            )
            cls_logits_unseen = (
                    (in_vocab_cls_results ** (1 - beta) * out_vocab_cls_probs ** beta).log()
                    * (1 - category_overlapping_mask)
            )
            cls_results = cls_logits_seen + cls_logits_unseen

            # This is used to filtering void predictions.
            is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
            mask_cls_probs = torch.cat([
                cls_results.softmax(-1) * (1.0 - is_void_prob),
                is_void_prob], dim=-1)
            mask_cls_results = torch.log(mask_cls_probs + 1e-8)
            outputs["pred_logits"][0] = mask_cls_results  # t q c

            # for minvis
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

        out_logits = sum(out_logits)/len(out_logits)
        out_masks = torch.stack(out_masks, dim=1)  # q h w -> q t h w

        out_logits = out_logits.unsqueeze(0)
        out_masks = out_masks.unsqueeze(0)

        outputs['pred_logits'] = out_logits
        outputs['pred_masks'] = out_masks

        return outputs

    def run_window_inference(self, images_tensor, window_size=30, text_classifier=None, num_templates=None):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size

            features = self.backbone(images_tensor[start_idx:end_idx])
            features['text_classifier'] = text_classifier
            features['num_templates'] = num_templates
            out = self.sem_seg_head(features)
            del features['res2'], features['res3'], features['res4'], features['res5']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            # out['pred_masks'] = out['pred_masks'].detach().cpu().to(torch.float32)
            out['pred_masks'] = out['pred_masks'].detach()
            out['clip_vis_dense'] = features['clip_vis_dense']
            out_list.append(out)

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1).detach()
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2).detach()
        outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in out_list], dim=2).detach()
        outputs['clip_vis_dense'] = torch.cat([x['clip_vis_dense'] for x in out_list], dim=0).detach()

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

            gt_classes_per_video = gt_classes_per_video[valid_idx]          # N,
            gt_ids_per_video = gt_ids_per_video[valid_idx]                  # N, num_frames

            gt_instances.append({"labels": gt_classes_per_video, "ids": gt_ids_per_video})
            gt_masks_per_video = gt_masks_per_video[valid_idx].float()          # N, num_frames, H, W
            gt_instances[-1].update({"masks": gt_masks_per_video})

        return gt_instances

    def inference_video_vis(
        self, pred_cls, pred_masks, img_size, output_height, output_width, first_resize_size,
    ):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            labels = torch.arange(
                # self.sem_seg_head.num_classes, device=self.device
                pred_cls.shape[-1] - 1, device=self.device
            ).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-K predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(10, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // (pred_cls.shape[-1] - 1)
            pred_masks = pred_masks[topk_indices]

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
            "task": "vis",
        }

        return video_output

    def inference_video_vps(
        self, pred_cls, pred_masks, img_size, output_height, output_width, first_resize_size,
    ):
        pred_cls = F.softmax(pred_cls, dim=-1)
        mask_pred = pred_masks
        scores, labels = pred_cls.max(-1)

        # filter out the background prediction
        keep = labels.ne(pred_cls.shape[-1] - 1) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]

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
        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask
            return {
                "image_size": (output_height, output_width),
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "task": "vps",
            }
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)  # (t, h, w)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class < len(self.test_metadata[self.name].thing_dataset_id_to_contiguous_id)
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
            return {
                "image_size": (output_height, output_width),
                "pred_masks": panoptic_seg.cpu(),
                "segments_infos": segments_infos,
                "task": "vps",
            }

    def inference_video_vss(
        self, pred_cls, pred_masks, img_size, output_height, output_width, first_resize_size,
    ):
        mask_cls = F.softmax(pred_cls, dim=-1)[..., :-1]
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
class DVIS_online_OV(MinVIS_OV):
    """
    Online version of DVIS, including a segmenter and a referring tracker.
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
        train_metadatas: dict,
        test_metadatas: dict,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        tracker,
        num_frames,
        window_inference,
        max_num,
        max_iter_num,
        window_size,
        task,
        # fc-clip
        geometric_ensemble_alpha: float,
        geometric_ensemble_beta: float,
        ensemble_on_valid_mask: bool,
        # multi datasets
        test2train={},
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
            # video
            tracker: a tracker module, e.g. ReferringTracker
            num_frames: number of frames sampled during training
            window_inference: if the GPU memory is insufficient to predict the entire video at
                once, inference needs to be performed clip by clip
            num_class: the categories number of the dataset
            max_num: the maximum number of instances retained for a video, only used in VIS
            max_iter_num: the iter nums
            window_size: the number of images processed by the segmenter at a time
            task: VIS, VSS or VPS
        """
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            train_metadatas=train_metadatas,
            test_metadatas=test_metadatas,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            # video
            num_frames=num_frames,
            window_inference=window_inference,
            # dc clip
            geometric_ensemble_alpha=geometric_ensemble_alpha,
            geometric_ensemble_beta=geometric_ensemble_beta,
            ensemble_on_valid_mask=ensemble_on_valid_mask,
            # multi datasets
            test2train=test2train,
        )
        # frozen the void classifier
        for p in self.void_embedding.parameters():
            p.requires_grad_(False)
        # frozen the segmenter
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        for p in self.sem_seg_head.parameters():
            p.requires_grad_(False)

        self.tracker = tracker
        self.max_num = max_num
        self.iter = 0
        self.max_iter_num = max_iter_num

        self.window_size = window_size
        self.task = task
        assert self.task in ['vis', 'vss', 'vps'], "Only support vis, vss and vps !"
        inference_dict = {
            'vis': self.inference_video_vis,
            'vss': self.inference_video_vss,
            'vps': self.inference_video_vps,
        }
        self.inference_video_task = inference_dict[self.task]

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
        matcher = VideoHungarianMatcher_Consistent(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            frames=cfg.INPUT.SAMPLING_FRAME_NUM
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        if cfg.MODEL.TRACKER.USE_CL:
            weight_dict.update({'loss_reid': 2})

        losses = ["labels", "masks"]

        criterion = VideoSetCriterion_ov(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        tracker = ReferringTracker_noiser_OV(
            hidden_channel=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            feedforward_channel=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            num_head=cfg.MODEL.MASK_FORMER.NHEADS,
            decoder_layer_num=cfg.MODEL.TRACKER.DECODER_LAYERS,
            noise_mode=cfg.MODEL.TRACKER.NOISE_MODE,
            mask_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            mask_pooling=sem_seg_head.predictor.mask_pooling,
            mask_pooling_proj=sem_seg_head.predictor._mask_pooling_proj,
            class_embed=sem_seg_head.predictor.class_embed,
            logit_scale=sem_seg_head.predictor.logit_scale,
            mask_embed=sem_seg_head.predictor.mask_embed,
            decoder_norm=sem_seg_head.predictor.decoder_norm,
        )

        max_iter_num = cfg.SOLVER.MAX_ITER

        train_metadatas = {}
        test_metadatas = {}
        for name in cfg.DATASETS.TRAIN:
            train_metadatas[name] = MetadataCatalog.get(name)
        for name in cfg.DATASETS.TEST:
            test_metadatas[name] = MetadataCatalog.get(name)

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "train_metadatas": train_metadatas,
            "test_metadatas": test_metadatas,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "tracker": tracker,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE,
            "max_num": cfg.MODEL.MASK_FORMER.TEST.MAX_NUM,
            "max_iter_num": max_iter_num,
            "window_size": cfg.MODEL.MASK_FORMER.TEST.WINDOW_SIZE,
            "task": cfg.MODEL.MASK_FORMER.TEST.TASK,
            # fc clip
            "geometric_ensemble_alpha": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_ALPHA,
            "geometric_ensemble_beta": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_BETA,
            "ensemble_on_valid_mask": cfg.MODEL.FC_CLIP.ENSEMBLE_ON_VALID_MASK,
            # multi datasets
            "test2train": {x: y for x, y in zip(cfg.DATASETS.TEST, cfg.DATASETS.TEST2TRAIN)},
        }

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
            dict:
                For specific task, the dict contains the following keys:
                * For VIS:
                    "image_size": (output_height, output_width).
                    "pred_scores": score for per instance.
                    "pred_labels": class for per instance.
                    "pred_masks": list[Tensor], bit-masks for per instance, Tensor shape is (t, h, w).
                    "pred_ids": list, query ids for per instance, list length is N.
                    "task": "vis",
                * For VSS:
                    "image_size": (output_height, output_width).
                    "pred_masks": A Tensor that represents the
                        per-pixel segmentation prediced by the head.
                        The prediction has shape (t, h, w) that represents
                        the category ID for each pixel.
                    "task": "vss".
                * For VPS:
                    "image_size": (output_height, output_width).
                    "pred_masks": Tensor, shape is (t, h, w),
                        that represents the unique ID for the object which each pixel belong to.
                    "segments_infos": list[dict], info dicts for per object.
                        Info dict including unique ID, category ID and isthing.
                    "pred_ids": list, query ids for per thing and stuff, list length is N.
                    "task": "vps".
        """
        name = batched_inputs[0]['name']

        for batched_input in batched_inputs:
            assert name == batched_input['name']

        # for running demo on very long videos
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

        text_classifier, num_templates = self._set_class_information(batched_inputs[0]['name'], self.training)
        # Append void class weight
        text_classifier, num_templates = self.get_text_classifier_with_void(text_classifier, num_templates,
                                                                            name=batched_inputs[0]['name'])

        if not self.training and self.window_inference:
            outputs = self.run_window_inference(images.tensor, window_size=self.window_size,
                                                text_classifier=text_classifier,
                                                num_templates=num_templates)
        else:
            self.backbone.eval()
            self.sem_seg_head.eval()
            with torch.no_grad():
                features = self.backbone(images.tensor)
                features['text_classifier'] = text_classifier
                features['num_templates'] = num_templates
                image_outputs = self.sem_seg_head(features)
                if 'transformer_features' in image_outputs.keys():
                    cur_features = image_outputs['transformer_features']
                else:
                    cur_features = None
                object_labels = self._get_instance_labels(image_outputs['pred_logits'])
                frame_embds = image_outputs['pred_embds'].clone().detach()  # (b, c, t, q)
                frame_embds_no_norm = image_outputs['pred_embds_without_norm'].clone().detach()  # (b, c, t, q)
                mask_features = image_outputs['mask_features'].clone().detach().unsqueeze(0)
                del image_outputs['mask_features']
                torch.cuda.empty_cache()
            outputs, indices = self.tracker(frame_embds, mask_features, return_indices=True,
                                            resume=self.keep, frame_classes=object_labels,
                                            frame_embeds_no_norm=frame_embds_no_norm,
                                            cur_feature=cur_features, text_classifier=text_classifier,
                                            num_templates=num_templates)
            image_outputs = self.reset_image_output_order(image_outputs, indices)

        if self.training:
            targets = self.prepare_targets(batched_inputs, images)
            image_outputs, outputs, targets = self.frame_decoder_loss_reshape(
                outputs, targets, image_outputs=image_outputs
            )
            # use the segmenter prediction results to guide the matching process during early training phase
            if self.iter < self.max_iter_num // 2:
                losses, reference_match_result = self.criterion(outputs, targets,
                                                                matcher_outputs=image_outputs,
                                                                ret_match_result=True)
            else:
                losses, reference_match_result = self.criterion(outputs, targets,
                                                                matcher_outputs=None,
                                                                ret_match_result=True)

            # bipartite matching-based loss
            self.iter += 1
            losses_cl = self.get_cl_loss_ref(outputs, reference_match_result)
            losses.update(losses_cl)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            # when inference, bs must be 1
            mask_cls_results = outputs["pred_logits"][0].to(self.device)  # t q c

            # We ensemble the pred logits of in-vocab and out-vocab
            pooled_clip_feature = outputs['pooled_clip_feature']
            out_vocab_cls_results = get_classification_logits(pooled_clip_feature, text_classifier,
                                                              self.backbone.clip_model.logit_scale, num_templates)

            in_vocab_cls_results = mask_cls_results[..., :-1]  # remove void
            out_vocab_cls_results = out_vocab_cls_results[..., :-1]  # remove void

            # Reference: https://github.com/NVlabs/ODISE/blob/main/odise/modeling/meta_arch/odise.py#L1506
            out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
            in_vocab_cls_results = in_vocab_cls_results.softmax(-1)
            category_overlapping_mask = self.category_overlapping_mask.to(self.device)

            if self.ensemble_on_valid_mask:
                # Only include out_vocab cls results on masks with valid pixels
                # We empirically find that this is important to obtain reasonable AP/mIOU score with ResNet CLIP models
                valid_masking = outputs['valid_masking']
                valid_masking = valid_masking.to(in_vocab_cls_results).unsqueeze(-1)
                alpha = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_alpha
                beta = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_beta
                alpha = alpha * valid_masking
                beta = beta * valid_masking
            else:
                alpha = self.geometric_ensemble_alpha
                beta = self.geometric_ensemble_beta

            cls_logits_seen = (
                    (in_vocab_cls_results ** (1 - alpha) * out_vocab_cls_probs ** alpha).log()
                    * category_overlapping_mask
            )
            cls_logits_unseen = (
                    (in_vocab_cls_results ** (1 - beta) * out_vocab_cls_probs ** beta).log()
                    * (1 - category_overlapping_mask)
            )
            cls_results = cls_logits_seen + cls_logits_unseen

            # This is used to filtering void predictions.
            is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
            mask_cls_probs = torch.cat([
                cls_results.softmax(-1) * (1.0 - is_void_prob),
                is_void_prob], dim=-1)
            mask_cls_results = torch.log(mask_cls_probs + 1e-8)
            outputs["pred_logits"][0] = mask_cls_results  # t q c

            outputs = self.post_processing(outputs)
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            pred_ids = outputs["ids"]

            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            pred_id = pred_ids[0]
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video_task)(
                mask_cls_result, mask_pred_result, image_size, height, width, first_resize_size, pred_id
            )

    def get_cl_loss_ref(self, outputs, referecne_match_result):
        references = outputs['pred_references']

        # per frame
        contrastive_items = []
        for i in range(references.size(0)):
            if i == 0:
                continue
            frame_reference = references[i]  # (q, c)
            frame_reference_ = references[i - 1]  # (q, c)

            if i != references.size(0) - 1:
                frame_reference_next = references[i + 1]
            else:
                frame_reference_next = None

            frame_ref_gt_indices = referecne_match_result[i]

            gt2ref = {}
            for i_ref, i_gt in zip(frame_ref_gt_indices[0], frame_ref_gt_indices[1]):
                gt2ref[i_gt.item()] = i_ref.item()
            # per instance
            for i_gt in gt2ref.keys():
                i_ref = gt2ref[i_gt]

                anchor_embeds = frame_reference[[i_ref]]
                pos_embeds = frame_reference_[[i_ref]]
                neg_range = list(range(0, i_ref)) + list(range(i_ref + 1, frame_reference.size(0)))
                neg_embeds = frame_reference_[neg_range]

                num_positive = pos_embeds.shape[0]
                # concate pos and neg to get whole constractive samples
                pos_neg_embedding = torch.cat(
                    [pos_embeds, neg_embeds], dim=0)
                # generate label, pos is 1, neg is 0
                pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                                                            dtype=torch.int64)  # noqa
                pos_neg_label[:num_positive] = 1.

                # dot product
                dot_product = torch.einsum(
                    'ac,kc->ak', [pos_neg_embedding, anchor_embeds])
                aux_normalize_pos_neg_embedding = nn.functional.normalize(
                    pos_neg_embedding, dim=1)
                aux_normalize_anchor_embedding = nn.functional.normalize(
                    anchor_embeds, dim=1)

                aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                   aux_normalize_anchor_embedding])
                contrastive_items.append({
                    'dot_product': dot_product,
                    'cosine_similarity': aux_cosine_similarity,
                    'label': pos_neg_label})

                if frame_reference_next is not None:
                    pos_embeds = frame_reference_next[[i_ref]]
                    neg_range = list(range(0, i_ref)) + list(range(i_ref + 1, frame_reference.size(0)))
                    # print(neg_range, '---------', i_key)
                    neg_embeds = frame_reference_next[neg_range]

                    num_positive = pos_embeds.shape[0]
                    # concate pos and neg to get whole constractive samples
                    pos_neg_embedding = torch.cat(
                        [pos_embeds, neg_embeds], dim=0)
                    # generate label, pos is 1, neg is 0
                    pos_neg_label = pos_neg_embedding.new_zeros((pos_neg_embedding.shape[0],),
                                                                dtype=torch.int64)  # noqa
                    pos_neg_label[:num_positive] = 1.

                    # dot product
                    dot_product = torch.einsum(
                        'ac,kc->ak', [pos_neg_embedding, anchor_embeds])
                    aux_normalize_pos_neg_embedding = nn.functional.normalize(
                        pos_neg_embedding, dim=1)
                    aux_normalize_anchor_embedding = nn.functional.normalize(
                        anchor_embeds, dim=1)

                    aux_cosine_similarity = torch.einsum('ac,kc->ak', [aux_normalize_pos_neg_embedding,
                                                                       aux_normalize_anchor_embedding])
                    contrastive_items.append({
                        'dot_product': dot_product,
                        'cosine_similarity': aux_cosine_similarity,
                        'label': pos_neg_label})

        losses = loss_reid(contrastive_items, outputs)
        return losses

    def _get_instance_labels(self, pred_logits):
        # b, t, q, c
        pred_logits = pred_logits[0]  # (t, q, c)
        scores = F.softmax(pred_logits, dim=-1)
        labels = torch.argmax(scores, dim=2)  # (t, q)
        labels[labels == pred_logits.size(2) - 1] = -1
        return labels

    def frame_decoder_loss_reshape(self, outputs, targets, image_outputs=None):
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
        outputs['pred_logits'] = einops.rearrange(outputs['pred_logits'], 'b t q c -> (b t) q c')
        outputs['pred_references'] = einops.rearrange(outputs['pred_references'], 'b c t q -> (b t) q c')
        if image_outputs is not None:
            image_outputs['pred_masks'] = einops.rearrange(image_outputs['pred_masks'], 'b q t h w -> (b t) q () h w')
            image_outputs['pred_logits'] = einops.rearrange(image_outputs['pred_logits'], 'b t q c -> (b t) q c')
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
        return image_outputs, outputs, gt_instances

    def reset_image_output_order(self, output, indices):
        """
        in order to maintain consistency between the initial query and the guided results (segmenter prediction)
        :param output: segmenter prediction results (image-level segmentation results)
        :param indices: matched indicates
        :return: reordered outputs
        """
        indices = torch.Tensor(indices).to(torch.int64)  # (t, q)
        frame_indices = torch.range(0, indices.shape[0] - 1).to(indices).unsqueeze(1).repeat(1, indices.shape[1])
        # pred_masks, shape is (b, q, t, h, w)
        output['pred_masks'][0] = output['pred_masks'][0][indices, frame_indices].transpose(0, 1)
        # pred logits, shape is (b, t, q, c)
        output['pred_logits'][0] = output['pred_logits'][0][frame_indices, indices]
        return output

    def post_processing(self, outputs, aux_logits=None):
        """
        average the class logits and append query ids
        """
        pred_logits = outputs['pred_logits']
        pred_logits = pred_logits[0]  # (t, q, c)

        out_logits = torch.mean(pred_logits, dim=0).unsqueeze(0)
        if aux_logits is not None:
            aux_logits = aux_logits[0]
            aux_logits = torch.mean(aux_logits, dim=0)  # (q, c)
        outputs['pred_logits'] = out_logits
        outputs['ids'] = [torch.arange(0, outputs['pred_masks'].size(1))]
        if aux_logits is not None:
            return outputs, aux_logits
        return outputs

    def run_window_inference(self, images_tensor, window_size=30, text_classifier=None, num_templates=None):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1
        out_list = []
        maskpool_embeddings = []  # (windows q c)
        pixel_nums = []
        valid_masks = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size
            # segmeter inference
            features = self.backbone(images_tensor[start_idx:end_idx])
            features['text_classifier'] = text_classifier
            features['num_templates'] = num_templates
            out = self.sem_seg_head(features)
            if 'transformer_features' in out.keys():
                cur_features = out['transformer_features']
            else:
                cur_features = None
            # remove unnecessary variables to save GPU memory
            del features['res2'], features['res3'], features['res4'], features['res5']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            # referring tracker inference
            frame_embds = out['pred_embds']  # (b, c, t, q)
            frame_embds_no_norm = out['pred_embds_without_norm']
            mask_features = out['mask_features'].unsqueeze(0)
            if i != 0 or self.keep:
                track_out = self.tracker(frame_embds, mask_features,
                                         resume=True, frame_embeds_no_norm=frame_embds_no_norm,
                                         cur_feature=cur_features, text_classifier=text_classifier,
                                         num_templates=num_templates)
            else:
                track_out = self.tracker(frame_embds, mask_features, frame_embeds_no_norm=frame_embds_no_norm,
                                         cur_feature=cur_features, text_classifier=text_classifier,
                                         num_templates=num_templates)

            # get clip embeddings
            mask_for_pooling_ = F.interpolate(track_out['pred_masks'][0].transpose(0, 1),
                                              size=features['clip_vis_dense'].shape[-2:],
                                              mode='bilinear', align_corners=False)  # (t, q, h, w)

            if "convnext" in self.backbone.model_name.lower():
                pooled_clip_feature, pixel_num = self.mask_pooling(features['clip_vis_dense'], mask_for_pooling_,
                                                                   return_num=True)
                maskpool_embeddings.append(pooled_clip_feature)  # (windows q c)
                pixel_nums.append(pixel_num)  # (windows q 1)
            elif "rn" in self.backbone.model_name.lower():
                pooled_clip_feature = self.backbone.visual_prediction_forward(features['clip_vis_dense'],
                                                                              mask_for_pooling_)  # (t, q, c)
                maskpool_embeddings.append(pooled_clip_feature)

                valid_masking = (mask_for_pooling_ > 0).to(mask_for_pooling_).sum(-1).sum(-1) > 0  # (t, q)
                valid_masks.append(valid_masking)

            else:
                raise NotImplementedError

            # remove unnecessary variables to save GPU memory
            del mask_features
            for j in range(len(track_out['aux_outputs'])):
                del track_out['aux_outputs'][j]['pred_masks'], track_out['aux_outputs'][j]['pred_logits']
            track_out['pred_logits'] = track_out['pred_logits'].to(torch.float32).detach().cpu()
            track_out['pred_masks'] = track_out['pred_masks'].to(torch.float32).detach().cpu()
            track_out['pred_embds'] = track_out['pred_embds'].to(torch.float32).detach().cpu()
            out_list.append(track_out)

        # merge outputs
        outputs = {}
        outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in out_list], dim=1)
        outputs['pred_masks'] = torch.cat([x['pred_masks'] for x in out_list], dim=2)
        outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in out_list], dim=2)

        if len(pixel_nums) == 0:
            pooled_clip_feature = torch.cat(maskpool_embeddings, dim=0)  # (t, q, c)
            valid_masks = torch.cat(valid_masks, dim=0)  # (t, q)
            outputs['valid_masking'] = valid_masks
        else:
            maskpool_embeddings = torch.cat(maskpool_embeddings, dim=0)
            pixel_nums = torch.cat(pixel_nums, dim=0)[:, :, :, 0]
            pixel_nums = pixel_nums / torch.sum(pixel_nums, dim=0, keepdim=True)
            maskpool_embeddings = maskpool_embeddings * pixel_nums
            maskpool_embeddings = torch.sum(maskpool_embeddings, dim=0, keepdim=True)
            pooled_clip_feature = self.backbone.visual_prediction_forward(maskpool_embeddings)  # (1 q c)
        outputs['pooled_clip_feature'] = pooled_clip_feature

        return outputs

    def inference_video_vis(
        self, pred_cls, pred_masks, img_size, output_height, output_width,
        first_resize_size, pred_id, aux_pred_cls=None,
    ):
        if len(pred_cls) > 0:
            scores = F.softmax(pred_cls, dim=-1)[:, :-1]
            if aux_pred_cls is not None:
                aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
                scores = torch.maximum(scores, aux_pred_cls.to(scores))
            labels = torch.arange(
                # self.sem_seg_head.num_classes, device=self.device
                pred_cls.shape[-1] - 1, device=self.device
            ).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
            # keep top-K predictions
            scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.max_num, sorted=False)
            labels_per_image = labels[topk_indices]
            topk_indices = topk_indices // (pred_cls.shape[-1] - 1)
            pred_masks = pred_masks[topk_indices]
            pred_ids = pred_id[topk_indices]

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
        first_resize_size, pred_id, aux_pred_cls=None,
    ):
        pred_cls = F.softmax(pred_cls, dim=-1)
        if aux_pred_cls is not None:
            aux_pred_cls = F.softmax(aux_pred_cls, dim=-1)[:, :-1]
            pred_cls[:, :-1] = torch.maximum(pred_cls[:, :-1], aux_pred_cls.to(pred_cls))
        mask_pred = pred_masks
        scores, labels = pred_cls.max(-1)

        # filter out the background prediction
        keep = labels.ne(pred_cls.shape[-1] - 1) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_ids = pred_id[keep]
        cur_masks = mask_pred[keep]

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
                isthing = pred_class < len(self.test_metadata[self.name].thing_dataset_id_to_contiguous_id)
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
        first_resize_size, pred_id, aux_pred_cls=None,
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
class DVIS_offline_OV(DVIS_online_OV):
    """
    Offline version of DVIS, including a segmenter, a referring tracker and a temporal refiner.
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
        train_metadatas: dict,
        test_metadatas: dict,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # video
        tracker,
        refiner,
        num_frames,
        window_inference,
        max_num,
        max_iter_num,
        window_size,
        task,
        # fc-clip
        geometric_ensemble_alpha: float,
        geometric_ensemble_beta: float,
        ensemble_on_valid_mask: bool,
        # multi datasets
        test2train={},
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
            # video
            tracker: a tracker module, e.g. ReferringTracker
            refiner: a refiner module, e.g. TemporalRefiner
            num_frames: number of frames sampled during training
            window_inference: if the GPU memory is insufficient to predict the entire video at
                once, inference needs to be performed clip by clip
            num_class: the categories number of the dataset
            max_num: the maximum number of instances retained for a video, only used in VIS
            max_iter_num: the iter nums
            window_size: the number of images processed by the segmenter at a time
            task: VIS, VSS or VPS
        """
        super().__init__(
            backbone=backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            object_mask_threshold=object_mask_threshold,
            overlap_threshold=overlap_threshold,
            train_metadatas=train_metadatas,
            test_metadatas=test_metadatas,
            size_divisibility=size_divisibility,
            sem_seg_postprocess_before_inference=sem_seg_postprocess_before_inference,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            # video
            tracker=tracker,
            num_frames=num_frames,
            window_inference=window_inference,
            max_num=max_num,
            max_iter_num=max_iter_num,
            window_size=window_size,
            task=task,
            # fc-clip
            geometric_ensemble_alpha=geometric_ensemble_alpha,
            geometric_ensemble_beta=geometric_ensemble_beta,
            ensemble_on_valid_mask=ensemble_on_valid_mask,
            # multi datasets
            test2train=test2train,
        )

        # frozen the referring tracker
        for p in self.tracker.parameters():
            p.requires_grad_(False)

        self.refiner = refiner

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
            # since when calculating the loss, the t frames of a video are flattened into a image with size of (th, w),
            # the number of sampling points is increased t times accordingly.
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS * cfg.INPUT.SAMPLING_FRAME_NUM,
        )

        weight_dict = {
            "loss_ce": class_weight,
            "loss_mask": mask_weight,
            "loss_dice": dice_weight,
        }

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = VideoSetCriterion_ov(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS * cfg.INPUT.SAMPLING_FRAME_NUM,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )

        tracker = ReferringTracker_noiser_OV(
            hidden_channel=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            feedforward_channel=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            num_head=cfg.MODEL.MASK_FORMER.NHEADS,
            decoder_layer_num=cfg.MODEL.TRACKER.DECODER_LAYERS,
            noise_mode=cfg.MODEL.TRACKER.NOISE_MODE,
            mask_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            mask_pooling=sem_seg_head.predictor.mask_pooling,
            mask_pooling_proj=sem_seg_head.predictor._mask_pooling_proj,
            class_embed=sem_seg_head.predictor.class_embed,
            logit_scale=sem_seg_head.predictor.logit_scale,
            mask_embed=sem_seg_head.predictor.mask_embed,
            decoder_norm=sem_seg_head.predictor.decoder_norm,
        )

        refiner = TemporalRefiner_OV(
            hidden_channel=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            feedforward_channel=cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD,
            num_head=cfg.MODEL.MASK_FORMER.NHEADS,
            decoder_layer_num=cfg.MODEL.REFINER.DECODER_LAYERS,
            mask_dim=cfg.MODEL.MASK_FORMER.HIDDEN_DIM,
            windows=cfg.MODEL.MASK_FORMER.TEST.WINDOW_SIZE,
            mask_pooling=sem_seg_head.predictor.mask_pooling,
            mask_pooling_proj=sem_seg_head.predictor._mask_pooling_proj,
            class_embed=sem_seg_head.predictor.class_embed,
            logit_scale=sem_seg_head.predictor.logit_scale,
            mask_embed=sem_seg_head.predictor.mask_embed,
            decoder_norm=sem_seg_head.predictor.decoder_norm,
        )

        max_iter_num = cfg.SOLVER.MAX_ITER

        train_metadatas = {}
        test_metadatas = {}
        for name in cfg.DATASETS.TRAIN:
            train_metadatas[name] = MetadataCatalog.get(name)
        for name in cfg.DATASETS.TEST:
            test_metadatas[name] = MetadataCatalog.get(name)

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "train_metadatas": train_metadatas,
            "test_metadatas": test_metadatas,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": True,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # video
            "tracker": tracker,
            "refiner": refiner,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "window_inference": cfg.MODEL.MASK_FORMER.TEST.WINDOW_INFERENCE,
            "max_num": cfg.MODEL.MASK_FORMER.TEST.MAX_NUM,
            "max_iter_num": max_iter_num,
            "window_size": cfg.MODEL.MASK_FORMER.TEST.WINDOW_SIZE,
            "task": cfg.MODEL.MASK_FORMER.TEST.TASK,
            # fc-clip
            "geometric_ensemble_alpha": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_ALPHA,
            "geometric_ensemble_beta": cfg.MODEL.FC_CLIP.GEOMETRIC_ENSEMBLE_BETA,
            "ensemble_on_valid_mask": cfg.MODEL.FC_CLIP.ENSEMBLE_ON_VALID_MASK,
            # multi datasets
            "test2train": {x: y for x, y in zip(cfg.DATASETS.TEST, cfg.DATASETS.TEST2TRAIN)},
        }

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
            dict:
                For specific task, the dict contains the following keys:
                * For VIS:
                    "image_size": (output_height, output_width).
                    "pred_scores": score for per instance.
                    "pred_labels": class for per instance.
                    "pred_masks": list[Tensor], bit-masks for per instance, Tensor shape is (t, h, w).
                    "pred_ids": list, query ids for per instance, list length is N.
                    "task": "vis",
                * For VSS:
                    "image_size": (output_height, output_width).
                    "pred_masks": A Tensor that represents the
                        per-pixel segmentation prediced by the head.
                        The prediction has shape (t, h, w) that represents
                        the category ID for each pixel.
                    "task": "vss".
                * For VPS:
                    "image_size": (output_height, output_width).
                    "pred_masks": Tensor, shape is (t, h, w),
                        that represents the unique ID for the object which each pixel belong to.
                    "segments_infos": list[dict], info dicts for per object.
                        Info dict including unique ID, category ID and isthing.
                    "pred_ids": list, query ids for per thing and stuff, list length is N.
                    "task": "vps".
        """
        name = batched_inputs[0]['name']

        for batched_input in batched_inputs:
            assert name == batched_input['name']

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
        self.backbone.eval()
        self.sem_seg_head.eval()
        self.tracker.eval()

        text_classifier, num_templates = self._set_class_information(batched_inputs[0]['name'], self.training)
        # Append void class weight
        text_classifier, num_templates = self.get_text_classifier_with_void(text_classifier, num_templates,
                                                                            name=batched_inputs[0]['name'])
        if not self.training and self.window_inference:
            outputs, online_pred_logits = self.run_window_inference(images.tensor, window_size=self.window_size,
                                                                    text_classifier=text_classifier,
                                                                    num_templates=num_templates)
        else:
            with torch.no_grad():
                # due to GPU memory limitations, the segmenter processes the video clip by clip.
                image_outputs = self.segmentor_windows_inference(images.tensor, window_size=21,
                                                                 text_classifier=text_classifier,
                                                                 num_templates=num_templates)
                frame_embds = image_outputs['pred_embds'].clone().detach()  # (b, c, t, q)
                frame_embds_no_norm = image_outputs['pred_embds_without_norm'].clone().detach()  # (b, c, t, q)
                mask_features = image_outputs['mask_features'].clone().detach().unsqueeze(0)
                del image_outputs['mask_features'], image_outputs['pred_embds_without_norm'],\
                    image_outputs['pred_logits'], image_outputs['pred_embds']

                # perform tracker/alignment
                image_outputs = self.tracker(
                    frame_embds, mask_features,
                    resume=self.keep, frame_embeds_no_norm=frame_embds_no_norm,
                    text_classifier=text_classifier,
                    num_templates=num_templates
                )
                online_pred_logits = image_outputs['pred_logits']  # (b, t, q, c)
                # frame_embds_ = self.tracker.frame_forward(frame_embds)
                frame_embds_ = frame_embds_no_norm.clone().detach()
                instance_embeds = image_outputs['pred_embds'].clone().detach()

                del frame_embds, frame_embds_no_norm
                del image_outputs['pred_embds']
                for j in range(len(image_outputs['aux_outputs'])):
                    del image_outputs['aux_outputs'][j]['pred_masks'], image_outputs['aux_outputs'][j]['pred_logits']
                torch.cuda.empty_cache()
            # do temporal refine
            outputs = self.refiner(instance_embeds, frame_embds_, mask_features,
                                   text_classifier=text_classifier,
                                   num_templates=num_templates)

        if self.training:
            # mask classification target
            targets = self.prepare_targets(batched_inputs, images)
            # use the online prediction results to guide the matching process during early training phase
            if self.iter < self.max_iter_num // 2:
                image_outputs, outputs, targets = self.frame_decoder_loss_reshape(
                    outputs, targets, image_outputs=image_outputs
                )
            else:
                image_outputs, outputs, targets = self.frame_decoder_loss_reshape(
                    outputs, targets, image_outputs=None
                )
            self.iter += 1

            # bipartite matching-based loss
            losses, matching_result = self.criterion(outputs, targets,
                                                     matcher_outputs=image_outputs,
                                                     ret_match_result=True)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            # when inference, bs must be 1
            mask_pred_results = outputs["pred_masks"][0].transpose(0, 1)  # t q h w
            mask_cls_results = outputs["pred_logits"][0].to(self.device)  # t q c

            # We ensemble the pred logits of in-vocab and out-vocab
            clip_feature = outputs["clip_vis_dense"]
            pooled_clip_feature, valid_masking = self.windows_get_maskpool_embeds(clip_feature, mask_pred_results)
            out_vocab_cls_results = get_classification_logits(pooled_clip_feature, text_classifier,
                                                              self.backbone.clip_model.logit_scale, num_templates)
            in_vocab_cls_results = mask_cls_results[..., :-1]  # remove void
            out_vocab_cls_results = out_vocab_cls_results[..., :-1]  # remove void
            # Reference: https://github.com/NVlabs/ODISE/blob/main/odise/modeling/meta_arch/odise.py#L1506
            out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
            in_vocab_cls_results = in_vocab_cls_results.softmax(-1)
            category_overlapping_mask = self.category_overlapping_mask.to(self.device)

            if self.ensemble_on_valid_mask:
                # Only include out_vocab cls results on masks with valid pixels
                # We empirically find that this is important to obtain reasonable AP/mIOU score with ResNet CLIP models
                valid_masking = valid_masking.to(in_vocab_cls_results).unsqueeze(-1)
                alpha = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_alpha
                beta = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_beta
                alpha = alpha * valid_masking
                beta = beta * valid_masking
            else:
                alpha = self.geometric_ensemble_alpha
                beta = self.geometric_ensemble_beta

            cls_logits_seen = (
                    (in_vocab_cls_results ** (1 - alpha) * out_vocab_cls_probs ** alpha).log()
                    * category_overlapping_mask
            )
            cls_logits_unseen = (
                    (in_vocab_cls_results ** (1 - beta) * out_vocab_cls_probs ** beta).log()
                    * (1 - category_overlapping_mask)
            )
            cls_results = cls_logits_seen + cls_logits_unseen
            # This is used to filtering void predictions.
            is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
            mask_cls_probs = torch.cat([
                cls_results.softmax(-1) * (1.0 - is_void_prob),
                is_void_prob], dim=-1)
            mask_cls_results = torch.log(mask_cls_probs + 1e-8)
            outputs["pred_logits"][0] = mask_cls_results  # t q c


            outputs, aux_pred_logits = self.post_processing(outputs, aux_logits=online_pred_logits)
            aux_pred_logits = None

            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            pred_ids = outputs["ids"]

            mask_cls_result = mask_cls_results[0]
            mask_pred_result = mask_pred_results[0]
            pred_id = pred_ids[0]
            first_resize_size = (images.tensor.shape[-2], images.tensor.shape[-1])

            input_per_image = batched_inputs[0]
            image_size = images.image_sizes[0]  # image size without padding after data augmentation

            height = input_per_image.get("height", image_size[0])  # raw image size before data augmentation
            width = input_per_image.get("width", image_size[1])

            return retry_if_cuda_oom(self.inference_video_task)(
                mask_cls_result, mask_pred_result, image_size, height, width,
                first_resize_size, pred_id, aux_pred_cls=aux_pred_logits,
            )

    def windows_get_maskpool_embeds(self, clip_feature, mask_pred_results, windows=36):
        """
        for windows prediction, because mask features consumed too much GPU memory
        """
        # clip_feature, (t, c, h, w)
        # mask_pred_results,  t, q, h, w
        iters = clip_feature.size(0) // windows
        if clip_feature.size(0) % windows != 0:
            iters += 1
        maskpool_embeddings = []
        pixel_nums = []
        valid_masking = []
        for i in range(iters):
            start_idx = i * windows
            end_idx = (i + 1) * windows
            clip_feature_ = clip_feature[start_idx: end_idx].to(self.device)
            mask_pred_results_ = mask_pred_results[start_idx: end_idx].to(self.device)
            mask_for_pooling_ = F.interpolate(mask_pred_results_, size=clip_feature.shape[-2:],
                                              mode='bilinear', align_corners=False)

            if "convnext" in self.backbone.model_name.lower():
                pooled_clip_feature, pixel_num = self.mask_pooling(clip_feature_, mask_for_pooling_, return_num=True)
                maskpool_embeddings.append(pooled_clip_feature)  # (windows q c)
                pixel_nums.append(pixel_num)  # (windows q 1)
            elif "rn" in self.backbone.model_name.lower():
                pooled_clip_feature = self.backbone.visual_prediction_forward(clip_feature_,
                                                                              mask_for_pooling_)  # (t, q, c)
                maskpool_embeddings.append(pooled_clip_feature)
                valid_masking_ = (mask_for_pooling_ > 0).to(mask_for_pooling_).sum(-1).sum(-1) > 0
                valid_masking.append(valid_masking_)
            else:
                raise NotImplementedError

        if len(pixel_nums) == 0:
            pooled_clip_feature = torch.cat(maskpool_embeddings, dim=0)  # (t, q, c)
            valid_masks = torch.cat(valid_masking, dim=0)  # (t, q)
            return pooled_clip_feature, valid_masks
        else:
            maskpool_embeddings = torch.cat(maskpool_embeddings, dim=0)
            pixel_nums = torch.cat(pixel_nums, dim=0)[:, :, :, 0]
            pixel_nums = pixel_nums / torch.sum(pixel_nums, dim=0, keepdim=True)
            maskpool_embeddings = maskpool_embeddings * pixel_nums
            maskpool_embeddings = torch.sum(maskpool_embeddings, dim=0, keepdim=True)
            pooled_clip_feature = self.backbone.visual_prediction_forward(maskpool_embeddings)  # (1 q c)
            return pooled_clip_feature, None

    def segmentor_windows_inference(self, images_tensor, window_size=5,
                                    text_classifier=None, num_templates=None):
        image_outputs = {}
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1

        outs_list = []
        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i + 1) * window_size

            features = self.backbone(images_tensor[start_idx:end_idx])
            features['text_classifier'] = text_classifier
            features['num_templates'] = num_templates
            out = self.sem_seg_head(features)

            del features['res2'], features['res3'], features['res4'], features['res5']
            del out['pred_masks']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']
            outs_list.append(out)

        image_outputs['pred_embds'] = torch.cat([x['pred_embds'] for x in outs_list], dim=2).detach()
        image_outputs['mask_features'] = torch.cat([x['mask_features'] for x in outs_list], dim=0).detach()
        image_outputs['pred_logits'] = torch.cat([x['pred_logits'] for x in outs_list], dim=1).detach()
        image_outputs['pred_embds_without_norm'] = torch.cat([x['pred_embds_without_norm'] for x in outs_list], dim=2).detach()
        return image_outputs

    def frame_decoder_loss_reshape(self, outputs, targets, image_outputs=None):
        # flatten the t frames as an image with size of (th, w)
        outputs['pred_masks'] = einops.rearrange(outputs['pred_masks'], 'b q t h w -> b q () (t h) w')
        outputs['pred_logits'] = outputs['pred_logits'][:, 0, :, :]
        if image_outputs is not None:
            image_outputs['pred_masks'] = einops.rearrange(image_outputs['pred_masks'], 'b q t h w -> b q () (t h) w')
            image_outputs['pred_logits'] = image_outputs['pred_logits'].mean(dim=1)
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

    def run_window_inference(self, images_tensor, window_size=30,
                             text_classifier=None, num_templates=None):
        iters = len(images_tensor) // window_size
        if len(images_tensor) % window_size != 0:
            iters += 1

        overall_mask_features = []
        overall_frame_embds = []
        overall_instance_embds = []
        online_pred_logits = []
        overall_clip_features = []

        for i in range(iters):
            start_idx = i * window_size
            end_idx = (i+1) * window_size

            # sementer inference
            features = self.backbone(images_tensor[start_idx:end_idx])
            features['text_classifier'] = text_classifier
            features['num_templates'] = num_templates
            overall_clip_features.append(features['clip_vis_dense'].cpu())
            out = self.sem_seg_head(features)

            del features['res2'], features['res3'], features['res4'], features['res5']
            del out['pred_masks']
            for j in range(len(out['aux_outputs'])):
                del out['aux_outputs'][j]['pred_masks'], out['aux_outputs'][j]['pred_logits']

            object_labels = self._get_instance_labels(out['pred_logits'])
            frame_embds = out['pred_embds']  # (b, c, t, q)
            frame_embds_no_norm = out['pred_embds_without_norm']
            mask_features = out['mask_features'].unsqueeze(0)
            overall_mask_features.append(mask_features.cpu())
            overall_frame_embds.append(frame_embds_no_norm)

            # referring tracker inference
            if i != 0:
                track_out = self.tracker(frame_embds, mask_features, resume=True,
                                         frame_classes=object_labels,
                                         frame_embeds_no_norm=frame_embds_no_norm,
                                         text_classifier=text_classifier, num_templates=num_templates)
            else:
                track_out = self.tracker(frame_embds, mask_features, frame_classes=object_labels,
                                         frame_embeds_no_norm=frame_embds_no_norm,
                                         text_classifier=text_classifier, num_templates=num_templates)
            online_pred_logits.append(track_out['pred_logits'].clone())

            del track_out['pred_masks'], track_out['pred_logits']
            for j in range(len(track_out['aux_outputs'])):
                del track_out['aux_outputs'][j]['pred_masks'], track_out['aux_outputs'][j]['pred_logits']

            instance_embds = track_out['pred_embds']
            overall_instance_embds.append(instance_embds)

        overall_frame_embds = torch.cat(overall_frame_embds, dim=2)
        overall_instance_embds = torch.cat(overall_instance_embds, dim=2)
        overall_mask_features = torch.cat(overall_mask_features, dim=1)
        online_pred_logits = torch.cat(online_pred_logits, dim=1)
        overall_clip_features = torch.cat(overall_clip_features, dim=0)

        # temporal refiner inference
        outputs = self.refiner(overall_instance_embds, overall_frame_embds, overall_mask_features,
                               text_classifier=text_classifier, num_templates=num_templates)
        outputs.update({'clip_vis_dense': overall_clip_features})
        return outputs, online_pred_logits