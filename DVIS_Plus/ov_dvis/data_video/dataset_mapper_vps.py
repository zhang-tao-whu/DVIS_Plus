import logging

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import transforms as T
from dvis_Plus.data_video.dataset_mapper_vps import PanopticDatasetVideoMapper

__all__ = ["OpenVocabularyPanopticDatasetVideoMapper"]

class OpenVocabularyPanopticDatasetVideoMapper(PanopticDatasetVideoMapper):
    @configurable
    def __init__(
            self,
            is_train=True,
            is_tgt=True,  # not used, vps not support category mapper
            *,
            augmentations,
            image_format,
            ignore_label=255,
            thing_ids_to_continue_dic={},
            stuff_ids_to_continue_dic={},
            # sample
            sampling_frame_num: int = 2,
            sampling_frame_range: int = 5,
            reverse_agu: bool = False,
            src_dataset_name: str = "",  # not used
            tgt_dataset_name: str = "",  # not used
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """

        self.name = src_dataset_name
        meta = MetadataCatalog.get(src_dataset_name)
        ignore_label = meta.ignore_label

        #######
        thing_ids = list(meta.thing_dataset_id_to_contiguous_id.values())
        thing_ids_to_continue_dic = {}
        for ii, id_ in enumerate(sorted(thing_ids)):
            thing_ids_to_continue_dic[id_] = ii

        stuff_ids = list(meta.stuff_dataset_id_to_contiguous_id.values())
        stuff_ids_to_continue_dic = {}
        for ii, id_ in enumerate(sorted(stuff_ids)):
            stuff_ids_to_continue_dic[id_] = ii

        self.is_train = is_train
        self.tfm_gens = augmentations
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.thing_ids_to_continue_dic = thing_ids_to_continue_dic
        self.stuff_ids_to_continue_dic = stuff_ids_to_continue_dic

        self.sampling_frame_num = sampling_frame_num
        self.sampling_frame_range = sampling_frame_range
        self.sampling_frame_ratio = 1.0
        self.reverse_agu = reverse_agu

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        if is_train:
            augs = [
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    # cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                    "choice",
                )
            ]
            if cfg.INPUT.CROP.ENABLED:
                augs.append(
                    T.RandomCrop_CategoryAreaConstraint(
                        cfg.INPUT.CROP.TYPE,
                        cfg.INPUT.CROP.SIZE,
                        1.0,
                    )
                )
            augs.append(T.RandomFlip())
        else:
            # Resize
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            sample_style = "choice"
            augs = [T.ResizeShortestEdge(min_size, max_size, sample_style)]

        #######
        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.SAMPLING_FRAME_RANGE
        reverse_agu = cfg.INPUT.REVERSE_AGU

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "reverse_agu": reverse_agu,
        }
        return ret

    def convert2ytvis(self, dataset_dict):
        super(OpenVocabularyPanopticDatasetVideoMapper, self).convert2ytvis(dataset_dict)
        dataset_dict.update({'name': self.name})
        return