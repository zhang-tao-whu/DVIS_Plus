# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/sukjunhwang/IFC

import os
import json
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.coco import register_coco_instances

from .ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta,
    _get_ovis_instances_meta,
    _get_bdd_obj_track_meta,
)

# from .uninext_ytvis import register_uninext_ytvis_instances

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
                         "ytvis_2019/train.json"),
    "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
                       "ytvis_2019/valid.json"),
    "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
                        "ytvis_2019/test.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("ytvis_2021/train/JPEGImages",
                         "ytvis_2021/train.json"),
    "ytvis_2021_val": ("ytvis_2021/valid/JPEGImages",
                       "ytvis_2021/valid.json"),
    "ytvis_2021_test": ("ytvis_2021/test/JPEGImages",
                        "ytvis_2021/test.json"),
}

_PREDEFINED_SPLITS_YTVIS_2022 = {
    "ytvis_2022_val": ("ytvis_2022/valid/JPEGImages",
                       "ytvis_2022/valid/instances.json"),
}

# ==== Predefined splits for OVIS ===========
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train": ("ovis/train",
                         "ovis/annotations/annotations_train.json"),
    "ovis_val": ("ovis/valid",
                       "ovis/annotations/annotations_valid.json"),
    "ovis_test": ("ovis/test",
                        "ovis/annotations/annotations_test.json"),
    "ovis_rebuttal_train": ("ovis/train",
                            "ovis/annotations/annotations_train_rebuttal.json"),
    "ovis_rebuttal_val": ("ovis/train",
                          "ovis/annotations/annotations_valid_rebuttal.json"),
    "ovis_rebuttal_val_0p3": ("ovis/train",
                              "ovis/annotations/annotations_valid_rebuttal_0p3.json"),
    "ovis_rebuttal_val_0p4": ("ovis/train",
                              "ovis/annotations/annotations_valid_rebuttal_0p4.json"),
    "ovis_rebuttal_val_0p5": ("ovis/train",
                              "ovis/annotations/annotations_valid_rebuttal_0p5.json"),
    "ovis_rebuttal_val_0p6": ("ovis/train",
                              "ovis/annotations/annotations_valid_rebuttal_0p6.json"),
}

_PREDEFINED_SPLITS_COCO_VIDEO = {
    "coco2ytvis2019_train": ("coco/train2017", "coco/annotations/coco2ytvis2019_train.json"),
    "coco2ytvis2019_val": ("coco/val2017", "coco/annotations/coco2ytvis2019_val.json"),
    "coco2ytvis2021_train": ("coco/train2017", "coco/annotations/coco2ytvis2021_train.json"),
    "coco2ytvis2021_val": ("coco/val2017", "coco/annotations/coco2ytvis2021_val.json"),
    "coco2ovis_train": ("coco/train2017", "coco/annotations/coco2ovis_train.json"),
    "coco2ovis_val": ("coco/val2017", "coco/annotations/coco2ovis_val.json"),
}


_PREDEFINED_SPLITS_BDD_SEG_TRACK = {
    "bdd_seg_track_train": ("bdd100k/images/seg_track_20/train", "bdd100k/labels/seg_track_20/seg_track_train_cocoformat_uni.json"),
    "bdd_seg_track_val": ("bdd100k/images/seg_track_20/val", "bdd100k/labels/seg_track_20/seg_track_val_cocoformat_uni.json"),
}

_PREDEFINED_SPLITS_BDD2OVIS_SEG_TRACK = {
    "bdd2ovis_seg_track_train": ("bdd100k/images/seg_track_20/train", "bdd100k/labels/seg_track_20/seg_track_train_cocoformat_uni_ovis.json")
}


def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_ytvis_2022(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2022.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_coco_video(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_VIDEO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata("coco"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_bdd_seg_track(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD_SEG_TRACK.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_bdd_obj_track_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
            # has_mask=True
        )

def register_all_bdd2ovis_seg_track(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_BDD2OVIS_SEG_TRACK.items():
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


_PREDEFINED_SPLITS_SOT = {
    "sot_mose_train": ("MOSE/train/JPEGImages", "MOSE/train/train.json"),
    "sot_mose_val": ("MOSE/valid/JPEGImages", "MOSE/valid/valid.json"),
}

SOT_CATEGORIES = [{"color": [220, 20, 60], "isthing": 1, "id": 1, "name": "object"}]  # only one class for visual grounding

def _get_sot_meta():
    thing_ids = [k["id"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    assert len(thing_ids) == 1, len(thing_ids)

    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in SOT_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def register_all_sot(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_SOT.items():
        has_mask = ("coco" in key) or ("vos" in key) or ("davis" in key)
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_sot_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    #register_all_ytvis_2019(_root)
    #register_all_ytvis_2021(_root)
    register_all_ytvis_2022(_root)
    register_all_ovis(_root)
    register_all_coco_video(_root)
    register_all_bdd_seg_track(_root)
    register_all_bdd2ovis_seg_track(_root)
    from . import vps
    from . import vss
    register_all_sot(_root)
