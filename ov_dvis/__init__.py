# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/MinVIS/blob/main/LICENSE

# Copyright (c) Facebook, Inc. and its affiliates.

# config
from .config import add_ov_dvis_config

from .video_mask2former_transformer_decoder_ov import VideoMultiScaleMaskedTransformerDecoder_minvis_OV, \
     VideoMultiScaleMaskedTransformerDecoder_dvis_OV

from .meta_architecture_ov import MinVIS_OV, DVIS_online_OV, DVIS_offline_OV
from .backbones.clip import CLIP

# video
from .data_video import (
    OpenVocabularyYTVISDatasetMapper,
    OpenVocabularyCocoClipDatasetMapper,
    OpenVocabularyCocoPanoClipDatasetMapper,
    OpenVocabularyPanopticDatasetVideoMapper,
    OpenVocabularySemanticDatasetVideoMapper,
)
