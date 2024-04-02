<div align="center">

# [DVIS++: Improved Decoupled Framework for Universal Video Segmentation](https://arxiv.org/abs/2312.13305)
[Tao Zhang](https://scholar.google.com/citations?user=3xu4a5oAAAAJ&hl=zh-CN), XingYe Tian, Yikang Zhou, 
[ShunPing Ji](https://scholar.google.com/citations?user=FjoRmF4AAAAJ&hl=zh-CN), Xuebo Wang, Xin Tao,

Yuan Zhang, Pengfei Wan, Zhongyuan Wang and 
[Yu Wu](https://scholar.google.com/citations?hl=zh-CN&user=23SZHUwAAAAJ)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvis-improved-decoupled-framework-for/video-instance-segmentation-on-ovis-1)](https://paperswithcode.com/sota/video-instance-segmentation-on-ovis-1?p=dvis-improved-decoupled-framework-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvis-improved-decoupled-framework-for/video-instance-segmentation-on-youtube-vis-1)](https://paperswithcode.com/sota/video-instance-segmentation-on-youtube-vis-1?p=dvis-improved-decoupled-framework-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvis-improved-decoupled-framework-for/video-instance-segmentation-on-youtube-vis-2)](https://paperswithcode.com/sota/video-instance-segmentation-on-youtube-vis-2?p=dvis-improved-decoupled-framework-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvis-improved-decoupled-framework-for/video-instance-segmentation-on-youtube-vis-3)](https://paperswithcode.com/sota/video-instance-segmentation-on-youtube-vis-3?p=dvis-improved-decoupled-framework-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvis-improved-decoupled-framework-for/video-semantic-segmentation-on-vspw)](https://paperswithcode.com/sota/video-semantic-segmentation-on-vspw?p=dvis-improved-decoupled-framework-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvis-improved-decoupled-framework-for/video-panoptic-segmentation-on-vipseg)](https://paperswithcode.com/sota/video-panoptic-segmentation-on-vipseg?p=dvis-improved-decoupled-framework-for)

<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/radar.png" width="400"/>
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/radar_ov.png" width="400"/>
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/overview.png" width="800"/>
</div>

## News
- DVIS-DAQ achieves 57.1 AP on the OVIS dataset and also sets a new SOTA performance on YTVIS19/21 and VIPSeg. The code will be released in this repository. The paper is available at [DVIS-DAQ: Improving Video Segmentation via Dynamic Anchor Queries](https://arxiv.org/pdf/2404.00086.pdf) and the project page can be found in [project page](https://zhang-tao-whu.github.io/projects/DVIS_DAQ/).
- DVIS and DVIS++ achieved **1st place** in the VPS Track of the PVUW challenge at CVPR 2023. `2023.5.25`
- DVIS and DVIS++ achieved **1st place** in the VIS Track of the 5th LSVOS challenge at ICCV 2023. `2023.8.15`

## Features
- DVIS++ is a universal video segmentation framework that supports VIS, VPS and VSS.
- DVIS++ can run in both online and offline modes. 
- DVIS++ achieved SOTA performance on YTVIS 2019&2021&2022, OVIS, VIPSeg and VSPW datasets.
- OV-DVIS++ is the first open-vocabulary video universal segmentation framework with powerful zero-shot segmentation capability. 

## Demos
### VIS
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/34df9b7e.gif" width="200"/> <img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/52ee3d90.gif" width="530"/>
### VSS
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/440_iKo5Acne.gif" width="365"/> <img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/6560e80d02.gif" width="365"/>
### VPS
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/9c4419eb12.gif" width="365"/> <img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/37b4ec2e1a.gif" width="365"/>
### Open-vocabulary demos
<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/f7255a57d0.gif" width="365"/> <img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis_Plus/ffd7c15f47.gif" width="365"/>
## Installation

See [Installation Instructions](INSTALL.md).

## Getting Started

See [Preparing Datasets for DVIS++](datasets/README.md).

See [Getting Started with DVIS++](GETTING_STARTED.md).

## Model Zoo

Trained models are available for download in the [DVIS++ Model Zoo](MODEL_ZOO.md).

## <a name="CitingDVIS"></a>Citing DVIS and DVIS++

```BibTeX
@article{zhang2023dvis,
  title={DVIS: Decoupled Video Instance Segmentation Framework},
  author={Zhang, Tao and Tian, Xingye and Wu, Yu and Ji, Shunping and Wang, Xuebo and Zhang, Yuan and Wan, Pengfei},
  journal={arXiv preprint arXiv:2306.03413},
  year={2023}
}

@article{zhang2023dvisplus,
  title={DVIS++: Improved Decoupled Framework for Universal Video Segmentation}, 
  author={Tao Zhang and Xingye Tian and Yikang Zhou and Shunping Ji and Xuebo Wang and Xin Tao and Yuan Zhang and Pengfei Wan and Zhongyuan Wang and Yu Wu},
  journal={arXiv preprint arXiv:2312.13305},
  year={2023},
}

@article{dvisdaq,
  title={DVIS-DAQ: Improving Video Segmentation via Dynamic Anchor Queries}, 
  author={Yikang Zhou and Tao Zhang and Shunping Ji and Shuicheng Yan and Xiangtai Li},
  journal={arXiv},
  year={2024},
}
```

## Acknowledgement

This repo is largely based on [Mask2Former](https://github.com/facebookresearch/Mask2Former), 
[MinVIS](https://github.com/NVlabs/MinVIS), [VITA](https://github.com/sukjunhwang/VITA),
[CTVIS](https://github.com/KainingYing/CTVIS),
[FC-CLIP](https://github.com/bytedance/fc-clip) and [DVIS](https://github.com/zhang-tao-whu/DVIS).
Thanks for their excellent works.
