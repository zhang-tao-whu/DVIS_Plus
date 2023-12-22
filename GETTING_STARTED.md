## Getting Started with DVIS

This document provides a brief intro of the usage of DVIS++ and OV-DVIS++.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.

### Training
We provide two scripts, `train_net_video.py` and `train_net_video_ov.py`, which are designed to train
all the configs provided in DVIS++ and OV-DVIS++ respectively.

To train a model with `train_net_video.py`/`train_net_video_ov.py`, first setup the corresponding datasets following
[datasets/README.md](./datasets/README.md), then download the pre-trained weights from [here](MODEL_ZOO.md) and put them in the current working directory.
Once these are set up, run:
```
# For close-vocabulary

# train the segmenter of the DVIS++
python train_net_video.py \
  --num-gpus $NUM_GPUS \
  --config-file /path/to/CTVIS_R50.yaml \
  --resume MODEL.WEIGHTS /path/to/mask2former_pretrained_weights.pth \
  SOLVER.IMS_PER_BATCH $NUM_GPUS

# Using contrastive learning during the training process of the segmenter requires a significant amount of GPU memory.
# If the GPU memory requirement cannot be met, contrastive learning can be omitted,
# but it may result in a performance decrease.
python train_net_video.py \
  --num-gpus $NUM_GPUS \
  --config-file /path/to/MinVIS_R50.yaml \
  --resume MODEL.WEIGHTS /path/to/mask2former_pretrained_weights.pth \
  SOLVER.IMS_PER_BATCH $NUM_GPUS

# if using multi machines
# at machine 1
python train_net_video.py \
  --num-gpus $NUM_GPUS \
  --num-machines=$NUM_MACHINES \
  --machine-rank 1 \
  --dist-url tcp://$IP_MACHINE1:13325
  --config-file /path/to/CTVIS_R50.yaml \
  --resume MODEL.WEIGHTS /path/to/mask2former_pretrained_weights.pth \
  SOLVER.IMS_PER_BATCH $NUM_GPUS*$NUM_MACHINES
# at other machines
python train_net_video.py \
  --num-gpus $NUM_GPUS \
  --num-machines=$NUM_MACHINES \
  --machine-rank $MACHINE_RANK \
  --dist-url tcp://$IP_MACHINE1:13325
  --config-file /path/to/CTVIS_R50.yaml \
  --resume MODEL.WEIGHTS /path/to/mask2former_pretrained_weights.pth \
  SOLVER.IMS_PER_BATCH $NUM_GPUS*$NUM_MACHINES

# train the DVIS++(online)
python train_net_video.py \
  --num-gpus $NUM_GPUS \
  --config-file /path/to/DVIS_Plus_Online_R50.yaml \
  --resume MODEL.WEIGHTS /path/to/segmenter_pretrained_weights.pth \
  SOLVER.IMS_PER_BATCH $NUM_GPUS

# train the DVIS++(offline)
python train_net_video.py \
  --num-gpus $NUM_GPUS \
  --config-file /path/to/DVIS_Plus_Offline_R50.yaml \
  --resume MODEL.WEIGHTS /path/to/online_weights.pth \
  SOLVER.IMS_PER_BATCH $NUM_GPUS
```

```
# For open-vocabulary

# if only use the COCO dataset for training, you can directly load
# the pre-trained weights of FC-CLIP without fine-tuning the segmenter.

# if use both the COCO dataset and video datasets for training
# finetune the segmenter of the OV-DVIS++
python train_net_video_ov.py \
  --num-gpus $NUM_GPUS \
  --config-file /path/to/open_vocabulary/FC-CLIP_combine_480p.yaml \
  --resume MODEL.WEIGHTS /path/to/fcclip_pretrained_weights.pth \
  SOLVER.IMS_PER_BATCH $NUM_GPUS

# train the OV-DVIS++(online)
python train_net_video_ov.py \
  --num-gpus $NUM_GPUS \
  --config-file /path/to/open_vocabulary/DVIS_Online_combine_480p.yaml \
  --resume MODEL.WEIGHTS /path/to/segmenter_pretrained_weights.pth \
  SOLVER.IMS_PER_BATCH $NUM_GPUS

# train the OV-DVIS++(offline)
python train_net_video_ov.py \
  --num-gpus $NUM_GPUS \
  --config-file /path/to/open_vocabulary/DVIS_Offline_combine_480p.yaml \
  --resume MODEL.WEIGHTS /path/to/online_weights.pth \
  SOLVER.IMS_PER_BATCH $NUM_GPUS
```

### Evaluation

Prepare the datasets following [datasets/README.md](./datasets/README.md) and download trained weights from [here](MODEL_ZOO.md).
Once these are set up, run:
    ```
    python train_net_video.py \
      --num-gpus 8 \
      --config-file /path/to/config.yaml \
      --eval-only MODEL.WEIGHTS /path/to/weight.pth 

    # For the ytvis and ovis datasets, 
    # you need to submit the predicted results.json to the evaluation server to obtain the evaluation results.

    # For the VIPSeg dataset
    python utils/eval_vpq_vspw.py --submit_dir output/inference \
    --truth_dir datasets/VIPSeg/VIPSeg_720P/panomasksRGB \
    --pan_gt_json_file datasets/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val.json

    python utils/eval_stq_vspw.py --submit_dir output/inference \
    --truth_dir datasets/VIPSeg/VIPSeg_720P/panomasksRGB \
    --pan_gt_json_file datasets/VIPSeg/VIPSeg_720P/panoptic_gt_VIPSeg_val.json

    # For the VSPW dataset
    python utils/eval_miou_vspw.py datasets/VSPW_480p output/inference

    python utils/eval_vc_vspw.py datasets/VSPW_480p output/inference
    ```


### Visualization

1. Pick a trained model and its config file. To start, you can pick from
  [model zoo](MODEL_ZOO.md),
  for example, `configs/ovis/DVIS_Plus_Offline_R50.yaml`. 
2. For DVIS++, we provide `demo_video/demo_long_video.py` to visualize outputs of a trained model. Run it with:
    ```
    python demo_video/demo_long_video.py \
    --config-file /path/to/config.yaml \
    --input /path/to/images_folder \
    --output /path/to/output_folder \  
    --opts MODEL.WEIGHTS /path/to/checkpoint_file.pth

    # if the video if long (> 300 frames), plese set the 'windows_size'
    python demo_video/demo_long_video.py \
    --config-file /path/to/config.yaml \
    --input /path/to/images_folder \
    --output /path/to/output_folder \  
    --windows_size 300 \
    --opts MODEL.WEIGHTS /path/to/checkpoint_file.pth
    ```
    The input is a folder containing video frames saved as images. For example, `ytvis_2019/valid/JPEGImages/00f88c4f0a`.
3. For OV-DVIS++, we provide `demo_video/open_vocabulary/demo.py` to visualize outputs of a trained
   open-vocabulary model. Run it with:
    ```
    python demo_video/open_vocabulary/demo.py \
    --config-file configs/open_vocabulary/DVIS_Offline_combine_480p_demo.yaml \
    --input /path/to/images_folder \
    --output /path/to/output_folder \  
    --merge \
    --thing_classes new_cls_1 new_cls_2 ... new_cls_n \
    --stuff_classes new_cls_1 new_cls_2 ... new_cls_n \
    --opts MODEL.WEIGHTS /path/to/checkpoint_file.pth \

    # "--merge" refers to using a merged vocabulary set of training datasets.
    # "--thing_classes" refers to the new "thing" vocabularies you want to add additionally.
    # "--stuff_classes" refers to the new "stuff" vocabularies you want to add additionally.
    # "--clear" refers to clearing the default vocabulary set and using only the additional vocabulary set provided by you.

    # a example
    python demo_video/open_vocabulary/demo.py \
    --config-file configs/open_vocabulary/DVIS_Offline_combine_480p_demo.yaml \
    --input /path/to/images_folder \
    --output /path/to/output_folder \  
    --merge \
    --thing_classes  carrot,carrots lantern,lanterns \
    --stuff_classes hay \
    --opts MODEL.WEIGHTS /path/to/checkpoint_file.pth \
    ```