## Getting Started with DVIS

This document provides a brief intro of the usage of DVIS-DAQ.

Please see [Getting Started with Detectron2](https://github.com/facebookresearch/detectron2/blob/master/GETTING_STARTED.md) for full usage.

We provide a script `train_net_video.py`  which is designed to train and evaluate
all the configs provided in `configs/dvis_daq`.

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
```

### Training


To train a model with `train_net_video.py`, first setup the corresponding datasets following
[datasets/README.md](./datasets/README.md), then download the pre-trained weights from [DVIS++ repository](../DVIS_Plus/MODEL_ZOO.md) and put them in the current working directory.
Once these are set up, run:
```
# train the DVIS-DAQ(online), take ovis dataset as example
python train_net_video.py \
  --num-gpus $NUM_GPUS \
  --config-file configs/dvis_daq/ovis/DAQ_Online_R50.py \
  --resume MODEL.WEIGHTS /path/to/segmenter_pretrained_weights.pth \
  SOLVER.IMS_PER_BATCH $NUM_GPUS

# train the DVIS-DAQ(offline)
python train_net_video.py \
  --num-gpus $NUM_GPUS \
  --config-file configs/dvis_daq/ovis/DAQ_Offline_R50.py \
  --resume MODEL.WEIGHTS /path/to/online_weights.pth \
  SOLVER.IMS_PER_BATCH $NUM_GPUS
```

