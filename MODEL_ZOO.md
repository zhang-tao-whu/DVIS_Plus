# DVIS++ Model Zoo

## Introduction

This file documents a collection of trained DVIS++ and OV-DVIS++ models.
The "Config" column contains a link to the config file.

## Weights

### Pretrained segmenter
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Train datasets</th>
<th valign="bottom">Used for</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->

<!-- ROW: DINO V2 VITL -->
 <tr><td align="center">VIT-L(DINOv2)</td>
<td align="center">VIT-L</td>
<td align="center">-</td>
<td align="center">-</td>
<td align="center"><a href="https://pan.baidu.com/s/14GniU-RD-CoH_l6wDgdcBw?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: R50 Mask2Former(instance) -->
 <tr><td align="center">Mask2Former(instance)</td>
<td align="center">R50</td>
<td align="center">COCO</td>
<td align="center">OVIS,YTVIS19&21</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/instance/maskformer2_R50_bs16_50ep/model_final_3c8ec9.pkl">link</a></td>
</tr>

<!-- ROW: VIT-L Mask2Former(instance) -->
 <tr><td align="center">Mask2Former(instance)</td>
<td align="center">VIT-L</td>
<td align="center">COCO</td>
<td align="center">OVIS,YTVIS19&21</td>
<td align="center"><a href="https://pan.baidu.com/s/14GniU-RD-CoH_l6wDgdcBw?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: R50 Mask2Former(panoptic) -->
 <tr><td align="center">Mask2Former(panoptic)</td>
<td align="center">R50</td>
<td align="center">COCO</td>
<td align="center">VSPW,VIPSeg</td>
<td align="center"><a href="https://dl.fbaipublicfiles.com/maskformer/mask2former/coco/panoptic/maskformer2_R50_bs16_50ep/model_final_94dc52.pkl">link</a></td>
</tr>

<!-- ROW: VIT-L Mask2Former(panoptic) -->
 <tr><td align="center">Mask2Former(panoptic)</td>
<td align="center">VIT-L</td>
<td align="center">COCO</td>
<td align="center">VSPW,VIPSeg</td>
<td align="center"><a href="https://pan.baidu.com/s/14GniU-RD-CoH_l6wDgdcBw?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: R50 FC-CLIP -->
 <tr><td align="center">FC-CLIP</td>
<td align="center">R50</td>
<td align="center">COCO</td>
<td align="center">OVIS,YTVIS19&21,VSPW,VIPSeg</td>
<td align="center"><a href="https://drive.google.com/file/d/1tcB-8FNON-LwckXQbUyKcBA2G7TU65Zh/view?usp=sharing">link</a></td>
</tr>

<!-- ROW: convnextl FC-CLIP -->
 <tr><td align="center">FC-CLIP</td>
<td align="center">ConvNext-L</td>
<td align="center">COCO</td>
<td align="center">OVIS,YTVIS19&21,VSPW,VIPSeg</td>
<td align="center"><a href="https://drive.google.com/file/d/1-91PIns86vyNaL3CzMmDD39zKGnPMtvj/view?usp=sharing">link</a></td>
</tr>
<tbody><table>

### Finetuned segmenter
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Train datasets</th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->

<!-- ROW: R50 Mask2Former -->
 <tr><td align="center">Mask2Former</td>
<td align="center">R50</td>
<td align="center">COCO+OVIS</td>
<td align="center"><a href="configs/dvis_Plus/ovis/CTVIS_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/14GniU-RD-CoH_l6wDgdcBw?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: R50 Mask2Former -->
 <tr><td align="center">Mask2Former</td>
<td align="center">R50</td>
<td align="center">COCO+YTVIS19</td>
<td align="center"><a href="configs/dvis_Plus/ytvis19/CTVIS_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/14GniU-RD-CoH_l6wDgdcBw?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: R50 Mask2Former -->
 <tr><td align="center">Mask2Former</td>
<td align="center">R50</td>
<td align="center">COCO+YTVIS21</td>
<td align="center"><a href="configs/dvis_Plus/ytvis21/CTVIS_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/14GniU-RD-CoH_l6wDgdcBw?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: R50 Mask2Former -->
 <tr><td align="center">Mask2Former</td>
<td align="center">R50</td>
<td align="center">VIPSeg</td>
<td align="center"><a href="configs/dvis_Plus/VIPSeg/CTVIS_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/14GniU-RD-CoH_l6wDgdcBw?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: R50 Mask2Former -->
 <tr><td align="center">Mask2Former</td>
<td align="center">R50</td>
<td align="center">VSPW</td>
<td align="center"><a href="configs/dvis_Plus/VSPW/CTVIS_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/14GniU-RD-CoH_l6wDgdcBw?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: VIT-L Mask2Former -->
 <tr><td align="center">Mask2Former</td>
<td align="center">VIT-L</td>
<td align="center">COCO+OVIS</td>
<td align="center"><a href="configs/dvis_Plus/ovis/vit_adapter/CTVIS_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/14GniU-RD-CoH_l6wDgdcBw?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: VIT-L Mask2Former -->
 <tr><td align="center">Mask2Former</td>
<td align="center">VIT-L</td>
<td align="center">COCO+YTVIS19</td>
<td align="center"><a href="configs/dvis_Plus/ytvis19/vit_adapter/CTVIS_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/14GniU-RD-CoH_l6wDgdcBw?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: VIT-L Mask2Former -->
 <tr><td align="center">Mask2Former</td>
<td align="center">VIT-L</td>
<td align="center">COCO+YTVIS21</td>
<td align="center"><a href="configs/dvis_Plus/ytvis21/vit_adapter/CTVIS_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/14GniU-RD-CoH_l6wDgdcBw?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: VIT-L Mask2Former -->
 <tr><td align="center">Mask2Former</td>
<td align="center">VIT-L</td>
<td align="center">VIPSeg</td>
<td align="center"><a href="configs/dvis_Plus/VIPSeg/vit_adapter/CTVIS_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/14GniU-RD-CoH_l6wDgdcBw?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: VIT-L Mask2Former -->
 <tr><td align="center">Mask2Former</td>
<td align="center">VIT-L</td>
<td align="center">VSPW</td>
<td align="center"><a href="configs/dvis_Plus/VSPW/vit_adapter/CTVIS_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/14GniU-RD-CoH_l6wDgdcBw?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: R50 FC-CLIP -->
 <tr><td align="center">FC-CLIP</td>
<td align="center">R50</td>
<td align="center">COCO+OVIS+YTVIS19&21+VIPSeg</td>
<td align="center"><a href="configs/open_vocabulary/R50/FC-CLIP_combine_480p_r50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/14GniU-RD-CoH_l6wDgdcBw?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: ConvNext-L FC-CLIP -->
 <tr><td align="center">FC-CLIP</td>
<td align="center">ConvNext-L</td>
<td align="center">COCO+OVIS+YTVIS19&21+VIPSeg</td>
<td align="center"><a href="configs/open_vocabulary/FC-CLIP_combine_480p.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/14GniU-RD-CoH_l6wDgdcBw?pwd=dvis">baidupan</a></td>
</tr>
<tbody><table>

### Close-vocabulary (DVIS++)
#### OVIS
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Queries</th>
<th valign="bottom">Video</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP50</th>
<th valign="bottom">AP75</th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->

<!-- ROW: R50 Online -->
 <tr><td align="center">Online</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">480P</td>
<td align="center">37.2</td>
<td align="center">62.8</td>
<td align="center">37.3</td>
<td align="center"><a href="configs/dvis_Plus/ovis/DVIS_Plus_Online_R50_16wIter.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: R50 Offline -->
 <tr><td align="center">Offline</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">480P</td>
<td align="center">41.2</td>
<td align="center">68.9</td>
<td align="center">40.9</td>
<td align="center"><a href="configs/dvis_Plus/ovis/DVIS_Plus_Offline_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: VIT-L Online -->
 <tr><td align="center">Online</td>
<td align="center">VIT-L</td>
<td align="center">200</td>
<td align="center">480P</td>
<td align="center">49.6</td>
<td align="center">72.5</td>
<td align="center">55.0</td>
<td align="center"><a href="configs/dvis_Plus/ovis/vit_adapter/DVIS_Plus_Online_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: VIT-L Offline -->
 <tr><td align="center">Offline</td>
<td align="center">VIT-L</td>
<td align="center">200</td>
<td align="center">480P</td>
<td align="center">53.4</td>
<td align="center">78.9</td>
<td align="center">58.5</td>
<td align="center"><a href="configs/dvis_Plus/ovis/vit_adapter/DVIS_Plus_Offline_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

</tbody></table>

### YouTubeVIS 2019

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Queries</th>
<th valign="bottom">Video</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP50</th>
<th valign="bottom">AP75</th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->

<!-- ROW: R50 Online -->
 <tr><td align="center">Online</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">480P</td>
<td align="center">55.5</td>
<td align="center">80.2</td>
<td align="center">60.1</td>
<td align="center"><a href="configs/dvis_Plus/ytvis19/DVIS_Plus_Online_R50_8wIter.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: R50 Offline -->
 <tr><td align="center">Offline</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">480P</td>
<td align="center">56.7</td>
<td align="center">81.4</td>
<td align="center">62.0</td>
<td align="center"><a href="configs/dvis_Plus/ytvis19/DVIS_Plus_Offline_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: VIT-L Online -->
 <tr><td align="center">Online</td>
<td align="center">VIT-L</td>
<td align="center">200</td>
<td align="center">480P</td>
<td align="center">67.7</td>
<td align="center">88.8</td>
<td align="center">75.3</td>
<td align="center"><a href="configs/dvis_Plus/ytvis19/vit_adapter/DVIS_Plus_Online_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: VIT-L Offline -->
 <tr><td align="center">Offline</td>
<td align="center">VIT-L</td>
<td align="center">200</td>
<td align="center">480P</td>
<td align="center">68.3</td>
<td align="center">90.3</td>
<td align="center">76.1</td>
<td align="center"><a href="configs/dvis_Plus/ytvis19/vit_adapter/DVIS_Plus_Offline_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

</tbody></table>

### YouTubeVIS 2021

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Queries</th>
<th valign="bottom">Video</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP50</th>
<th valign="bottom">AP75</th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->

<!-- ROW: R50 Online -->
 <tr><td align="center">Online</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">480P</td>
<td align="center">50.0</td>
<td align="center">72.2</td>
<td align="center">54.5</td>
<td align="center"><a href="configs/dvis_Plus/ytvis21/DVIS_Plus_Online_R50_8wIter.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: R50 Offline -->
 <tr><td align="center">Offline</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">480P</td>
<td align="center">52.0</td>
<td align="center">75.4</td>
<td align="center">57.8</td>
<td align="center"><a href="configs/dvis_Plus/ytvis21/DVIS_Plus_Offline_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: VIT-L Online -->
 <tr><td align="center">Online</td>
<td align="center">VIT-L</td>
<td align="center">200</td>
<td align="center">480P</td>
<td align="center">62.3</td>
<td align="center">82.7</td>
<td align="center">70.2</td>
<td align="center"><a href="configs/dvis_Plus/ytvis21/vit_adapter/DVIS_Plus_Online_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: VIT-L Offline -->
 <tr><td align="center">Offline</td>
<td align="center">VIT-L</td>
<td align="center">200</td>
<td align="center">480P</td>
<td align="center">63.9</td>
<td align="center">86.7</td>
<td align="center">71.5</td>
<td align="center"><a href="configs/dvis_Plus/ytvis21/vit_adapter/DVIS_Plus_Offline_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

</tbody></table>

### VIPSeg

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Queries</th>
<th valign="bottom">Video</th>
<th valign="bottom">VPQ</th>
<th valign="bottom">VPQ<sub>thing</sub> </th>
<th valign="bottom">VPQ<sub>stuff</sub> </th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->

<!-- ROW: R50 Online -->
 <tr><td align="center">Online</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">720P</td>
<td align="center">41.9</td>
<td align="center">41.0</td>
<td align="center">42.7</td>
<td align="center"><a href="configs/dvis_Plus/VIPSeg/DVIS_Plus_Online_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: R50 Offline -->
 <tr><td align="center">Offline</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">720P</td>
<td align="center">44.2</td>
<td align="center">44.5</td>
<td align="center">43.9</td>
<td align="center"><a href="configs/dvis_Plus/VIPSeg/DVIS_Plus_Offline_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: VIT-L Online -->
 <tr><td align="center">Online</td>
<td align="center">VIT-L</td>
<td align="center">200</td>
<td align="center">720P</td>
<td align="center">56.0</td>
<td align="center">58.0</td>
<td align="center">54.3</td>
<td align="center"><a href="configs/dvis_Plus/VIPSeg/vit_adapter/DVIS_Plus_Online_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: VIT-L Offline -->
 <tr><td align="center">Offline</td>
<td align="center">VIT-L</td>
<td align="center">200</td>
<td align="center">720P</td>
<td align="center">58.0</td>
<td align="center">61.2</td>
<td align="center">55.2</td>
<td align="center"><a href="configs/dvis_Plus/VIPSeg/vit_adapter/DVIS_Plus_Offline_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

</tbody></table>


### VSPW

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Queries</th>
<th valign="bottom">Video</th>
<th valign="bottom">VC8</th>
<th valign="bottom">VC16</th>
<th valign="bottom">mIOU</th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->

<!-- ROW: R50 Online -->
 <tr><td align="center">Online</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">720P</td>
<td align="center">92.3</td>
<td align="center">91.1</td>
<td align="center">46.9</td>
<td align="center"><a href="configs/dvis_Plus/VSPW/DVIS_Plus_Online_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: R50 Offline -->
 <tr><td align="center">Offline</td>
<td align="center">R50</td>
<td align="center">100</td>
<td align="center">720P</td>
<td align="center">93.4</td>
<td align="center">92.4</td>
<td align="center">48.6</td>
<td align="center"><a href="configs/dvis_Plus/VSPW/DVIS_Plus_Offline_R50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: VIT-L Online -->
 <tr><td align="center">Online</td>
<td align="center">VIT-L</td>
<td align="center">200</td>
<td align="center">720P</td>
<td align="center">95.0</td>
<td align="center">94.2</td>
<td align="center">62.8</td>
<td align="center"><a href="configs/dvis_Plus/VSPW/vit_adapter/DVIS_Plus_Online_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: VIT-L Offline -->
 <tr><td align="center">Offline</td>
<td align="center">VIT-L</td>
<td align="center">200</td>
<td align="center">720P</td>
<td align="center">95.7</td>
<td align="center">95.1</td>
<td align="center">63.8</td>
<td align="center"><a href="configs/dvis_Plus/VSPW/vit_adapter/DVIS_Plus_Offline_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/10rYMoLDwQ5Mb3zveO-DFAg?pwd=dvis">baidupan</a></td>
</tr>

</tbody></table>

## Open-vocabulary (OV-DVIS++)

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">Training datasets</th>
<th valign="bottom">Video</th>
<th valign="bottom">AP(OVIS)</th>
<th valign="bottom">AP(YTVIS19)</th>
<th valign="bottom">AP(YTVIS21)</th>
<th valign="bottom">mIOU(VSPW)</th>
<th valign="bottom">VPQ(VIPSeg)</th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->

<!-- ROW: zero-shot R50 Online -->
 <tr><td align="center">Online</td>
<td align="center">R50</td>
<td align="center">COCO</td>
<td align="center">480P</td>
<td align="center">14.8</td>
<td align="center">34.5</td>
<td align="center">30.9</td>
<td align="center">27.6</td>
<td align="center">24.4</td>
<td align="center"><a href="configs/open_vocabulary/R50/DVIS_Online_zero_shot_r50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1YhXmlrIkEmNGAWfgOnHjDA?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: zero-shot R50 Offline -->
 <tr><td align="center">Offline</td>
<td align="center">R50</td>
<td align="center">COCO</td>
<td align="center">480P</td>
<td align="center">13.0</td>
<td align="center">34.4</td>
<td align="center">31.0</td>
<td align="center">28.4</td>
<td align="center">23.8</td>
<td align="center"><a href="configs/open_vocabulary/R50/DVIS_Offline_zero_shot_r50.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1YhXmlrIkEmNGAWfgOnHjDA?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: zero-shot ConvNext-L Online -->
 <tr><td align="center">Online</td>
<td align="center">ConvNext-L</td>
<td align="center">COCO</td>
<td align="center">480P</td>
<td align="center">24.0</td>
<td align="center">48.8</td>
<td align="center">44.5</td>
<td align="center">34.3</td>
<td align="center">28.9</td>
<td align="center"><a href="configs/open_vocabulary/DVIS_Online_zero_shot_convnextl.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1YhXmlrIkEmNGAWfgOnHjDA?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: zero-shot ConvNext-L Offline -->
 <tr><td align="center">Offline</td>
<td align="center">ConvNext-L</td>
<td align="center">COCO</td>
<td align="center">480P</td>
<td align="center">21.6</td>
<td align="center">48.7</td>
<td align="center">44.2</td>
<td align="center">34.1</td>
<td align="center">30.4</td>
<td align="center"><a href="configs/open_vocabulary/DVIS_Offline_zero_shot_convnextl.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1YhXmlrIkEmNGAWfgOnHjDA?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: supervised ConvNext-L Online -->
 <tr><td align="center">Online</td>
<td align="center">ConvNext-L</td>
<td align="center">COCO+OVIS+YTVIS19&21+VIPSeg</td>
<td align="center">480P</td>
<td align="center">38.9</td>
<td align="center">60.1</td>
<td align="center">56.0</td>
<td align="center">53.3</td>
<td align="center">49.7</td>
<td align="center"><a href="configs/open_vocabulary/DVIS_Online_supervised_convnextl.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1YhXmlrIkEmNGAWfgOnHjDA?pwd=dvis">baidupan</a></td>
</tr>

<!-- ROW: supervised ConvNext-L Offline -->
 <tr><td align="center">Offline</td>
<td align="center">ConvNext-L</td>
<td align="center">COCO+OVIS+YTVIS19&21+VIPSeg</td>
<td align="center">480P</td>
<td align="center">40.6</td>
<td align="center">61.1</td>
<td align="center">56.7</td>
<td align="center">56.4</td>
<td align="center">51.7</td>
<td align="center"><a href="configs/open_vocabulary/DVIS_Offline_supervised_convnextl.yaml">yaml</a></td>
<td align="center"><a href="https://pan.baidu.com/s/1YhXmlrIkEmNGAWfgOnHjDA?pwd=dvis">baidupan</a></td>
</tr>
</tbody></table>






