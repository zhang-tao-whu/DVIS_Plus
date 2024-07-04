# DVIS-DAQ Model Zoo

## Introduction

This file documents a collection of trained DVIS-DAQ models.
The "Config" column contains a link to the config file.

## Weights

The weights for all the following can be found on (HuggingFace)[https://huggingface.co/zhangtao-whu/DVIS_Plus/tree/main/DVIS%2B%2B], and you can also download them from there.
### OVIS
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP50</th>
<th valign="bottom">AP75</th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->

<!-- ROW: R50 Online -->
 <tr><td align="center">Online</td>
<td align="center">R50</td>
<td align="center">38.7</td>
<td align="center">65.5</td>
<td align="center">37.6</td>
<td align="center"><a href="configs/dvis_daq/ovis/DAQ_Online_R50.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/zhouyik/DVIS-DAQ">🤗model_ovis_online_r50</a></td>
</tr>

<!-- ROW: R50 Offline -->
 <tr><td align="center">Offline</td>
<td align="center">R50</td>
<td align="center">43.5</td>
<td align="center">72.5</td>
<td align="center">43.8</td>
<td align="center"><a href="configs/dvis_daq/ovis/DAQ_Offline_R50.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/zhouyik/DVIS-DAQ">🤗model_ovis_offline_r50</a></td>
</tr>

<!-- ROW: Swin-L Online -->
 <tr><td align="center">Online</td>
<td align="center">Swin-L</td>
<td align="center">49.5</td>
<td align="center">76.1</td>
<td align="center">51.7</td>
<td align="center"><a href="configs/dvis_daq/ovis/DAQ_Online_SwinL.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/zhouyik/DVIS-DAQ">🤗model_ovis_online_swinl</a></td>
</tr>

<!-- ROW: VIT-L Online -->
 <tr><td align="center">Online</td>
<td align="center">VIT-L</td>
<td align="center">53.7</td>
<td align="center">78.7</td>
<td align="center">58.2</td>
<td align="center"><a href="configs/dvis_daq/ovis/vit_adapter/DAQ_Online_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/zhouyik/DVIS-DAQ">🤗model_ovis_online_vitl</a></td>
</tr>

<!-- ROW: VIT-L Offline -->
 <tr><td align="center">Offline</td>
<td align="center">VIT-L</td>
<td align="center">57.1</td>
<td align="center">83.8</td>
<td align="center">62.9</td>
<td align="center"><a href="configs/dvis_daq/ovis/vit_adapter/DAQ_Offline_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/zhouyik/DVIS-DAQ">🤗model_ovis_offline_vitl</a></td>
</tr>

</tbody></table>

### YouTubeVIS 2019

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP50</th>
<th valign="bottom">AP75</th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->

<!-- ROW: R50 Online -->
 <tr><td align="center">Online</td>
<td align="center">R50</td>
<td align="center">55.2</td>
<td align="center">78.7</td>
<td align="center">61.9</td>
<td align="center"><a href="configs/dvis_daq/ytvis19/DAQ_Online_R50.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/zhouyik/DVIS-DAQ">🤗model_ytvis19_online_r50</a></td>
</tr>

<!-- ROW: Swin-L Online -->
 <tr><td align="center">Online</td>
<td align="center">Swin-L</td>
<td align="center">65.7</td>
<td align="center">88.1</td>
<td align="center">73.6</td>
<td align="center"><a href="configs/dvis_daq/ytvis19/DAQ_Online_SwinL.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/zhouyik/DVIS-DAQ">🤗model_ytvis19_online_swinl</a></td>
</tr>

<!-- ROW: VIT-L Online -->
 <tr><td align="center">Online</td>
<td align="center">VIT-L</td>
<td align="center">68.3</td>
<td align="center">88.5</td>
<td align="center">76.1</td>
<td align="center"><a href="configs/dvis_daq/ytvis19/vit_adapter/DAQ_Online_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/zhouyik/DVIS-DAQ">🤗model_ytvis19_online_vitl</a></td>
</tr>

<!-- ROW: VIT-L Offline -->
 <tr><td align="center">Offline</td>
<td align="center">VIT-L</td>
<td align="center">69.2</td>
<td align="center">90.8</td>
<td align="center">76.8</td>
<td align="center"><a href="configs/dvis_daq/ytvis19/vit_adapter/DAQ_Offline_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/zhouyik/DVIS-DAQ">🤗model_ytvis19_offline_vitl</a></td>
</tr>

</tbody></table>

### YouTubeVIS 2021

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">AP</th>
<th valign="bottom">AP50</th>
<th valign="bottom">AP75</th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->

<!-- ROW: R50 Online -->
 <tr><td align="center">Online</td>
<td align="center">R50</td>
<td align="center">50.4</td>
<td align="center">72.4</td>
<td align="center">55.0</td>
<td align="center"><a href="configs/dvis_daq/ytvis21/DAQ_Online_R50.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/zhouyik/DVIS-DAQ">🤗model_ytvis21_online_r50</a></td>
</tr>

<!-- ROW: Swin-L Online -->
 <tr><td align="center">Online</td>
<td align="center">Swin-L</td>
<td align="center">61.1</td>
<td align="center">84.2</td>
<td align="center">68.2</td>
<td align="center"><a href="configs/dvis_daq/ytvis21/DAQ_Online_SwinL.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/zhouyik/DVIS-DAQ">🤗model_ytvis21_online_swinl</a></td>
</tr>

<!-- ROW: VIT-L Online -->
 <tr><td align="center">Online</td>
<td align="center">VIT-L</td>
<td align="center">62.4</td>
<td align="center">83.6</td>
<td align="center">70.8</td>
<td align="center"><a href="configs/dvis_daq/ytvis21/vit_adapter/DAQ_Online_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/zhouyik/DVIS-DAQ">🤗model_ytvis21_online_vitl</a></td>
</tr>

<!-- ROW: VIT-L Offline -->
 <tr><td align="center">Offline</td>
<td align="center">VIT-L</td>
<td align="center">64.5</td>
<td align="center">86.3</td>
<td align="center">72.4</td>
<td align="center"><a href="configs/dvis_daq/ytvis21/vit_adapter/DAQ_Offline_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/zhouyik/DVIS-DAQ">🤗model_ytvis21_offline_vitl</a></td>
</tr>

</tbody></table>

### VIPSeg

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Model</th>
<th valign="bottom">Backbone</th>
<th valign="bottom">VPQ</th>
<th valign="bottom">VPQ<sub>thing</sub> </th>
<th valign="bottom">VPQ<sub>stuff</sub> </th>
<th valign="bottom">Config</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->

<!-- ROW: R50 Online -->
 <tr><td align="center">Online</td>
<td align="center">R50</td>
<td align="center">42.1</td>
<td align="center">41.7</td>
<td align="center">42.5</td>
<td align="center"><a href="configs/dvis_daq/vipseg/DAQ_Online_R50.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/zhouyik/DVIS-DAQ">🤗model_vipseg_online_r50</a></td>
</tr>

<!-- ROW: VIT-L Online -->
 <tr><td align="center">Online</td>
<td align="center">VIT-L</td>
<td align="center">57.4</td>
<td align="center">60.4</td>
<td align="center">54.7</td>
<td align="center"><a href="configs/dvis_daq/vipseg/vit_adapter/DAQ_Online_VitAdapterL.yaml">yaml</a></td>
<td align="center"><a href="https://huggingface.co/zhouyik/DVIS-DAQ">🤗model_vipseg_online_vitl</a></td>
</tr>


</tbody></table>






