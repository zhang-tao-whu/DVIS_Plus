import math
import random
from typing import Tuple, List
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import fvcore.nn.weight_init as weight_init
from scipy.optimize import linear_sum_assignment
import numpy as np

from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import SelfAttentionLayer,\
    CrossAttentionLayer, FFNLayer, MLP, _get_activation_fn
from dvis_Plus.tracker import ReferringCrossAttentionLayer
from .slot_attention import SlotCrossAttentionLayer

class VideoInstanceSequence(object):
    def __init__(self, start_time: int, matched_gt_id: int = -1, maximum_chache=10):
        self.sT = start_time
        self.eT = -1
        self.maximum_chache = maximum_chache
        self.dead = False
        self.gt_id = matched_gt_id
        self.invalid_frames = 0
        self.embeds = []
        self.slots = []
        self.slot_disappear = []
        # self.pos_embeds = []
        self.pred_logits = []
        self.pred_masks = []
        self.appearance = []

        # CTVIS
        self.pos_embeds = []
        self.long_scores = []
        self.similarity_guided_pos_embed = None
        self.similarity_guided_pos_embed_list = []
        self.momentum = 0.75

        self.reid_embeds = []
        self.similarity_guided_reid_embed = None
        self.similarity_guided_reid_embed_list = []

    def update(self, reid_embed):
        self.reid_embeds.append(reid_embed)

        if len(self.similarity_guided_reid_embed_list) == 0:
            self.similarity_guided_reid_embed = reid_embed
            self.similarity_guided_reid_embed_list.append(reid_embed)
        else:
            assert len(self.reid_embeds) > 1
            # Similarity-Guided Feature Fusion
            # https://arxiv.org/abs/2203.14208v1
            all_reid_embed = []
            for embedding in self.reid_embeds[:-1]:
                all_reid_embed.append(embedding)
            all_reid_embed = torch.stack(all_reid_embed, dim=0)

            similarity = torch.sum(torch.einsum("bc,c->b",
                                                F.normalize(all_reid_embed, dim=-1),
                                                F.normalize(reid_embed.squeeze(), dim=-1)
                                                )) / all_reid_embed.shape[0]
            beta = max(0, similarity)
            self.similarity_guided_reid_embed = (1 - beta) * self.similarity_guided_reid_embed + beta * reid_embed
            self.similarity_guided_reid_embed_list.append(self.similarity_guided_reid_embed)

        if len(self.reid_embeds) > self.maximum_chache:
            self.reid_embeds.pop(0)

    def update_pos(self, pos_embed):
        self.pos_embeds.append(pos_embed)

        if len(self.similarity_guided_pos_embed_list) == 0:
            self.similarity_guided_pos_embed = pos_embed
            self.similarity_guided_pos_embed_list.append(pos_embed)
        else:
            assert len(self.pos_embeds) > 1
            # Similarity-Guided Feature Fusion
            # https://arxiv.org/abs/2203.14208v1
            all_pos_embed = []
            # for embedding in self.pos_embeds[:-1]:
            #     all_pos_embed.append(embedding)
            sidx = len(self.pos_embeds) - self.maximum_chache
            sidx = max(0, sidx)
            for embedding in self.pos_embeds[sidx:-1]:
                all_pos_embed.append(embedding)
            all_pos_embed = torch.stack(all_pos_embed, dim=0)

            similarity = torch.sum(torch.einsum("bc,c->b",
                                                F.normalize(all_pos_embed, dim=-1),
                                                F.normalize(pos_embed.squeeze(), dim=-1)
                                                )) / all_pos_embed.shape[0]

            # TODO, using different similarity function
            beta = max(0, similarity)
            self.similarity_guided_pos_embed = (1 - beta) * self.similarity_guided_pos_embed + beta * pos_embed
            self.similarity_guided_pos_embed_list.append(self.similarity_guided_pos_embed)

        # if len(self.pos_embeds) > self.maximum_chache:
        #     self.pos_embeds.pop(0)


class VideoInstanceCutter(nn.Module):

    def __init__(
            self,
            hidden_dim: int = 256,
            feedforward_dim: int = 2048,
            num_head: int = 8,
            decoder_layer_num: int = 6,
            mask_dim: int = 256,
            num_classes: int = 25,
            num_new_ins: int = 100,
            training_select_threshold: float = 0.1,
            # inference
            inference_select_threshold: float = 0.1,
            kick_out_frame_num: int = 8,
            mask_nms_thr: float = 0.6,
            match_score_thr: float = 0.3,
            num_slots: int = 5,
            keep_threshold: float = 0.01,
            task: str = "vis",
            ovis_infer: bool = True
    ):
        super().__init__()

        self.num_heads = num_head
        self.hidden_dim = hidden_dim
        self.num_layers = decoder_layer_num
        self.num_classes = num_classes
        self.task = task
        self.ovis_infer = ovis_infer
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.slot_cross_attention_layers = nn.ModuleList()
        self.slot_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=feedforward_dim,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.slot_cross_attention_layers.append(
                SlotCrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.slot_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=feedforward_dim,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.pos_embed = MLP(mask_dim, hidden_dim, hidden_dim, 3)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        # mask features projection
        self.mask_feature_proj = nn.Conv2d(
            mask_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0
        )

        self.new_ins_embeds = nn.Embedding(1, hidden_dim)
        self.bg_slots = nn.Embedding(num_slots, hidden_dim)

        # record previous frame information
        self.last_seq_ids = None
        self.track_queries = None
        self.track_embeds = None
        self.cur_disappear_embeds = None
        self.prev_frame_indices = None
        self.tgt_ids_for_track_queries = None
        self.disappear_fq_mask = None
        self.disappear_tgt_id = None
        self.disappear_trcQ_id = None
        self.disappeared_tgt_ids = []
        self.exist_frames = None
        self.prev_valid_frame_embeds = None
        self.prev_valid_frame_embeds_indices = None
        self.prev_frame_mask_area = None
        self.memory_seq_ids = []  # as dead ins_seq will be removed from self.video_ins_hub
        self.video_ins_hub = dict()
        self.gt_ins_hub = dict()

        self.num_new_ins = num_new_ins
        self.num_slots = num_slots
        self.training_select_thr = training_select_threshold
        self.inference_select_thr = inference_select_threshold
        self.kick_out_frame_num = kick_out_frame_num
        self.mask_nms_thr = mask_nms_thr
        self.match_score_thr = match_score_thr
        self.keep_threshold = keep_threshold

        self.point_seq_id = None


    def _clear_memory(self):
        del self.video_ins_hub
        self.video_ins_hub = dict()
        self.gt_ins_hub = dict()
        self.last_seq_ids = None
        self.track_queries = None
        self.track_embeds = None
        self.cur_disappear_embeds = None
        self.prev_frame_indices = None
        self.tgt_ids_for_track_queries = None
        self.disappear_fq_mask = None
        self.disappear_tgt_id = None
        self.disappeared_tgt_ids = []
        self.disappear_trcQ_id = None
        self.exist_frames = None
        self.prev_frame_mask_area = None
        return

    def readout(self, read_type: str = "last"):
        assert read_type in ["last", "last_pos", "gt_ids"]

        if read_type == "last":
            out_embeds = []
            for seq_id in self.last_seq_ids:
                idx = -1
                while self.video_ins_hub[seq_id].embeds[idx] is None:
                    idx = idx - 1
                out_embeds.append(self.video_ins_hub[seq_id].embeds[idx])
            if len(out_embeds):
                return torch.stack(out_embeds, dim=0).unsqueeze(1)  # q, 1, c
            else:
                return torch.empty(size=(0, 1, self.hidden_dim), dtype=torch.float32).to("cuda")
        elif read_type == "last_pos":
            out_pos_embeds = []
            for seq_id in self.last_seq_ids:
                out_pos_embeds.append(self.video_ins_hub[seq_id].similarity_guided_pos_embed)
            if len(out_pos_embeds):
                return torch.stack(out_pos_embeds, dim=0).unsqueeze(1)  # q, 1, c
            else:
                return torch.empty(size=(0, 1, self.hidden_dim), dtype=torch.float32).to("cuda")
        elif read_type == "gt_ids":
            out_ids = []
            for seq_id in self.last_seq_ids:
                out_ids.append(self.video_ins_hub[seq_id].gt_id)
            if len(out_ids):
                return torch.stack(out_ids, dim=0)
            else:
                return torch.empty(size=(0, ), dtype=torch.int64).to("cuda")
        else:
            raise NotImplementedError

    def modeling_disappear(self, frames_info, frame_idx, stage=2):
        fQ = len(frames_info["aux_indices"][frame_idx][0][1])
        disappear_fq_mask = torch.zeros(size=(fQ, ), dtype=torch.bool).to("cuda")
        if self.prev_frame_indices is not None and len(self.prev_frame_indices[0]) > 3:
            select_idx = random.randrange(0, len(self.prev_frame_indices[0]))
            select_tgt_id = self.prev_frame_indices[1][select_idx]
            if stage == 2 or select_tgt_id == -1 or self.task == "vps":
                self.disappear_tgt_id = None
                self.disappear_trcQ_id = None
            else:
                aux_tgt_i_for_each_fq = frames_info["aux_indices"][frame_idx][0][1]
                disappear_fq_mask[aux_tgt_i_for_each_fq == select_tgt_id] = True
                self.disappear_tgt_id = select_tgt_id
                self.disappear_trcQ_id = self.prev_frame_indices[0][select_idx]
        else:
            self.disappear_tgt_id = None
            self.disappear_trcQ_id = None
        return ~disappear_fq_mask

    def forward(self, frame_embeds, mask_features, targets, frames_info, matcher, resume=False, stage=1):
        ori_mask_features = mask_features
        mask_features_shape = mask_features.shape
        mask_features = self.mask_feature_proj(mask_features.flatten(0, 1)).reshape(*mask_features_shape)  # (b, t, c, h, w)

        frame_embeds = frame_embeds.permute(2, 3, 0, 1)  # t, q, b, c
        T, fQ, B, _ = frame_embeds.shape
        assert B == 1
        all_outputs, all_slot_outputs = [], []

        seg_query_feat = frames_info["seg_query_feat"].weight.unsqueeze(1).repeat(1, B, 1)
        new_ins_embeds = self.new_ins_embeds.weight.unsqueeze(1).repeat(self.num_new_ins, B, 1)  # nq, b, c
        bg_slot_embeds = self.bg_slots.weight.unsqueeze(1).repeat(1, B, 1)
        for i in range(T):
            ms_outputs, slot_outputs = [], []
            single_frame_embeds = frame_embeds[i]  # q, b, c
            targets_i = targets[i].copy()
            valid_fq_bool = frames_info["valid"][i][0]
            if i == 0 and resume is False:
                self._clear_memory()
                output = single_frame_embeds
                ms_outputs.append(output)
                for j in range(self.num_layers):
                    output = self.transformer_cross_attention_layers[j](
                        output, single_frame_embeds,
                    )
                    output = self.transformer_self_attention_layers[j](output)
                    output = self.transformer_ffn_layers[j](output)
                    ms_outputs.append(output)
            else:
                frame_queries_pos, _ = self.get_mask_pos_embed(frames_info["pred_masks"][i][0][None],
                                                               ori_mask_features[:, i, ...])

                trc_det_queries = torch.cat([self.track_queries, new_ins_embeds])
                trc_det_queries_pos = torch.cat([self.track_embeds, frame_queries_pos])

                ms_outputs.append(trc_det_queries)
                for j in range(self.num_layers):
                    trc_det_queries = self.transformer_cross_attention_layers[j](
                        trc_det_queries, single_frame_embeds,
                        query_pos=trc_det_queries_pos,  pos=frame_queries_pos,
                    )
                    trc_det_queries = self.transformer_self_attention_layers[j](trc_det_queries)
                    trc_det_queries = self.transformer_ffn_layers[j](trc_det_queries)
                    ms_outputs.append(trc_det_queries)

                # slot attention
                sq_id_for_tq = self.match_with_embeds(torch.cat([self.track_queries, bg_slot_embeds]), seg_query_feat)
                slots_feats = seg_query_feat[sq_id_for_tq]
                slots_pos = torch.cat([self.track_queries, bg_slot_embeds], dim=0)
                slots_query = torch.cat([self.track_embeds, bg_slot_embeds], dim=0)
                valid_appear_fq_mask = self.modeling_disappear(frames_info, i, stage)
                attn_mask = ~valid_appear_fq_mask[None, None, None, :].repeat(B, self.num_heads, slots_feats.shape[0],1)
                attn_mask = attn_mask.flatten(0, 1)
                for j in range(self.num_layers):
                    slots_feats = self.slot_cross_attention_layers[j](
                        slots_feats, single_frame_embeds,
                        query_pos=slots_pos,  memory_mask=attn_mask,
                        slot_query=slots_query,
                    )
                    slots_feats = self.slot_ffn_layers[j](slots_feats)
                    slot_outputs.append(slots_feats)

            ms_outputs = torch.stack(ms_outputs, dim=0)  # (L, tQ+nQ, B, C)
            ms_outputs_class, ms_outputs_mask = self.prediction(ms_outputs, mask_features[:, i, ...])

            out_dict = {
                "pred_logits": ms_outputs_class[-1],  # b, q, k+1
                "pred_masks": ms_outputs_mask[-1],  # b, q, h, w
            }

            # matching with gt
            if self.prev_frame_indices is None:
                indices = frames_info["indices"][i]
            else:
                indices = matcher(out_dict, targets_i, self.prev_frame_indices)

            out_dict.update({
                "indices": indices,
                "aux_outputs": self._set_aux_loss(ms_outputs_class, ms_outputs_mask, self.disappeared_tgt_ids),
                "disappear_tgt_ids": self.disappeared_tgt_ids if self.disappeared_tgt_ids is not None else [],
                "slot_out": False
            })
            all_outputs.append(out_dict)

            if len(slot_outputs) != 0:
                slot_outputs = torch.stack(slot_outputs, dim=0)  # L, tQ, B, C
                slot_outputs_class, slot_outputs_mask = self.prediction(slot_outputs, mask_features[:, i, ...])  # TODO, use another prediction head

                slot_disappeared_tgt_ids = self.disappeared_tgt_ids if self.disappeared_tgt_ids is not None else []
                if self.disappear_tgt_id is not None:
                    slot_disappeared_tgt_ids.append(self.disappear_tgt_id)
                slot_out_dict = {
                    "pred_logits": slot_outputs_class[-1],  # b, tq, k+1
                    "pred_masks": slot_outputs_mask[-1],  # b, tq, h, w
                    "indices": [self.prev_frame_indices],
                    "aux_outputs": self._set_aux_loss(slot_outputs_class, slot_outputs_mask, slot_disappeared_tgt_ids,
                                                      slot_out=True, stage=stage),
                    "disappear_tgt_ids": slot_disappeared_tgt_ids,
                    "slot_out": True,
                    "stage": stage
                }
                all_slot_outputs.append(slot_out_dict)

            if stage == 1:
                tgt_ids_for_each_query = torch.full(size=(ms_outputs.shape[1],), dtype=torch.int64,
                                                    fill_value=-1).to("cuda")
                tgt_ids_for_each_query[indices[0][0]] = indices[0][1]
                activated_queries_bool = torch.ones(size=(ms_outputs.shape[1], )).to("cuda") < 0
            elif stage == 2:
                tgt_ids_for_each_query = torch.full(size=(ms_outputs.shape[1],), dtype=torch.int64,
                                                    fill_value=-1).to("cuda")
                tgt_ids_for_each_query[indices[0][0]] = indices[0][1]

                pred_scores = torch.max(ms_outputs_class[-1, 0].softmax(-1)[:, :-1], dim=-1)[0]
                pred_scores = pred_scores[indices[0][0]]
                sorted_idx = torch.argsort(pred_scores, 0)
                kick_out_src_indices = indices[0][0][sorted_idx[:len(pred_scores)//2]]

                activated_queries_bool = torch.ones(size=(ms_outputs.shape[1], )).to("cuda") < 0
                activated_queries_bool[indices[0][0]] = True
                activated_queries_bool[kick_out_src_indices] = False
            elif stage == 3:
                tgt_ids_for_each_query = torch.full(size=(ms_outputs.shape[1],), dtype=torch.int64,
                                                    fill_value=-1).to("cuda")
                tgt_ids_for_each_query[indices[0][0]] = indices[0][1]

                pred_scores = torch.max(ms_outputs_class[-1, 0].softmax(-1)[:, :-1], dim=-1)[0]
                activated_queries_bool = pred_scores > self.training_select_thr
            else:
                raise NotImplementedError

            self.track_queries = ms_outputs[-1][activated_queries_bool]  # q', b, c
            select_query_tgt_ids = tgt_ids_for_each_query[activated_queries_bool]  # q',
            prev_src_indices = torch.nonzero(select_query_tgt_ids + 1).squeeze(-1)
            prev_tgt_indices = torch.index_select(select_query_tgt_ids, dim=0, index=prev_src_indices)
            self.prev_frame_indices = (prev_src_indices, prev_tgt_indices)
            self.tgt_ids_for_track_queries = tgt_ids_for_each_query[activated_queries_bool]

            _pred_masks = ms_outputs_mask[-1, :, activated_queries_bool, :, :][:, select_query_tgt_ids + 1 > 0, :, :]
            self.prev_frame_mask_area = (_pred_masks > 0.).sum(-1).sum(-1)  # b, q'

            track_embeds, obj_embeds = self.get_mask_pos_embed(ms_outputs_mask[-1, ...],
                                                               ori_mask_features[:, i, ...])  # q', b, c
            cur_seq_ids = []
            for k, valid in enumerate(activated_queries_bool):
                if self.last_seq_ids is not None and k < len(self.last_seq_ids):
                    seq_id = self.last_seq_ids[k]
                else:
                    seq_id = random.randint(0, 100000)
                    while seq_id in self.video_ins_hub:
                        seq_id = random.randint(0, 100000)
                    assert not seq_id in self.video_ins_hub
                if valid:
                    if not seq_id in self.video_ins_hub:
                        self.video_ins_hub[seq_id] = VideoInstanceSequence(0, tgt_ids_for_each_query[k])
                    self.video_ins_hub[seq_id].update_pos(track_embeds[k, 0, :])
                    cur_seq_ids.append(seq_id)
            self.last_seq_ids = cur_seq_ids
            self.track_embeds = self.readout("last_pos")

            # detect disappear in the next frame
            disappear_gt_ids = []
            next_i = i+1 if i < T - 1 else -1
            for cur_gt_id in self.prev_frame_indices[1]:
                if cur_gt_id not in frames_info["indices"][next_i][0][1]:
                    disappear_gt_ids.append(cur_gt_id)
            if len(disappear_gt_ids) == 0:
                self.disappeared_tgt_ids = None
            else:
                self.disappeared_tgt_ids = disappear_gt_ids

        return all_outputs, all_slot_outputs

    def forward_offline_mode(self, frame_embeds, mask_features, frames_info, start_frame_id, resume=False, to_store="cuda"):
        ori_mask_features = mask_features
        mask_features_shape = mask_features.shape
        mask_features = self.mask_feature_proj(mask_features.flatten(0, 1)).reshape(
            *mask_features_shape)  # (b, t, c, h, w)

        frame_embeds = frame_embeds.permute(2, 3, 0, 1)  # t, q, b, c
        T, fQ, B, _ = frame_embeds.shape
        assert  B == 1

        seg_query_feat = frames_info["seg_query_feat"].weight.unsqueeze(1).repeat(1, B, 1)
        new_ins_embeds = self.new_ins_embeds.weight.unsqueeze(1).repeat(self.num_new_ins, B, 1)  # nq, b, c
        bg_slot_embeds = self.bg_slots.weight.unsqueeze(1).repeat(1, B, 1)
        for i in range(T):
            single_frame_embeds = frame_embeds[i]  # q, b, c
            valid_fq_mask = frames_info["valid"][i][0]
            if i == 0 and resume is False:
                self._clear_memory()
                output = single_frame_embeds
                slot_output = None
                for j in range(self.num_layers):
                    output = self.transformer_cross_attention_layers[j](
                        output, single_frame_embeds,
                    )
                    output = self.transformer_self_attention_layers[j](output)
                    output = self.transformer_ffn_layers[j](output)
            else:
                frame_queries_pos, _ = self.get_mask_pos_embed(frames_info["pred_masks"][i][0][None],
                                                               ori_mask_features[:, i, ...])

                trc_det_queries = torch.cat([self.track_queries, new_ins_embeds])
                trc_det_queries_pos = torch.cat([self.track_embeds, frame_queries_pos])

                for j in range(self.num_layers):
                    trc_det_queries = self.transformer_cross_attention_layers[j](
                        trc_det_queries, single_frame_embeds,
                        query_pos=trc_det_queries_pos, pos=frame_queries_pos,
                    )
                    trc_det_queries = self.transformer_self_attention_layers[j](trc_det_queries)
                    trc_det_queries = self.transformer_ffn_layers[j](trc_det_queries)
                output = trc_det_queries

                sq_id_for_tq = self.match_with_embeds(torch.cat([self.track_queries, bg_slot_embeds]),
                                                      seg_query_feat)
                slots_feats = seg_query_feat[sq_id_for_tq]
                slots_pos = torch.cat([self.track_queries, bg_slot_embeds], dim=0)
                slots_query = torch.cat([self.track_embeds, bg_slot_embeds], dim=0)
                for j in range(self.num_layers):
                    slots_feats = self.slot_cross_attention_layers[j](
                        slots_feats, single_frame_embeds,
                        query_pos=slots_pos,
                        slot_query=slots_query,
                    )
                    slots_feats = self.slot_ffn_layers[j](slots_feats)
                slot_output = slots_feats[:self.track_queries.shape[0]]

            ms_outputs_class, ms_outputs_mask = self.prediction(output.unsqueeze(0), mask_features[:, i, ...])
            if slot_output is not None:
                slot_outputs_class, slot_outputs_mask = self.prediction(slot_output[None], mask_features[:, i, ...])
            else:
                slot_outputs_class, slot_outputs_mask = None, None
            track_embeds, _ = self.get_mask_pos_embed(ms_outputs_mask[-1, ...],
                                                      ori_mask_features[:, i, ...])  # q', b, c

            cur_seq_ids = []
            if i == 0 and resume is False:
                valid_queries_bool = valid_fq_mask
            else:
                if self.ovis_infer:
                    num_tq = self.track_queries.shape[0]
                    trc_queries_logits = slot_outputs_class[-1, 0, :num_tq] * 0.5 + ms_outputs_class[-1, 0, :num_tq] * 0.5
                    trc_queries_scores = torch.max(trc_queries_logits.softmax(-1)[:, :-1], dim=1)[0]
                    det_queries_scores = torch.max(ms_outputs_class[-1, 0, -self.num_new_ins:].softmax(-1)[:, :-1], dim=1)[0]
                    valid_trc_queries_bool = trc_queries_scores > self.inference_select_thr
                    valid_det_queries_bool = det_queries_scores > self.inference_select_thr
                    valid_queries_bool = torch.cat([valid_trc_queries_bool, valid_det_queries_bool], dim=0)
                    assert valid_queries_bool.shape[0] == output.shape[0]
                else:
                    pred_scores = torch.max(ms_outputs_class[-1, 0].softmax(-1)[:, :-1], dim=1)[0]
                    valid_queries_bool = pred_scores > self.inference_select_thr
            for k, valid in enumerate(valid_queries_bool):
                if self.last_seq_ids is not None and k < len(self.last_seq_ids):
                    seq_id = self.last_seq_ids[k]
                else:
                    seq_id = random.randint(0, 100000)
                    while seq_id in self.video_ins_hub:
                        seq_id = random.randint(0, 100000)
                    assert not seq_id in self.video_ins_hub
                if valid:
                    if not seq_id in self.video_ins_hub:
                        self.video_ins_hub[seq_id] = VideoInstanceSequence(start_frame_id + i, seq_id)
                    self.video_ins_hub[seq_id].embeds.append(output[k, 0, :])
                    self.video_ins_hub[seq_id].update_pos(track_embeds[k, 0, :])

                    self.video_ins_hub[seq_id].pred_logits.append(ms_outputs_class[-1, 0, k, :])
                    if to_store == "cpu":
                        self.video_ins_hub[seq_id].pred_masks.append(
                            ms_outputs_mask[-1, 0, k, ...].to(to_store).to(torch.float32))
                    else:
                        self.video_ins_hub[seq_id].pred_masks.append(
                            ms_outputs_mask[-1, 0, k, ...])
                    self.video_ins_hub[seq_id].invalid_frames = 0
                    self.video_ins_hub[seq_id].appearance.append(True)

                    cur_seq_ids.append(seq_id)
                elif self.last_seq_ids is not None and seq_id in self.last_seq_ids:
                    self.video_ins_hub[seq_id].invalid_frames += 1
                    if self.video_ins_hub[seq_id].invalid_frames >= self.kick_out_frame_num:
                        self.video_ins_hub[seq_id].dead = True
                        continue
                    self.video_ins_hub[seq_id].embeds.append(output[k, 0, :])

                    self.video_ins_hub[seq_id].pred_logits.append(ms_outputs_class[-1, 0, k, :])
                    # self.video_ins_hub[seq_id].pred_masks.append(
                    #     ms_outputs_mask[-1, 0, k, ...].to(to_store).to(torch.float32))
                    if to_store == "cpu":
                        self.video_ins_hub[seq_id].pred_masks.append(
                            ms_outputs_mask[-1, 0, k, ...].to(to_store).to(torch.float32))
                    else:
                        self.video_ins_hub[seq_id].pred_masks.append(
                            ms_outputs_mask[-1, 0, k, ...])
                    self.video_ins_hub[seq_id].appearance.append(False)

                    cur_seq_ids.append(seq_id)
            self.last_seq_ids = cur_seq_ids
            self.track_queries = self.readout("last")
            self.track_embeds = self.readout("last_pos")

    def inference(self, frame_embeds, mask_features, frames_info, start_frame_id, resume=False, to_store="cpu"):

        ori_mask_features = mask_features
        mask_features_shape = mask_features.shape
        mask_features = self.mask_feature_proj(mask_features.flatten(0, 1)).reshape(
            *mask_features_shape)  # (b, t, c, h, w)

        frame_embeds = frame_embeds.permute(2, 3, 0, 1)  # t, q, b, c
        T, fQ, B, _ = frame_embeds.shape
        assert B == 1

        seg_query_feat = frames_info["seg_query_feat"].weight.unsqueeze(1).repeat(1, B, 1)
        new_ins_embeds = self.new_ins_embeds.weight.unsqueeze(1).repeat(self.num_new_ins, B, 1)  # nq, b, c
        bg_slot_embeds = self.bg_slots.weight.unsqueeze(1).repeat(1, B, 1)
        for i in range(T):
            ms_outputs, slot_outputs = [], []
            single_frame_embeds = frame_embeds[i]  # q, b, c
            valid_fq_bool = frames_info["valid"][i][0]
            if i == 0 and resume is False:
                self._clear_memory()
                output = single_frame_embeds
                ms_outputs.append(output)
                for j in range(self.num_layers):
                    output= self.transformer_cross_attention_layers[j](
                        output, single_frame_embeds,
                    )
                    output = self.transformer_self_attention_layers[j](output)
                    output = self.transformer_ffn_layers[j](output)
                    ms_outputs.append(output)
                # slot_output = None
            else:
                frame_queries_pos, _ = self.get_mask_pos_embed(frames_info["pred_masks"][i][0][None],
                                                               ori_mask_features[:, i, ...])

                trc_det_queries = torch.cat([self.track_queries, new_ins_embeds])
                trc_det_queries_pos = torch.cat([self.track_embeds, frame_queries_pos])

                ms_outputs.append(trc_det_queries)
                for j in range(self.num_layers):
                    trc_det_queries= self.transformer_cross_attention_layers[j](
                        trc_det_queries, single_frame_embeds,
                        query_pos=trc_det_queries_pos, pos=frame_queries_pos,
                    )
                    trc_det_queries = self.transformer_self_attention_layers[j](trc_det_queries)
                    trc_det_queries = self.transformer_ffn_layers[j](trc_det_queries)
                    ms_outputs.append(trc_det_queries)

                sq_id_for_tq = self.match_with_embeds(torch.cat([self.track_queries, bg_slot_embeds]),
                                                      seg_query_feat)
                slots_feats = seg_query_feat[sq_id_for_tq]
                slots_pos = torch.cat([self.track_queries, bg_slot_embeds], dim=0)
                slots_query = torch.cat([self.track_embeds, bg_slot_embeds], dim=0)
                for j in range(self.num_layers):
                    slots_feats = self.slot_cross_attention_layers[j](
                        slots_feats, single_frame_embeds,
                        query_pos=slots_pos,
                        slot_query=slots_query,
                    )
                    slots_feats = self.slot_ffn_layers[j](slots_feats)
                    slot_outputs.append(slots_feats)
                # slot_output = slots_feats[:self.track_queries.shape[0]]

            ms_outputs = torch.stack(ms_outputs, dim=0)  # (L2, tQ+nQ, B, C)
            ms_outputs_class, ms_outputs_mask = self.prediction(ms_outputs, mask_features[:, i, ...])
            if len(slot_outputs) != 0:
                slot_outputs = torch.stack(slot_outputs, dim=0)
                slot_outputs = slot_outputs[-1:, ...]
                slot_outputs_class, slot_outputs_mask = self.prediction(slot_outputs, mask_features[:, i, ...])
            else:
                slot_outputs_class, slot_outputs_mask = None, None

            track_embeds, _ = self.get_mask_pos_embed(ms_outputs_mask[-1, ...],
                                                      ori_mask_features[:, i, ...])  # q', b, c

            cur_seq_ids = []
            if i == 0 and resume is False:
                valid_queries_bool = valid_fq_bool
            else:
                num_tq = self.track_queries.shape[0]

                if self.ovis_infer:
                    trc_queries_logits = ms_outputs_class[-1, 0, :num_tq]
                    trc_queries_scores = torch.max(trc_queries_logits.softmax(-1)[:, :-1], dim=1)[0]
                    fg_trc_queries_scores = torch.max(slot_outputs_class[-1, 0, :num_tq].softmax(-1)[:, :-1], dim=1)[0]
                    det_queries_scores = \
                    torch.max(ms_outputs_class[-1, 0, -self.num_new_ins:].softmax(-1)[:, :-1], dim=1)[0]
                    valid_trc_queries_bool = ((trc_queries_scores > self.inference_select_thr) &
                                              (fg_trc_queries_scores > self.keep_threshold))
                    valid_det_queries_bool = det_queries_scores > self.inference_select_thr
                    valid_queries_bool = torch.cat([valid_trc_queries_bool, valid_det_queries_bool], dim=0)
                    assert valid_queries_bool.shape[0] == ms_outputs.shape[1]
                else:
                    pred_scores = torch.max(ms_outputs_class[-1, 0].softmax(-1)[:, :-1], dim=1)[0]
                    valid_queries_bool = pred_scores > self.inference_select_thr
            for k, valid in enumerate(valid_queries_bool):
                if self.last_seq_ids is not None and k < len(self.last_seq_ids):
                    seq_id = self.last_seq_ids[k]
                else:
                    seq_id = random.randint(0, 100000)
                    while seq_id in self.video_ins_hub or seq_id in self.memory_seq_ids:
                        seq_id = random.randint(0, 100000)
                    assert (not seq_id in self.video_ins_hub) and (not seq_id in self.memory_seq_ids)
                if valid:
                    if not seq_id in self.video_ins_hub:
                        self.video_ins_hub[seq_id] = VideoInstanceSequence(start_frame_id + i, seq_id)
                        self.memory_seq_ids.append(seq_id)
                    self.video_ins_hub[seq_id].embeds.append(ms_outputs[-1, k, 0, :])
                    self.video_ins_hub[seq_id].pred_logits.append(ms_outputs_class[-1, 0, k, :])
                    if to_store == "cpu":
                        self.video_ins_hub[seq_id].pred_masks.append(
                            ms_outputs_mask[-1, 0, k, ...].to(to_store).to(torch.float32))
                    else:
                        self.video_ins_hub[seq_id].pred_masks.append(
                            ms_outputs_mask[-1, 0, k, ...])
                    self.video_ins_hub[seq_id].invalid_frames = 0
                    self.video_ins_hub[seq_id].appearance.append(True)

                    self.video_ins_hub[seq_id].update_pos(track_embeds[k, 0, :])

                    cur_seq_ids.append(seq_id)
                elif self.last_seq_ids is not None and seq_id in self.last_seq_ids:
                    self.video_ins_hub[seq_id].invalid_frames += 1
                    if self.video_ins_hub[seq_id].invalid_frames >= self.kick_out_frame_num:
                        self.video_ins_hub[seq_id].dead = True
                        continue
                    self.video_ins_hub[seq_id].embeds.append(ms_outputs[-1, k, 0, :])
                    self.video_ins_hub[seq_id].pred_logits.append(ms_outputs_class[-1, 0, k, :])
                    if to_store == "cpu":
                        self.video_ins_hub[seq_id].pred_masks.append(
                            ms_outputs_mask[-1, 0, k, ...].to(to_store).to(torch.float32))
                    else:
                        self.video_ins_hub[seq_id].pred_masks.append(
                            ms_outputs_mask[-1, 0, k, ...])
                    # self.video_ins_hub[seq_id].pred_masks.append(
                    #     torch.zeros_like(ms_outputs_mask[-1, 0, k, ...]).to(to_store).to(torch.float32))
                    self.video_ins_hub[seq_id].appearance.append(False)

                    cur_seq_ids.append(seq_id)

            self.last_seq_ids = cur_seq_ids
            self.track_queries = self.readout("last")
            self.track_embeds = self.readout("last_pos")

    def match_with_embeds(self, trc_queries_feat, seg_queries_feat):
        trc_queries_feat, seg_queries_feat = trc_queries_feat.detach()[:, 0, :], seg_queries_feat.detach()[:, 0, :]
        trc_queries_feat = trc_queries_feat / (trc_queries_feat.norm(dim=1)[:, None] + 1e-6)
        seg_queries_feat = seg_queries_feat / (seg_queries_feat.norm(dim=1)[:, None] + 1e-6)
        cos_sim = torch.mm(trc_queries_feat, seg_queries_feat.transpose(0, 1))
        C = 1 - cos_sim
        least_cost_indices = torch.min(C, dim=1)[1]  # q',

        indices = linear_sum_assignment(C.cpu())
        least_cost_indices[indices[0]] = torch.as_tensor(indices[1], dtype=torch.int64).to("cuda")
        return least_cost_indices

    def prediction(self, outputs, mask_features):
        # outputs (l, q, b, c)
        # mask_features (b, c, h, w)
        decoder_output = self.decoder_norm(outputs.transpose(1, 2))
        outputs_class = self.class_embed(decoder_output)  # l, b, q, k+1
        mask_embed = self.mask_embed(decoder_output)      # l, b, q, c
        outputs_mask = torch.einsum("lbqc,bchw->lbqhw", mask_embed, mask_features)

        return outputs_class, outputs_mask

    def get_mask_pos_embed(self, mask, mask_features):
        """
        mask: b, q, h, w
        mask_features: b, c, h, w
        """
        mask = mask.to(mask_features.device)
        pos_embeds_list, obj_embeds_list = [], []
        num_chunk = mask.shape[1] // 50 + 1
        for i in range(num_chunk):
            start = i * 50
            end = start + 50 if start + 50 < mask.shape[1] else mask.shape[1]

            seg_mask = (mask[:, start:end, :, :].sigmoid() > 0.5).to("cuda")
            mask_feats = seg_mask[:, :, None, :, :] * mask_features[:, None, ...]  # b, q, c, h, w
            pos_embeds = torch.sum(mask_feats.flatten(3, 4), dim=-1) / (
                    torch.sum(seg_mask.flatten(2, 3), dim=-1, keepdim=True) + 1e-8)
            obj_embeds_list.append(pos_embeds.transpose(0, 1))
            pos_embeds = self.pos_embed(pos_embeds)
            pos_embeds_list.append(pos_embeds.transpose(0, 1))

        return torch.cat(pos_embeds_list, dim=0), torch.cat(obj_embeds_list, dim=0)

    @torch.jit.unused
    def _set_aux_loss(self, outputs_cls, outputs_mask, disappear_tgt_id=None, slot_out=False, stage=2):
        return [{"pred_logits": a,
                 "pred_masks": b,
                 "disappear_tgt_ids": disappear_tgt_id if disappear_tgt_id is not None else [],
                 "slot_out": slot_out,
                 "stage": stage
                 } for a, b
                in zip(outputs_cls[:-1], outputs_mask[:-1])]