from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import SelfAttentionLayer,\
    CrossAttentionLayer, FFNLayer, MLP, _get_activation_fn
from .noiser import Noiser
import torch
from torch import nn
import fvcore.nn.weight_init as weight_init

class ReferringCrossAttentionLayer(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dropout=0.0,
        activation="relu",
        normalize_before=False
    ):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        indentify,
        tgt,
        key,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(key, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(
        self,
        indentify,
        tgt,
        key,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(key, pos),
            value=memory, attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask)[0]
        tgt = indentify + self.dropout(tgt2)

        return tgt

    def forward(
        self,
        indentify,
        tgt,
        key,
        memory,
        memory_mask=None,
        memory_key_padding_mask=None,
        pos=None,
        query_pos=None
    ):
        # when set "indentify = tgt", ReferringCrossAttentionLayer is same as CrossAttentionLayer
        if self.normalize_before:
            return self.forward_pre(indentify, tgt, key, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(indentify, tgt, key, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)

class ReferringTracker_noiser(torch.nn.Module):
    def __init__(
        self,
        hidden_channel=256,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
        class_num=25,
        noise_mode='hard',
        noise_ratio=0.5,
    ):
        super(ReferringTracker_noiser, self).__init__()

        # init transformer layers
        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):

            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
                ReferringCrossAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_channel,
                    dim_feedforward=feedforward_channel,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

        self.use_memory = False
        if self.use_memory:
            self.memory_cross_attn = CrossAttentionLayer(
                d_model=hidden_channel,
                nhead=num_head,
                dropout=0.0,
                normalize_before=False,)
            self.references_memory = None

        self.decoder_norm = nn.LayerNorm(hidden_channel)

        # init heads
        self.class_embed = nn.Linear(2 * hidden_channel, class_num + 1)
        self.mask_embed = MLP(hidden_channel, hidden_channel, mask_dim, 3)

        # for cl learning
        self.ref_proj = MLP(hidden_channel, hidden_channel, hidden_channel, 3)

        for layer in self.ref_proj.layers:
            weight_init.c2_xavier_fill(layer)

        # mask features projection
        self.mask_feature_proj = nn.Conv2d(
            mask_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        # record previous frame information
        self.last_outputs = None
        self.last_frame_embeds = None
        self.last_reference = None

        self.noiser = Noiser(noise_ratio=noise_ratio, mode=noise_mode)

    def _clear_memory(self):
        del self.last_outputs
        self.last_outputs = None
        self.last_reference = None
        return

    def forward(self, frame_embeds, mask_features, resume=False,
                return_indices=False, frame_classes=None,
                frame_embeds_no_norm=None):
        """
        :param frame_embeds: the instance queries output by the segmenter
        :param mask_features: the mask features output by the segmenter
        :param resume: whether the first frame is the start of the video
        :param return_indices: whether return the match indices
        :return: output dict, including masks, classes, embeds.
        """
        # mask feature projection
        mask_features_shape = mask_features.shape
        mask_features = self.mask_feature_proj(mask_features.flatten(0, 1)).reshape(*mask_features_shape)  # (b, t, c, h, w)

        frame_embeds = frame_embeds.permute(2, 3, 0, 1)  # t, q, b, c
        if frame_embeds_no_norm is not None:
            frame_embeds_no_norm = frame_embeds_no_norm.permute(2, 3, 0, 1)  # t, q, b, c
        n_frame, n_q, bs, _ = frame_embeds.size()
        outputs = []
        ret_indices = []

        all_frames_references = []

        for i in range(n_frame):
            ms_output = []
            single_frame_embeds = frame_embeds[i]  # q b c
            if frame_embeds_no_norm is not None:
                single_frame_embeds_no_norm = frame_embeds_no_norm[i]
            else:
                single_frame_embeds_no_norm = single_frame_embeds
            if frame_classes is None:
                single_frame_classes = None
            else:
                single_frame_classes = frame_classes[i]

            frame_key = single_frame_embeds_no_norm

            # the first frame of a video
            if i == 0 and resume is False:
                self._clear_memory()
                for j in range(self.num_layers):
                    if j == 0:
                        indices, noised_init = self.noiser(
                            single_frame_embeds,
                            single_frame_embeds,
                            cur_embeds_no_norm=single_frame_embeds_no_norm,
                            activate=False,
                            cur_classes=single_frame_classes,
                        )
                        ms_output.append(single_frame_embeds_no_norm[indices])
                        self.last_frame_embeds = single_frame_embeds[indices]
                        ret_indices.append(indices)
                        output = self.transformer_cross_attention_layers[j](
                            noised_init, self.ref_proj(frame_key),
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )

                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
                    else:
                        output = self.transformer_cross_attention_layers[j](
                            ms_output[-1], self.ref_proj(ms_output[-1]),
                            frame_key, single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )

                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
                self.last_reference = self.ref_proj(frame_key)
            else:
                reference = self.ref_proj(self.last_outputs[-1])
                self.last_reference = reference

                for j in range(self.num_layers):
                    if j == 0:
                        indices, noised_init = self.noiser(
                            self.last_frame_embeds,
                            single_frame_embeds,
                            cur_embeds_no_norm=single_frame_embeds_no_norm,
                            activate=self.training,
                            cur_classes=single_frame_classes,
                        )
                        ms_output.append(single_frame_embeds_no_norm[indices])
                        self.last_frame_embeds = single_frame_embeds[indices]
                        ret_indices.append(indices)
                        output = self.transformer_cross_attention_layers[j](
                            noised_init, reference, frame_key,
                            single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )

                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)
                    else:
                        output = self.transformer_cross_attention_layers[j](
                            ms_output[-1], reference, frame_key,
                            single_frame_embeds_no_norm,
                            memory_mask=None,
                            memory_key_padding_mask=None,
                            pos=None, query_pos=None
                        )

                        output = self.transformer_self_attention_layers[j](
                            output, tgt_mask=None,
                            tgt_key_padding_mask=None,
                            query_pos=None
                        )
                        # FFN
                        output = self.transformer_ffn_layers[j](
                            output
                        )
                        ms_output.append(output)

            all_frames_references.append(self.last_reference)

            ms_output = torch.stack(ms_output, dim=0)  # (1 + layers, q, b, c)
            self.last_outputs = ms_output
            outputs.append(ms_output[1:])
        outputs = torch.stack(outputs, dim=0)  # (t, l, q, b, c)

        all_frames_references = torch.stack(all_frames_references, dim=0)  # (t, q, b, c)

        mask_features_ = mask_features
        if not self.training:
            outputs = outputs[:, -1:]
            del mask_features
        outputs_class, outputs_masks = self.prediction(outputs, mask_features_, all_frames_references)
        out = {
           'pred_logits': outputs_class[-1].transpose(1, 2),  # (b, t, q, c)
           'pred_masks': outputs_masks[-1],  # (b, q, t, h, w)
           'aux_outputs': self._set_aux_loss(
               outputs_class, outputs_masks
           ),
           'pred_embds': outputs[:, -1].permute(2, 3, 0, 1),  # (b, c, t, q),
           'pred_references': all_frames_references.permute(2, 3, 0, 1),  # (b, c, t, q),
        }
        if return_indices:
            return out, ret_indices
        else:
            return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a.transpose(1, 2), "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]

    def prediction(self, outputs, mask_features, references):
        # outputs (t, l, q, b, c)
        # mask_features (b, t, c, h, w)
        # references (t, q, b, c)
        decoder_output = self.decoder_norm(outputs)
        decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)

        references = references.unsqueeze(1).repeat(1, decoder_output.size(0), 1, 1, 1).permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
        decoder_output_cls = torch.cat([references, decoder_output], dim=-1)
        outputs_class = self.class_embed(decoder_output_cls).transpose(2, 3)  # (l, b, q, t, cls+1)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features)
        return outputs_class, outputs_mask