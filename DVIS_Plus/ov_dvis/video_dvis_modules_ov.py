import torch
from torch import nn
from mask2former_video.modeling.transformer_decoder.video_mask2former_transformer_decoder import SelfAttentionLayer,\
    CrossAttentionLayer, FFNLayer, MLP
from dvis_Plus.noiser import Noiser
from dvis_Plus.tracker import ReferringCrossAttentionLayer
import torch.nn.functional as F

def get_classification_logits(x, text_classifier, logit_scale, num_templates=None):
    # x in shape of [B, *, C]
    # text_classifier in shape of [num_classes, C]
    # logit_scale is a learnable scalar https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model.py#L201
    # return: [B, *, num_classes]
    x = F.normalize(x, dim=-1)
    logit_scale = torch.clamp(logit_scale.exp(), max=100)
    pred_logits = logit_scale * x @ text_classifier.T # B, *, N + 1
    # max ensembel as in OpenSeg/ODISE
    final_pred_logits = []
    cur_idx = 0
    for num_t in num_templates[:-1]:
        final_pred_logits.append(pred_logits[..., cur_idx: cur_idx + num_t].max(-1).values)
        cur_idx += num_t
    # final_pred_logits.append(pred_logits[:, :, -1]) # the last classifier is for void
    final_pred_logits.append(pred_logits[..., -num_templates[-1]:].max(-1).values)
    final_pred_logits = torch.stack(final_pred_logits, dim=-1)
    return final_pred_logits

class ReferringTracker_noiser_OV(torch.nn.Module):
    def __init__(
        self,
        hidden_channel=256,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
        noise_mode='hard',
        # frozen fc-clip head
        mask_pooling=None,
        mask_pooling_proj=None,
        class_embed=None,
        logit_scale=None,
        mask_embed=None,
        decoder_norm=None,
    ):
        super(ReferringTracker_noiser_OV, self).__init__()

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

        # for cl learning
        self.ref_proj = MLP(hidden_channel, hidden_channel, hidden_channel, 3)
        # for reference and query merge
        self.merge = nn.Linear(hidden_channel * 2, hidden_channel)

        # record previous frame information
        self.last_outputs = None
        self.last_frame_embeds = None
        self.last_reference = None

        self.noiser = Noiser(noise_ratio=0.5, mode=noise_mode)

        # FC-CLIP
        self.mask_pooling = mask_pooling
        self._mask_pooling_proj = mask_pooling_proj
        self.class_embed = class_embed
        self.logit_scale = logit_scale
        self.mask_embed = mask_embed
        self.decoder_norm = decoder_norm

    def _clear_memory(self):
        del self.last_outputs
        self.last_outputs = None
        self.last_reference = None
        return

    def forward(self, frame_embeds, mask_features, resume=False,
                return_indices=False, frame_classes=None,
                frame_embeds_no_norm=None, cur_feature=None,
                text_classifier=None, num_templates=None,
                ):
        """
        :param frame_embeds: the instance queries output by the segmenter
        :param mask_features: the mask features output by the segmenter
        :param resume: whether the first frame is the start of the video
        :param return_indices: whether return the match indices
        :return: output dict, including masks, classes, embeds.
        """

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
        outputs_class, outputs_masks = self.prediction(outputs, mask_features_, text_classifier,
                                                       num_templates, all_frames_references)
        out = {
           'pred_logits': outputs_class[-1].transpose(1, 2),  # (b, t, q, c)
           'pred_masks': outputs_masks[-1],  # (b, q, t, h, w)
           'aux_outputs': self._set_aux_loss(
               outputs_class, outputs_masks
           ),
           'pred_embds': outputs[:, -1].permute(2, 3, 0, 1),  # (b, c, t, q)
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

    def prediction(self, outputs, mask_features, text_classifier, num_templates, references):
        # outputs (t, l, q, b, c)
        # mask_features (b, t, c, h, w)
        decoder_output = self.decoder_norm(outputs)
        decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features)

        references = references.unsqueeze(1).repeat(1, decoder_output.size(0), 1, 1, 1).permute(1, 3, 0, 2, 4)
        decoder_output_cls = torch.cat([references, decoder_output], dim=-1)
        decoder_output_cls = self.merge(decoder_output_cls)

        # fc-clip class head forward
        # mean pooling
        b, t, c, _, _ = mask_features.shape
        l, b, q, t, _, _ = outputs_mask.shape
        mask_features = mask_features.unsqueeze(0).repeat(l, 1, 1, 1, 1, 1).flatten(0, 2)  # lbt, c, h, w
        outputs_mask_ = outputs_mask.permute(0, 1, 3, 2, 4, 5).flatten(0, 2)  # (lbt, q, h, w)
        maskpool_embeddings = self.mask_pooling(x=mask_features, mask=outputs_mask_)  # [lbt, q, c]
        maskpool_embeddings = maskpool_embeddings.reshape(l, b, t, *maskpool_embeddings.shape[-2:])  # (l b t q c)
        maskpool_embeddings = self._mask_pooling_proj(maskpool_embeddings)
        class_embed = self.class_embed(maskpool_embeddings + decoder_output_cls)
        outputs_class = get_classification_logits(class_embed, text_classifier, self.logit_scale, num_templates)
        outputs_class = outputs_class.transpose(2, 3)  # (l, b, q, t, cls+1)

        return outputs_class, outputs_mask

class TemporalRefiner_OV(torch.nn.Module):
    def __init__(
        self,
        hidden_channel=256,
        feedforward_channel=2048,
        num_head=8,
        decoder_layer_num=6,
        mask_dim=256,
        class_num=25,
        windows=5,
        # resume segmenter prediction head
        mask_pooling=None,
        mask_pooling_proj=None,
        class_embed=None,
        logit_scale=None,
        mask_embed=None,
        decoder_norm=None,
    ):
        super(TemporalRefiner_OV, self).__init__()

        self.windows = windows

        # init transformer layers
        self.num_heads = num_head
        self.num_layers = decoder_layer_num
        self.transformer_obj_self_attention_layers = nn.ModuleList()
        self.transformer_time_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        self.conv_short_aggregate_layers = nn.ModuleList()
        self.conv_norms = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_time_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.conv_short_aggregate_layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_channel, hidden_channel,
                              kernel_size=5, stride=1,
                              padding='same', padding_mode='replicate'),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(hidden_channel, hidden_channel,
                              kernel_size=3, stride=1,
                              padding='same', padding_mode='replicate'),
                )
            )

            self.conv_norms.append(nn.LayerNorm(hidden_channel))

            self.transformer_obj_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_channel,
                    nhead=num_head,
                    dropout=0.0,
                    normalize_before=False,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
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

        self.decoder_norm = nn.LayerNorm(hidden_channel)

        # FC-CLIP
        self.mask_pooling = mask_pooling
        self._mask_pooling_proj = mask_pooling_proj
        self.class_embed = class_embed
        self.logit_scale = logit_scale
        self.mask_embed = mask_embed
        self.decoder_norm = decoder_norm

        self.activation_proj = nn.Linear(hidden_channel, 1)

    def forward(self, instance_embeds, frame_embeds, mask_features,
                text_classifier=None, num_templates=None,):
        """
        :param instance_embeds: the aligned instance queries output by the tracker, shape is (b, c, t, q)
        :param frame_embeds: the instance queries processed by the tracker.frame_forward function, shape is (b, c, t, q)
        :param mask_features: the mask features output by the segmenter, shape is (b, t, c, h, w)
        :return: output dict, including masks, classes, embeds.
        """
        n_batch, n_channel, n_frames, n_instance = instance_embeds.size()

        outputs = []
        output = instance_embeds
        frame_embeds = frame_embeds.permute(3, 0, 2, 1).flatten(1, 2)

        for i in range(self.num_layers):
            output = output.permute(2, 0, 3, 1)  # (t, b, q, c)
            output = output.flatten(1, 2)  # (t, bq, c)

            # do long temporal attention
            output = self.transformer_time_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )

            # do short temporal conv
            output = output.permute(1, 2, 0)  # (bq, c, t)
            output = self.conv_norms[i](
                (self.conv_short_aggregate_layers[i](output) + output).transpose(1, 2)
            ).transpose(1, 2)
            output = output.reshape(
                n_batch, n_instance, n_channel, n_frames
            ).permute(1, 0, 3, 2).flatten(1, 2)  # (q, bt, c)

            # do objects self attention
            output = self.transformer_obj_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )

            # do cross attention
            output = self.transformer_cross_attention_layers[i](
                output, frame_embeds,
                memory_mask=None,
                memory_key_padding_mask=None,
                pos=None, query_pos=None
            )

            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )

            output = output.reshape(n_instance, n_batch, n_frames, n_channel).permute(1, 3, 2, 0)  # (b, c, t, q)
            outputs.append(output)

        outputs = torch.stack(outputs, dim=0).permute(3, 0, 4, 1, 2)  # (l, b, c, t, q) -> (t, l, q, b, c)
        outputs_class, outputs_masks = self.prediction(outputs, mask_features, text_classifier, num_templates)
        outputs = self.decoder_norm(outputs)
        out = {
           'pred_logits': outputs_class[-1].transpose(1, 2),  # (b, t, q, c)
           'pred_masks': outputs_masks[-1],  # (b, q, t, h, w)
           'aux_outputs': self._set_aux_loss(
               outputs_class, outputs_masks
           ),
           'pred_embds': outputs[:, -1].permute(2, 3, 0, 1)  # (b, c, t, q)
        }
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a.transpose(1, 2), "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
                ]

    def windows_prediction(self, outputs, mask_features, text_classifier, num_templates, windows=5):
        """
        for windows prediction, because mask features consumed too much GPU memory
        """
        iters = outputs.size(0) // windows
        if outputs.size(0) % windows != 0:
            iters += 1
        outputs_classes = []
        outputs_masks = []
        maskpool_embeddings = []
        pixel_nums = []
        for i in range(iters):
            start_idx = i * windows
            end_idx = (i + 1) * windows
            clip_outputs = outputs[start_idx:end_idx]
            decoder_output = self.decoder_norm(clip_outputs)
            decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
            mask_embed = self.mask_embed(decoder_output)
            mask_features_clip = mask_features[:, start_idx:end_idx].to(mask_embed.device)
            outputs_mask = torch.einsum(
                "lbtqc,btchw->lbqthw",
                mask_embed,
                mask_features_clip
            )
            maskpool_embedding, pixel_num = self._get_maskpool_embeddings(mask_features_clip, outputs_mask, return_num=True)
            maskpool_embeddings.append(maskpool_embedding)  # (l b 1 q c)
            pixel_nums.append(pixel_num) # (l b 1 q)
            outputs_classes.append(decoder_output)
            outputs_masks.append(outputs_mask.cpu().to(torch.float32))
        outputs_classes = torch.cat(outputs_classes, dim=2)
        T = outputs.size(0)
        outputs_classes = self._temoral_weighting(outputs_classes)  # (l, b, 1, q, c)

        maskpool_embeddings = torch.cat(maskpool_embeddings, dim=2)
        pixel_nums = torch.cat(pixel_nums, dim=2)
        pixel_nums = pixel_nums / torch.sum(pixel_nums, dim=2, keepdim=True)
        maskpool_embeddings = maskpool_embeddings * pixel_nums.unsqueeze(-1)
        maskpool_embeddings = torch.sum(maskpool_embeddings, dim=2, keepdim=True)
        maskpool_embeddings = self._mask_pooling_proj(maskpool_embeddings)  # (l b 1 q c)

        class_embed = self.class_embed(maskpool_embeddings + outputs_classes)
        outputs_classes = get_classification_logits(class_embed, text_classifier, self.logit_scale, num_templates)
        outputs_classes = outputs_classes.repeat(1, 1, T, 1, 1).transpose(2, 3)  # (l, b, q, t, cls+1)
        return outputs_classes.cpu().to(torch.float32), torch.cat(outputs_masks, dim=3)

    def _get_maskpool_embeddings(self, mask_features, outputs_mask, return_num=False):
        b, t, c, _, _ = mask_features.shape
        l, b, q, t, _, _ = outputs_mask.shape

        mask_features = mask_features.unsqueeze(0).repeat(l,
                        1, 1, 1, 1, 1).permute(0, 1, 3, 2, 4, 5).flatten(0, 1).flatten(2, 3)  # lb, c, th, w
        outputs_mask_ = outputs_mask.flatten(0, 1).flatten(2, 3)  # (lb, q, th, w)
        if return_num:
            maskpool_embeddings, pixel_num = self.mask_pooling(x=mask_features, mask=outputs_mask_, return_num=True)
            # maskpool_embeddings [lb, q, c], pixel_num [lb, q]
            pixel_num = pixel_num.reshape(l, b, q)
            pixel_num = pixel_num.unsqueeze(2)  # (l b 1 q)
        else:
            maskpool_embeddings = self.mask_pooling(x=mask_features, mask=outputs_mask_)  # [lb, q, c]
        maskpool_embeddings = maskpool_embeddings.reshape(l, b, *maskpool_embeddings.shape[-2:])  # (l b q c)
        maskpool_embeddings = maskpool_embeddings.unsqueeze(2)  # (l b 1 q c)
        if return_num:
            return maskpool_embeddings, pixel_num
        else:
            return maskpool_embeddings

    def _temoral_weighting(self, decoder_output):
        # compute the weighted average of the decoder_output
        activation = self.activation_proj(decoder_output).softmax(dim=2)  # (l, b, t, q, 1)
        class_output = (decoder_output * activation).sum(dim=2, keepdim=True)  # (l, b, 1, q, c)
        return class_output

    def pred_class(self, decoder_output, mask_features, outputs_mask, text_classifier, num_templates):
        """
        fuse the objects queries of all frames and predict an overall score based on the fused objects queries
        :param decoder_output: instance queries, shape is (l, b, t, q, c)
        """
        T = decoder_output.size(2)

        # compute the weighted average of the decoder_output
        class_output = self._temoral_weighting(decoder_output)

        # fc-clip class head forward
        # mean pooling
        maskpool_embeddings = self._get_maskpool_embeddings(mask_features, outputs_mask)
        maskpool_embeddings = self._mask_pooling_proj(maskpool_embeddings)  # (l b 1 q c)

        class_embed = self.class_embed(maskpool_embeddings + class_output)
        outputs_class = get_classification_logits(class_embed, text_classifier, self.logit_scale, num_templates)
        outputs_class = outputs_class.repeat(1, 1, T, 1, 1).transpose(2, 3)  # (l, b, q, t, cls+1)
        return outputs_class

    def prediction(self, outputs, mask_features, text_classifier, num_templates):
        """
        :param outputs: instance queries, shape is (t, l, q, b, c)
        :param mask_features: mask features, shape is (b, t, c, h, w)
        :return: pred class and pred masks
        """
        if self.training:
            decoder_output = self.decoder_norm(outputs)
            decoder_output = decoder_output.permute(1, 3, 0, 2, 4)  # (l, b, t, q, c)
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("lbtqc,btchw->lbqthw", mask_embed, mask_features)
            outputs_class = self.pred_class(decoder_output, mask_features,
                                            outputs_mask, text_classifier, num_templates)
        else:
            outputs = outputs[:, -1:]
            outputs_class, outputs_mask = self.windows_prediction(outputs, mask_features, text_classifier,
                                                                  num_templates, windows=self.windows)
        return outputs_class, outputs_mask

