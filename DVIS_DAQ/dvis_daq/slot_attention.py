import torch
from torch.nn import functional as F
from typing import Optional
from torch import nn, Tensor

class SlotAttention(nn.Module):
    """Slot attention module that iteratively performs cross-attention."""

    def __init__(
        self,
        in_features,
        num_iterations,
        num_slots,
        slot_size,
        mlp_hidden_size,
        eps=1e-6,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.eps = eps
        self.attn_scale = self.slot_size ** -0.5

        self.norm_inputs = nn.LayerNorm(self.in_features)

        # Linear maps for the attention module.
        self.project_q = nn.Sequential(
            nn.LayerNorm(self.slot_size),
            nn.Linear(self.slot_size, self.slot_size, bias=False),
        )
        self.project_k = nn.Linear(in_features, self.slot_size, bias=False)
        # self.project_v = nn.Linear(in_features, self.slot_size, bias=False)

    def forward(self, inputs, inputs_k, slots):
        """Forward function

        Args:
            inputs (torch.Tensor): [B, N, C], flattened per-pixel features.
            slots (torch.Tensor): [B, num_slots, C] slot inits.

        Returns:
            updated slots, same shape as 'slots'.
        """
        bs, num_inputs, inputs_size = inputs.shape
        num_slots = slots.shape[1]
        inputs_k = self.norm_inputs(inputs_k)
        k = self.project_k(inputs_k)
        v = inputs

        assert len(slots.shape) == 3

        q = self.project_q(slots)

        attn_logits = self.attn_scale * torch.einsum("bnc,bmc->bnm", k, q)
        attn = F.softmax(attn_logits, dim=-1)

        # Normalize along spatial dim and do weighted mean.
        attn = attn + self.eps
        attn = attn / torch.sum(attn, dim=1, keepdim=True)  # When no inputs contributed to a slot, then the slot averages all inputs

        updates = torch.einsum("bnm,bnc->bmc", attn, v)

        return updates.transpose(0, 1)


def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SlotCrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.slot_attn = SlotAttention(
            in_features=d_model, num_iterations=1, num_slots=0,
            slot_size=d_model, mlp_hidden_size=d_model, eps=1e-6,
        )

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None,
                     slot_query: Optional[Tensor] = None,
                     ):
        if slot_query is None:
            slot_query = tgt
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        inputs = tgt2
        inputs_k = tgt + self.dropout(tgt2)
        tgt3 = self.slot_attn(inputs.transpose(0, 1), inputs_k.transpose(0, 1), slot_query.transpose(0, 1))

        tgt = tgt + self.dropout(tgt3)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None,
                    slot_query: Optional[Tensor] = None,
                    ):
        if slot_query is None:
            slot_query = tgt
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]

        inputs = tgt2
        inputs_k = tgt + self.dropout(tgt2)
        tgt3 = self.slot_attn(inputs, inputs_k, slot_query)

        tgt = tgt + self.dropout(tgt3)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                slot_query: Optional[Tensor] = None,
                ):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos, slot_query)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos, slot_query)

