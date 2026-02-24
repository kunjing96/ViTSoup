import logging
import warnings
import fvcore.nn.weight_init as weight_init
from typing import Optional, List
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.functional import _mha_shape_check, _in_projection_packed, _in_projection, _scaled_dot_product_attention

from detectron2.config import configurable

from ..pixel_decoder.super import SuperConv2d, SuperPositionEmbeddingSine, _get_norm_super
from ..backbone.swin_super import SuperLinear
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY


class SuperEmbedding(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None, device=None, dtype=None, scale=False):
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx, max_norm=max_norm, norm_type=norm_type, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight, device=device, dtype=dtype)

        self.scale = scale
        # sampled
        self.sampled_num_embeddings = None
        self.sampled_embedding_dim = None
        self.sampled_list = None
        self.sampled_weight = None
        self.sampled_scale = None

    def set_sample_config(self, sampled_num_embeddings, sampled_embedding_dim):
        self.sampled_embedding_dim = sampled_embedding_dim
        if sampled_num_embeddings is not None:
            if isinstance(sampled_num_embeddings, list):
                self.sampled_list = sampled_num_embeddings
                self.sampled_num_embeddings = len(sampled_num_embeddings)
                self.sampled_weight = self.weight[self.sampled_list, :self.sampled_embedding_dim]
            elif isinstance(sampled_num_embeddings, int):
                self.sampled_num_embeddings = sampled_num_embeddings
                self.sampled_weight = self.weight[:self.sampled_num_embeddings, :self.sampled_embedding_dim]
            else:
                raise ValueError("sampled_num_embeddings must be a list or int number")
        else:
            self.sampled_weight = self.weight[:, :self.sampled_embedding_dim]
        if self.scale:
            self.sampled_scale = self.embedding_dim / self.sampled_embedding_dim

    def forward(self, x):
        return F.embedding(x, self.sampled_weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse) * (self.sampled_scale if self.scale else 1)

    def params(self):
        return self.sampled_weight.numel()

    def flops(self, sequence_length):
        return 0


def multi_head_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    embed_dim_to_check: int,
    hidden_dim_to_check: int,
    num_heads: int,
    in_proj_weight: Tensor,
    in_proj_bias: Optional[Tensor],
    bias_k: Optional[Tensor],
    bias_v: Optional[Tensor],
    add_zero_attn: bool,
    dropout_p: float,
    out_proj_weight: Tensor,
    out_proj_bias: Optional[Tensor],
    training: bool = True,
    key_padding_mask: Optional[Tensor] = None,
    need_weights: bool = True,
    attn_mask: Optional[Tensor] = None,
    use_separate_proj_weight: bool = False,
    q_proj_weight: Optional[Tensor] = None,
    k_proj_weight: Optional[Tensor] = None,
    v_proj_weight: Optional[Tensor] = None,
    static_k: Optional[Tensor] = None,
    static_v: Optional[Tensor] = None,
    average_attn_weights: bool = True,
):

    is_batched = _mha_shape_check(query, key, value, key_padding_mask, attn_mask, num_heads)

    # For unbatched input, we unsqueeze at the expected batch-dim to pretend that the input
    # is batched, run the computation and before returning squeeze the
    # batch dimension so that the output doesn't carry this temporary batch dimension.
    if not is_batched:
        # unsqueeze if the input is unbatched
        query = query.unsqueeze(1)
        key = key.unsqueeze(1)
        value = value.unsqueeze(1)
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(0)

    # set up shape vars
    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    assert embed_dim == embed_dim_to_check, \
        f"was expecting embedding dimension of {embed_dim_to_check}, but got {embed_dim}"
    if isinstance(hidden_dim_to_check, torch.Tensor):
        # embed_dim can be a tensor when JIT tracing
        head_dim = hidden_dim_to_check.div(num_heads, rounding_mode='trunc')
    else:
        head_dim = hidden_dim_to_check // num_heads
    assert head_dim * num_heads == hidden_dim_to_check, f"head_dim {head_dim} not divisible by num_heads {num_heads}"
    assert head_dim == 32, f"head_dim {head_dim} does not equal to 32"
    if use_separate_proj_weight:
        # allow MHA to have different embedding dimensions when separate projection weights are used
        assert key.shape[:2] == value.shape[:2], \
            f"key's sequence and batch dims {key.shape[:2]} do not match value's {value.shape[:2]}"
    else:
        assert key.shape == value.shape, f"key shape {key.shape} does not match value shape {value.shape}"

    #
    # compute in-projection
    #
    if not use_separate_proj_weight:
        q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)
    else:
        assert q_proj_weight is not None, "use_separate_proj_weight is True but q_proj_weight is None"
        assert k_proj_weight is not None, "use_separate_proj_weight is True but k_proj_weight is None"
        assert v_proj_weight is not None, "use_separate_proj_weight is True but v_proj_weight is None"
        if in_proj_bias is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = in_proj_bias.chunk(3)
        q, k, v = _in_projection(query, key, value, q_proj_weight, k_proj_weight, v_proj_weight, b_q, b_k, b_v)

    # prep attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, \
                f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
        # ensure attn_mask's dim is 3
        if attn_mask.dim() == 2:
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    # prep key padding mask
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    # add bias along batch dimension (currently second)
    if bias_k is not None and bias_v is not None:
        assert static_k is None, "bias cannot be added to static key."
        assert static_v is None, "bias cannot be added to static value."
        k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
        v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))
    else:
        assert bias_k is None
        assert bias_v is None

    #
    # reshape q, k, v for multihead attention and make em batch first
    #
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if static_k is None:
        k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_k.size(0) == bsz * num_heads, \
            f"expecting static_k.size(0) of {bsz * num_heads}, but got {static_k.size(0)}"
        assert static_k.size(2) == head_dim, \
            f"expecting static_k.size(2) of {head_dim}, but got {static_k.size(2)}"
        k = static_k
    if static_v is None:
        v = v.contiguous().view(v.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
    else:
        # TODO finish disentangling control flow so we don't do in-projections when statics are passed
        assert static_v.size(0) == bsz * num_heads, \
            f"expecting static_v.size(0) of {bsz * num_heads}, but got {static_v.size(0)}"
        assert static_v.size(2) == head_dim, \
            f"expecting static_v.size(2) of {head_dim}, but got {static_v.size(2)}"
        v = static_v

    # add zero attention along batch dimension (now first)
    if add_zero_attn:
        zero_attn_shape = (bsz * num_heads, 1, head_dim)
        k = torch.cat([k, torch.zeros(zero_attn_shape, dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros(zero_attn_shape, dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = F.pad(key_padding_mask, (0, 1))

    # update source sequence length after adjustments
    src_len = k.size(1)

    # merge key padding and attention masks
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len).   \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask, float("-inf"))

    # convert mask to float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # adjust dropout probability
    if not training:
        dropout_p = 0.0

    #
    # (deep breath) calculate attention and out projection
    #
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, hidden_dim_to_check)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # optionally average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        if average_attn_weights:
            attn_output_weights = attn_output_weights.sum(dim=1) / num_heads

        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
            attn_output_weights = attn_output_weights.squeeze(0)
        return attn_output, attn_output_weights
    else:
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None


class SuperMultiheadAttention(nn.MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False, device=None, dtype=None, scale=False):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(nn.MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = 32
        self.hidden_dim = self.head_dim * self.num_heads

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = nn.Parameter(torch.empty((self.hidden_dim, embed_dim), **factory_kwargs))
            self.k_proj_weight = nn.Parameter(torch.empty((self.hidden_dim, self.kdim), **factory_kwargs))
            self.v_proj_weight = nn.Parameter(torch.empty((self.hidden_dim, self.vdim), **factory_kwargs))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = nn.Parameter(torch.empty((3 * self.hidden_dim, embed_dim), **factory_kwargs))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * self.hidden_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = SuperLinear(self.hidden_dim, embed_dim, bias=bias, scale=scale)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, self.hidden_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, self.hidden_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

        self.scale = scale
        # sampled
        self.sampled_embed_dim = None
        self.sampled_kdim = None
        self.sampled_vdim = None
        self.sampled_hidden_dim = None
        self.sampled_num_heads = None
        self.sampled_dropout = None
        self.sampled_q_proj_weight = None
        self.sampled_k_proj_weight = None
        self.sampled_v_proj_weight = None
        self.sampled_in_proj_weight = None
        self.sampled_in_proj_bias = None
        self.sampled_bias_k = None
        self.sampled_bias_v = None
        self.sampled_scale = None

    def set_sample_config(self, sampled_embed_dim=None, sampled_num_heads=None, sampled_kdim=None, sampled_vdim=None, sampled_dropout=None):
        self.sampled_embed_dim = sampled_embed_dim
        self.sampled_kdim = sampled_kdim if sampled_kdim is not None else sampled_embed_dim
        self.sampled_vdim = sampled_vdim if sampled_vdim is not None else sampled_embed_dim
        self._qkv_same_embed_dim = self.sampled_kdim == sampled_embed_dim and self.sampled_vdim == sampled_embed_dim
        self.sampled_hidden_dim = sampled_num_heads * self.head_dim
        self.sampled_num_heads = sampled_num_heads
        self.sampled_dropout = sampled_dropout
        if self.q_proj_weight is not None:
            self.sampled_q_proj_weight = self.q_proj_weight[:self.sampled_hidden_dim, self.sampled_embed_dim]
        if self.k_proj_weight is not None:
            self.sampled_k_proj_weight = self.k_proj_weight[:self.sampled_hidden_dim, self.sampled_kdim]
        if self.v_proj_weight is not None:
            self.sampled_v_proj_weight = self.v_proj_weight[:self.sampled_hidden_dim, self.sampled_vdim]
        if self.in_proj_weight is not None:
            self.sampled_in_proj_weight = torch.cat([self.in_proj_weight[i:self.sampled_hidden_dim*3:3, :self.sampled_embed_dim] for i in range(3)], dim=0)
        if self.in_proj_bias is not None:
            self.sampled_in_proj_bias = torch.cat([self.in_proj_bias[i:self.sampled_hidden_dim*3:3] for i in range(3)], dim=0)
        if self.bias_k is not None:
            self.sampled_bias_k = self.bias_k[:, :, :self.sampled_hidden_dim]
        if self.bias_v is not None:
            self.sampled_bias_v = self.bias_v[:, :, :self.sampled_hidden_dim]
        self.out_proj.set_sample_config(self.sampled_hidden_dim, self.sampled_embed_dim)
        if self.scale:
            self.sampled_scale = self.hidden_dim / self.sampled_hidden_dim

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None, average_attn_weights=True):
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        query, key, value = query * (self.sampled_scale if self.scale else 1), key * (self.sampled_scale if self.scale else 1), value * (self.sampled_scale if self.scale else 1)
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.sampled_embed_dim, self.sampled_hidden_dim, self.sampled_num_heads,
                self.sampled_in_proj_weight, self.sampled_in_proj_bias,
                self.sampled_bias_k, self.sampled_bias_v, self.add_zero_attn,
                self.sampled_dropout, self.out_proj.sampled_weight, self.out_proj.sampled_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.sampled_q_proj_weight, k_proj_weight=self.sampled_k_proj_weight,
                v_proj_weight=self.sampled_v_proj_weight, average_attn_weights=average_attn_weights)
        else:
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.sampled_embed_dim, self.sampled_hidden_dim, self.sampled_num_heads,
                self.sampled_in_proj_weight, self.sampled_in_proj_bias,
                self.sampled_bias_k, self.sampled_bias_v, self.add_zero_attn,
                self.sampled_dropout, self.out_proj.sampled_weight, self.out_proj.sampled_bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, average_attn_weights=average_attn_weights)
        attn_output = attn_output * (self.sampled_scale if self.scale else 1)
        attn_output = attn_output * (self.out_proj.sampled_scale if self.out_proj.scale else 1)
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

    def params(self):
        params = 0
        if self.q_proj_weight is not None:
            params += self.sampled_q_proj_weight.numel()
        if self.k_proj_weight is not None:
            params += self.sampled_k_proj_weight.numel()
        if self.v_proj_weight is not None:
            params += self.sampled_v_proj_weight.numel()
        if self.in_proj_weight is not None:
            params += self.sampled_in_proj_weight.numel()
        if self.in_proj_bias is not None:
            params += self.sampled_in_proj_bias.numel()
        if self.bias_k is not None:
            params += self.sampled_bias_k.numel()
        if self.bias_v is not None:
            params += self.sampled_bias_v.numel()
        params += self.out_proj.params()
        return params

    def flops(self, lq, lk, lv):
        assert lk == lv, 'the length of key and value must be same'
        flops = 0
        if not self._qkv_same_embed_dim:
            flops += lq * np.prod(self.sampled_q_proj_weight.size()) + lk * np.prod(self.sampled_k_proj_weight.size()) + lv * np.prod(self.sampled_v_proj_weight.size())
        else:
            flops += (lq + lk + lv) * np.prod(self.sampled_in_proj_weight.size()) / 3
        flops += self.sampled_num_heads * lq * lk * (self.sampled_hidden_dim // self.sampled_num_heads)
        flops += self.sampled_num_heads * lq * lv * (self.sampled_hidden_dim // self.sampled_num_heads)
        flops += self.out_proj.flops(lq)
        return flops


class SuperSelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, normalize_before=False, norm='LN', scale=False):
        super().__init__()
        self.dim_pre_channel = 32
        self.d_model = d_model
        self.drop_rate = dropout
        self.self_attn = SuperMultiheadAttention(d_model, nhead, dropout=dropout, scale=scale)

        self.norm = _get_norm_super(norm, d_model)
        self.dropout = nn.Dropout(dropout)

        self.normalize_before = normalize_before

        self._reset_parameters()

        self.scale = scale
        # sampled
        self.is_identity_layer = None
        self.sampled_d_model = None
        self.sampled_nhead = None
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def set_sample_config(self, is_identity_layer, d_model=None, nhead=None):
        if is_identity_layer:
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.sampled_d_model = d_model
        self.sampled_nhead = nhead
        self.dropout.p = self.drop_rate * self.sampled_d_model / self.d_model
        self.self_attn.set_sample_config(sampled_embed_dim=self.sampled_d_model, sampled_num_heads=self.sampled_nhead, sampled_kdim=None, sampled_vdim=None, sampled_dropout=self.dropout.p)
        self.norm.set_sample_config(self.sampled_d_model)

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.is_identity_layer:
            return tgt
        tgt2 = self.maybe_layer_norm(self.norm, tgt, before=True)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.maybe_layer_norm(self.norm, tgt, after=True)
        return tgt

    def params(self):
        if self.is_identity_layer:
            return 0
        return self.self_attn.params() + self.norm.params()

    def flops(self, sequence_length):
        if self.is_identity_layer:
            return 0
        return self.self_attn.flops(sequence_length, sequence_length, sequence_length) + self.norm.flops(sequence_length)


class SuperCrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0, normalize_before=False, norm='LN', scale=False):
        super().__init__()
        self.dim_pre_channel = 32
        self.d_model = d_model
        self.drop_rate = dropout
        self.multihead_attn = SuperMultiheadAttention(d_model, nhead, dropout=dropout, scale=scale)

        self.norm = _get_norm_super(norm, d_model)
        self.dropout = nn.Dropout(dropout)

        self.normalize_before = normalize_before

        self._reset_parameters()

        self.scale = scale
        # sampled
        self.is_identity_layer = None
        self.sampled_d_model = None
        self.sampled_nhead = None

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def set_sample_config(self, is_identity_layer, d_model=None, nhead=None):
        if is_identity_layer:
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.sampled_d_model = d_model
        self.sampled_nhead = nhead
        self.dropout.p = self.drop_rate * self.sampled_d_model / self.d_model
        self.multihead_attn.set_sample_config(sampled_embed_dim=self.sampled_d_model, sampled_num_heads=self.sampled_nhead, sampled_kdim=None, sampled_vdim=None, sampled_dropout=self.dropout.p)
        self.norm.set_sample_config(self.sampled_d_model)

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.is_identity_layer:
            return tgt
        tgt2 = self.maybe_layer_norm(self.norm, tgt, before=True)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.maybe_layer_norm(self.norm, tgt, after=True)
        return tgt

    def params(self):
        if self.is_identity_layer:
            return 0
        return self.multihead_attn.params() + self.norm.params()

    def flops(self, lq, lk, lv):
        if self.is_identity_layer:
            return 0
        return self.multihead_attn.flops(lq, lk, lv) + self.norm.flops(lq)


class SuperFFNLayer(nn.Module):

    def __init__(self, d_model, mlp_ratio=8.0, dropout=0.0,
                 activation="relu", normalize_before=False, norm='LN', scale=False):
        super().__init__()
        self.d_model = d_model
        self.drop_rate = dropout
        self.dim_feedforward = int(mlp_ratio * d_model)
        # Implementation of Feedforward model
        self.linear1 = SuperLinear(d_model, self.dim_feedforward, scale=scale)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = SuperLinear(self.dim_feedforward, d_model, scale=scale)

        self.norm = _get_norm_super(norm, d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

        self.scale = scale
        # sampled
        self.is_identity_layer = None
        self.sampled_d_model = None
        self.sampled_dim_feedforward = None
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x

    def set_sample_config(self, is_identity_layer, d_model=None, mlp_ratio=None):
        if is_identity_layer:
            self.is_identity_layer = True
            return
        self.is_identity_layer = False
        self.sampled_d_model = d_model
        self.dropout.p = self.drop_rate * self.sampled_d_model / self.d_model
        self.sampled_dim_feedforward = int(mlp_ratio * d_model)
        self.linear1.set_sample_config(self.sampled_d_model, self.sampled_dim_feedforward)
        self.linear2.set_sample_config(self.sampled_dim_feedforward, self.sampled_d_model)
        self.norm.set_sample_config(self.sampled_d_model)

    def forward(self, tgt):
        if self.is_identity_layer:
            return tgt
        tgt2 = self.maybe_layer_norm(self.norm, tgt, before=True)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.maybe_layer_norm(self.norm, tgt, after=True)
        return tgt

    def params(self):
        if self.is_identity_layer:
            return 0
        params = 0
        params += self.linear1.params()
        params += self.linear2.params()
        params += self.norm.params()
        return params

    def flops(self, sequence_length):
        if self.is_identity_layer:
            return 0
        flops = 0
        flops += self.linear1.flops(sequence_length)
        flops += self.linear2.flops(sequence_length)
        flops += self.norm.flops(sequence_length)
        return flops


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class SuperMlp(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, activation, num_layers, scale=False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(SuperLinear(n, k, scale=scale) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.activation = _get_activation_fn(activation)

        self.scale = scale
        # sampled
        self.sampled_input_dim = None
        self.sampled_hidden_dim = None
        self.sampled_output_dim = None
        self.sampled_num_layers = None

    def set_sample_config(self, sampled_input_dim, sampled_hidden_dim, sampled_output_dim, sampled_num_layers=None):
        self.sampled_input_dim = sampled_input_dim
        self.sampled_hidden_dim = sampled_hidden_dim
        self.sampled_output_dim = sampled_output_dim
        self.sampled_num_layers = sampled_num_layers if sampled_num_layers else self.num_layers
        h = [self.sampled_hidden_dim] * (self.num_layers - 1)
        dim_list = list(zip([self.sampled_input_dim] + h, h + [self.sampled_output_dim]))
        for i, layer in enumerate(self.layers):
            if i < self.sampled_num_layers-1:
                layer.set_sample_config(dim_list[i][0], dim_list[i][1])
        self.layers[-1].set_sample_config(dim_list[-1][0], dim_list[-1][1])

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i < self.sampled_num_layers-1:
                x = self.activation(layer(x))
        return self.layers[-1](x)

    def params(self):
        params = 0
        for i, layer in enumerate(self.layers):
            if i < self.sampled_num_layers-1:
                params += layer.params()
        params += self.layers[-1].params()
        return params

    def flops(self, sequence_length):
        flops = 0
        for i, layer in enumerate(self.layers):
            if i < self.sampled_num_layers-1:
                flops += layer.flops(sequence_length)
        flops += self.layers[-1].flops(sequence_length)
        return flops


@TRANSFORMER_DECODER_REGISTRY.register()
class SuperMultiScaleMaskedTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        maskformer_in_features: List[str],
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        mlp_ratio: int,
        dropout,
        dec_layers: int,
        act_layer: str,
        pre_norm: bool,
        norm,
        mask_dim: int,
        enforce_input_project: bool,
        scale=False,
        use_checkpoint=False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.pre_norm = pre_norm
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = SuperPositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SuperSelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=dropout,
                    normalize_before=pre_norm,
                    norm=norm,
                    scale=scale
                )
            )

            self.transformer_cross_attention_layers.append(
                SuperCrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=dropout,
                    normalize_before=pre_norm,
                    norm=norm,
                    scale=scale
                )
            )
            
            self.transformer_ffn_layers.append(
                SuperFFNLayer(
                    d_model=hidden_dim,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    activation=act_layer,
                    normalize_before=pre_norm,
                    norm=norm,
                    scale=scale
                )
            )

        self.decoder_norm = _get_norm_super(norm, hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = SuperEmbedding(num_queries, hidden_dim, scale=scale)
        # learnable query p.e.
        self.query_embed = SuperEmbedding(num_queries, hidden_dim, scale=scale)

        # level embedding
        self.maskformer_in_features = sorted(maskformer_in_features, key=lambda x: int(x[3:]))
        self.num_feature_levels = len(maskformer_in_features)
        self.level_embed = SuperEmbedding(self.num_feature_levels, hidden_dim, scale=scale)
        self.input_proj = nn.ModuleDict()
        for k in maskformer_in_features:
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj[k] = SuperConv2d(in_channels, hidden_dim, kernel_size=1, scale=scale)
                weight_init.c2_xavier_fill(self.input_proj[k])
            else:
                self.input_proj[k] = None

        # output FFNs
        if self.mask_classification:
            self.class_embed = SuperLinear(hidden_dim, num_classes + 1, scale=scale)
        self.mask_embed = SuperMlp(hidden_dim, hidden_dim, mask_dim, act_layer, 3, scale=scale)

        self.scale = scale
        # sampled
        self.sampled_hidden_dim = None
        self.sampled_N_steps = None
        self.sampled_self_attention_nheads = None
        self.sampled_cross_attention_nheads = None
        self.sampled_ffn_mlp_ratio = None
        self.sampled_num_layers = None
        self.sampled_maskformer_in_features = None
        self.sampled_num_feature_levels = None
        self.sampled_embedding_indices = None
        self.sampled_in_channels = None
        self.sampled_mask_dim = None
        self.sampled_multi_scale_pre_layer = None

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["maskformer_in_features"] = cfg.MODEL.MASK_FORMER.DEC_IN_FEATURES
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.DEC_N_HEADS
        ret["mlp_ratio"] = cfg.MODEL.MASK_FORMER.DEC_MLP_RATIO
        ret["dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_DEPTHS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_DEPTHS - 1
        ret["act_layer"] = cfg.MODEL.MASK_FORMER.ACT_LAYER
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["norm"] = cfg.MODEL.MASK_FORMER.NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM

        ret["scale"] = cfg.MODEL.MASK_FORMER.SCALE
        ret["use_checkpoint"] = cfg.MODEL.MASK_FORMER.USE_CHECKPOINT

        return ret

    def set_sample_config(self, config: dict):
        self.sampled_hidden_dim = config.MASK_FORMER.HIDDEN_DIM
        self.sampled_N_steps = self.sampled_hidden_dim // 2
        self.pe_layer.set_sample_config(self.sampled_N_steps)

        self.sampled_self_attention_nheads = config.MASK_FORMER.DEC_SELF_N_HEADS
        self.sampled_cross_attention_nheads = config.MASK_FORMER.DEC_CROSS_N_HEADS
        self.sampled_ffn_mlp_ratio = config.MASK_FORMER.DEC_FFN_MLP_RATIO
        assert config.MASK_FORMER.DEC_DEPTHS >= 1
        self.sampled_num_layers = config.MASK_FORMER.DEC_DEPTHS - 1
        for i in range(self.num_layers):
            if i < self.sampled_num_layers:
                self.transformer_self_attention_layers[i].set_sample_config(False, self.sampled_hidden_dim, self.sampled_self_attention_nheads[i])
                self.transformer_cross_attention_layers[i].set_sample_config(False, self.sampled_hidden_dim, self.sampled_cross_attention_nheads[i])
                self.transformer_ffn_layers[i].set_sample_config(False, self.sampled_hidden_dim, self.sampled_ffn_mlp_ratio[i])
            else:
                self.transformer_self_attention_layers[i].set_sample_config(True)
                self.transformer_cross_attention_layers[i].set_sample_config(True)
                self.transformer_ffn_layers[i].set_sample_config(True)

        self.decoder_norm.set_sample_config(self.sampled_hidden_dim)

        self.query_feat.set_sample_config(None, self.sampled_hidden_dim)
        self.query_embed.set_sample_config(None, self.sampled_hidden_dim)

        self.sampled_maskformer_in_features = sorted(config.MASK_FORMER.DEC_IN_FEATURES, key=lambda x: int(x[3:]))
        self.sampled_num_feature_levels = len(self.sampled_maskformer_in_features)
        self.sampled_embedding_indices = [self.maskformer_in_features.index(k) for k in self.sampled_maskformer_in_features]
        self.level_embed.set_sample_config(self.sampled_embedding_indices, self.sampled_hidden_dim)
        self.sampled_in_channels = config.SEM_SEG_HEAD.CONVS_DIM
        for k in self.sampled_maskformer_in_features:
            if self.input_proj[k] is not None:
                self.input_proj[k].set_sample_config(self.sampled_in_channels, self.sampled_hidden_dim)

        if self.mask_classification:
            self.class_embed.set_sample_config(self.sampled_hidden_dim, self.num_classes + 1)
        self.sampled_mask_dim = config.SEM_SEG_HEAD.MASK_DIM
        self.mask_embed.set_sample_config(self.sampled_hidden_dim, self.sampled_hidden_dim, self.sampled_mask_dim)
        self.sampled_multi_scale_pre_layer = config.MASK_FORMER.MULTI_SCALE_PER_LAYER
        assert len(self.sampled_multi_scale_pre_layer) == self.sampled_num_layers

    def forward(self, x, mask_features, mask = None):
        # x is a dict of multi-scale feature
        assert len(x) == self.sampled_num_feature_levels
        src = {}
        pos = {}
        size_list = {}

        # disable mask, it does not affect performance
        del mask

        for i, k in enumerate(self.sampled_maskformer_in_features):
            size_list[k] = x[k].shape[-2:]
            pos[k] = self.pe_layer(x[k], None).flatten(2)
            src[k] = self.input_proj[k](x[k]).flatten(2) if self.input_proj[k] is not None else x[k].flatten(2) + self.level_embed.sampled_weight[i][None, :, None]

            # flatten NxCxHxW to HWxNxC
            pos[k] = pos[k].permute(2, 0, 1)
            src[k] = src[k].permute(2, 0, 1)

        _, bs, _ = src[self.sampled_maskformer_in_features[-1]].shape

        # QxNxC
        query_embed = self.query_embed.sampled_weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.sampled_weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[self.sampled_multi_scale_pre_layer[0]], num_heads=self.sampled_cross_attention_nheads[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.sampled_num_layers):
            k = self.sampled_multi_scale_pre_layer[i]
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            if self.use_checkpoint:
                output = checkpoint.checkpoint(self.transformer_cross_attention_layers[i], output, src[k], attn_mask, None, pos[k], query_embed)
            else:
                output = self.transformer_cross_attention_layers[i](
                    output, src[k],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[k], query_pos=query_embed
                )

            if self.use_checkpoint:
                output = checkpoint.checkpoint(self.transformer_self_attention_layers[i], output, None, None, query_embed)
            else:
                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )
            
            # FFN
            if self.use_checkpoint:
                output = checkpoint.checkpoint(self.transformer_ffn_layers[i], output)
            else:
                output = self.transformer_ffn_layers[i](
                    output
                )

            if i + 1 < self.sampled_num_layers:
                outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[self.sampled_multi_scale_pre_layer[i+1]], num_heads=self.sampled_cross_attention_nheads[i+1])
            else:
                outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        assert len(predictions_class) == self.sampled_num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            )
        }
        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size=None, num_heads=None):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        if attn_mask_target_size is not None and num_heads is not None:
            # NOTE: prediction is of higher-resolution
            # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
            attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
            # must use bool type
            # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
            attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
            attn_mask = attn_mask.detach()
        else:
            attn_mask = None

        return outputs_class, outputs_mask, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]

    def params(self):
        params = 0
        params += self.pe_layer.params()
        for i in range(self.num_layers):
            params += self.transformer_cross_attention_layers[i].params()
            params += self.transformer_self_attention_layers[i].params()
            params += self.transformer_ffn_layers[i].params()
        params += self.decoder_norm.params()
        params += self.query_feat.params()
        params += self.query_embed.params()
        params += self.level_embed.params()
        for k in self.sampled_maskformer_in_features:
            if self.input_proj[k] is not None:
                params += self.input_proj[k].params()
        if self.mask_classification:
            params += self.class_embed.params()
        params += self.mask_embed.params()
        return params

    def flops(self, H, W, shapes):
        flops = 0
        for k in self.sampled_maskformer_in_features:
            flops += self.pe_layer.flops((H // shapes[k].stride) * (W // shapes[k].stride))
            if self.input_proj[k] is not None:
                flops += self.input_proj[k].flops((H // shapes[k].stride), (W // shapes[k].stride))
            flops += self.level_embed.flops(0)
        flops += self.query_feat.flops(0)
        flops += self.query_embed.flops(0)

        flops += self.decoder_norm.flops(self.num_queries)
        if self.mask_classification:
            flops += self.class_embed.flops(self.num_queries)
        flops += self.mask_embed.flops(self.num_queries)
        flops += self.num_queries * self.sampled_hidden_dim * (H // shapes['res2'].stride) * (W // shapes['res2'].stride)
        for i in range(self.sampled_num_layers):
            k = self.sampled_multi_scale_pre_layer[i]
            flops += self.transformer_cross_attention_layers[i].flops(self.num_queries, (H // shapes[k].stride) * (W // shapes[k].stride), (H // shapes[k].stride) * (W // shapes[k].stride))
            flops += self.transformer_self_attention_layers[i].flops(self.num_queries)
            flops += self.transformer_ffn_layers[i].flops(self.num_queries)
            flops += self.decoder_norm.flops(self.num_queries)
            if self.mask_classification:
                flops += self.class_embed.flops(self.num_queries)
            flops += self.mask_embed.flops(self.num_queries)
            flops += self.num_queries * self.sampled_hidden_dim * (H // shapes['res2'].stride) * (W // shapes['res2'].stride)
        return flops
