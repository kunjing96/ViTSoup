# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/fundamentalvision/Deformable-DETR

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

from ..functions import MSDeformAttnFunction
from ..functions.ms_deform_attn_func import ms_deform_attn_core_pytorch


class SuperLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, scale=False):
        super().__init__(in_features, out_features, bias=bias)

        self.scale = scale
        # sampled
        self.sampled_in_features = None
        self.sampled_out_features = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def set_sample_config(self, sampled_in_features, sampled_out_features):
        self.sampled_in_features = sampled_in_features
        self.sampled_out_features = sampled_out_features
        self.sampled_weight = self.weight[:self.sampled_out_features, :self.sampled_in_features]
        if self.bias is not None:
            self.sampled_bias = self.bias[:self.sampled_out_features]
        if self.scale:
            self.sampled_scale = self.out_features / self.sampled_out_features

    def forward(self, x):
        return F.linear(x, self.sampled_weight, self.sampled_bias) * (self.sampled_scale if self.scale else 1)

    def params(self):
        params = 0
        params += self.sampled_weight.numel()
        if self.sampled_bias is not None:
            params += self.sampled_bias.numel()
        return params

    def flops(self, N):
        flops = 0
        flops += N * np.prod(self.sampled_weight.size())
        if self.sampled_bias is not None:
            flops += N * np.prod(self.sampled_bias.size())
        return flops


class q_super(nn.Linear):

    def __init__(self, super_in_dim, n_heads, n_levels, n_points, dim=1, bias=True, scale=False):
        super_out_dim = n_heads * n_levels * n_points * dim
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.dim = dim

        # input_dim and output_dim indicate the current sampled size

        self.scale = scale
        # sampled
        self.sampled_in_dim = None
        self.sampled_out_dim = None
        self.sampled_n_heads = None
        self.sampled_n_levels = None
        self.sampled_n_points = None
        self.sampled_weight = None
        self.sampled_bias = None

    def set_sample_config(self, sampled_in_dim, sampled_n_heads, sampled_n_levels, sampled_n_points):
        self.sampled_in_dim = sampled_in_dim
        self.sampled_out_dim = sampled_n_heads * sampled_n_levels * sampled_n_points * self.dim
        self.sampled_n_heads = sampled_n_heads
        self.sampled_n_levels = sampled_n_levels
        self.sampled_n_points = sampled_n_points
        self.sampled_weight = self.weight.reshape(self.n_heads, self.n_levels, self.n_points, self.dim, self.super_in_dim)[:self.sampled_n_heads, :self.sampled_n_levels, :self.sampled_n_points, :, :self.sampled_in_dim].reshape(-1, self.sampled_in_dim)
        if self.bias is not None:
            self.sampled_bias = self.bias.reshape(self.n_heads, self.n_levels, self.n_points, self.dim)[:self.sampled_n_heads, :self.sampled_n_levels, :self.sampled_n_points, :].reshape(-1)
        if self.scale:
            self.sampled_scale = self.super_out_dim / self.sampled_out_dim

    def forward(self, x):
        return F.linear(x, self.sampled_weight, self.sampled_bias) * (self.sampled_scale if self.scale else 1)

    def params(self):
        params = 0
        params += self.sampled_weight.numel()
        if self.sampled_bias is not None:
            params += self.sampled_bias.numel()
        return params

    def flops(self, N):
        flops = 0
        flops += N * np.prod(self.sampled_weight.size())
        if self.sampled_bias is not None:
            flops += N * np.prod(self.sampled_bias.size())
        return flops


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0


class SuperMSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4, scale=False):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 128

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.q_dim = n_heads * 32

        self.sampling_offsets = q_super(d_model, n_heads, n_levels, n_points, 2, scale=scale)
        self.attention_weights = q_super(d_model, n_heads, n_levels, n_points, 1, scale=scale)
        self.value_proj = SuperLinear(d_model, self.q_dim, scale=scale)
        self.output_proj = SuperLinear(self.q_dim, d_model, scale=scale)

        self._reset_parameters()

        self.scale = scale
        # sampled
        self.sampled_scale = None
        self.sampled_d_model = None
        self.sampled_q_dim = None
        self.sampled_n_heads = None
        self.sampled_n_levels = None
        self.sampled_n_points = None

    def _reset_parameters(self):
        nn.init.constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        nn.init.constant_(self.attention_weights.weight.data, 0.)
        nn.init.constant_(self.attention_weights.bias.data, 0.)
        nn.init.xavier_uniform_(self.value_proj.weight.data)
        nn.init.constant_(self.value_proj.bias.data, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight.data)
        nn.init.constant_(self.output_proj.bias.data, 0.)

    def set_sample_config(self, d_model=None, q_dim=None, transformer_nlevel=None, transformer_enc_num_points=None, transformer_nheads=None):
        self.sampled_d_model = d_model
        self.sampled_q_dim = q_dim
        self.sampled_n_heads = transformer_nheads
        self.sampled_n_levels = transformer_nlevel
        self.sampled_n_points = transformer_enc_num_points
        self.sampling_offsets.set_sample_config(self.sampled_d_model, self.sampled_n_heads, self.sampled_n_levels, self.sampled_n_points)
        self.attention_weights.set_sample_config(self.sampled_d_model, self.sampled_n_heads, self.sampled_n_levels, self.sampled_n_points)
        self.value_proj.set_sample_config(self.sampled_d_model, self.sampled_q_dim)
        self.output_proj.set_sample_config(self.sampled_q_dim, self.sampled_d_model)
        if self.scale:
            self.sampled_scale = self.d_model / self.sampled_q_dim

    def forward(self, query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))
        value = value.view(N, Len_in, self.sampled_n_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.sampled_n_heads, self.sampled_n_levels, self.sampled_n_points, 2)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.sampled_n_heads, self.sampled_n_levels * self.sampled_n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.sampled_n_heads, self.sampled_n_levels, self.sampled_n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.sampled_n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        try:
            output = MSDeformAttnFunction.apply(
                value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step) * (self.sampled_scale if self.scale else 1)
        except:
            # CPU
            output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights) * (self.sampled_scale if self.scale else 1)
        # # For FLOPs calculation only
        # output = ms_deform_attn_core_pytorch(value, input_spatial_shapes, sampling_locations, attention_weights)
        output = self.output_proj(output)
        return output

    def params(self):
        params = 0
        params += self.sampling_offsets.params()
        params += self.attention_weights.params()
        params += self.value_proj.params()
        params += self.output_proj.params()
        return params

    def flops(self, sequence_length):
        flops = 0
        flops += self.sampling_offsets.flops(sequence_length)
        flops += self.attention_weights.flops(sequence_length)
        flops += self.value_proj.flops(sequence_length)
        flops += sequence_length * self.sampled_n_points * self.sampled_q_dim
        flops += self.output_proj.flops(sequence_length)
        return flops
