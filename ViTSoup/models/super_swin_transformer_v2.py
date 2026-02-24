# --------------------------------------------------------
# Swin Transformer V2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np


class SuperLayerNorm(nn.LayerNorm):
    def __init__(self, embed_dim):
        super().__init__(embed_dim)

        # sampled
        self.sampled_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None

    def set_sample_config(self, sampled_embed_dim):
        self.sampled_embed_dim = sampled_embed_dim
        self.sampled_weight = self.weight[:self.sampled_embed_dim]
        self.sampled_bias = self.bias[:self.sampled_embed_dim]

    def forward(self, x):
        return F.layer_norm(x, (self.sampled_embed_dim,), weight=self.sampled_weight, bias=self.sampled_bias, eps=self.eps)

    def params(self):
        params = 0
        params += self.sampled_weight.numel()
        if self.sampled_bias is not None:
            params += self.sampled_bias.numel()
        return params

    def flops(self, N):
        flops = N * self.sampled_embed_dim
        return flops


class SuperMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., scale=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features or in_features
        self.hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(self.in_features, self.hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(self.hidden_features, self.out_features)
        self.drop = nn.Dropout(drop)

        self.scale = scale
        # sampled
        self.sampled_in_features = None
        self.sampled_hidden_features = None
        self.sampled_out_features = None
        self.sampled_drop = None
        self.sampled_fc1_weight = None
        self.sampled_fc1_bias = None
        self.sampled_fc1_scale = None
        self.sampled_fc2_weight = None
        self.sampled_fc2_bias = None
        self.sampled_fc2_scale = None

    def set_sample_config(self, sampled_in_features, sampled_hidden_features, sampled_out_features, sampled_drop):
        self.sampled_in_features = sampled_in_features
        self.sampled_hidden_features = sampled_hidden_features or sampled_in_features
        self.sampled_out_features = sampled_out_features or sampled_in_features
        self.sampled_drop = sampled_drop

        self.sampled_fc1_weight = self.fc1.weight[:self.sampled_hidden_features, :self.sampled_in_features]
        if self.fc1.bias is not None:
            self.sampled_fc1_bias = self.fc1.bias[:self.sampled_hidden_features]
        if self.scale:
            self.sampled_fc1_scale = self.hidden_features / self.sampled_hidden_features
        self.sampled_fc2_weight = self.fc2.weight[:self.sampled_out_features, :self.sampled_hidden_features]
        if self.fc2.bias is not None:
            self.sampled_fc2_bias = self.fc2.bias[:self.sampled_out_features]
        if self.scale:
            self.sampled_fc2_scale = self.out_features / self.sampled_out_features
        self.drop.p = self.sampled_drop

    def forward(self, x):
        x = F.linear(x, self.sampled_fc1_weight, self.sampled_fc1_bias) * (self.sampled_fc1_scale if self.scale else 1)
        x = self.act(x)
        x = self.drop(x)
        x = F.linear(x, self.sampled_fc2_weight, self.sampled_fc2_bias) * (self.sampled_fc2_scale if self.scale else 1)
        x = self.drop(x)
        return x

    def params(self):
        params = 0
        params += self.sampled_fc1_weight.numel()
        if self.sampled_fc1_bias is not None:
            params += self.sampled_fc1_bias.numel()
        params += self.sampled_fc2_weight.numel()
        if self.sampled_fc2_bias is not None:
            params += self.sampled_fc2_bias.numel()
        return params

    def flops(self, N):
        flops = 0
        flops += N * self.sampled_in_features * self.sampled_hidden_features
        flops += N * self.sampled_hidden_features * self.sampled_out_features
        return flops


class SuperPatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, scale=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

        self.scale = scale
        # sampled
        self.sampled_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def set_sample_config(self, sampled_embed_dim):
        self.sampled_embed_dim = sampled_embed_dim
        self.sampled_weight = self.proj.weight[:sampled_embed_dim, ...]
        if self.proj.bias is not None:
            self.sampled_bias = self.proj.bias[:self.sampled_embed_dim, ...]
        if self.scale:
            self.sampled_scale = self.embed_dim / self.sampled_embed_dim
        if self.norm is not None:
            self.norm.set_sample_config(self.sampled_embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = F.conv2d(x, self.sampled_weight, self.sampled_bias, stride=self.patch_size, padding=self.proj.padding, dilation=self.proj.dilation).flatten(2).transpose(1, 2) * (self.sampled_scale if self.scale else 1)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def params(self):
        params = 0
        params += self.sampled_weight.numel()
        if self.sampled_bias is not None:
            params += self.sampled_bias.numel()
        if self.norm is not None:
            params += self.norm.params()
        return params

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = 0
        flops += Ho * Wo * np.prod(self.sampled_weight.size())
        if self.sampled_bias is not None:
            flops += self.sampled_bias.size(0)
        if self.norm is not None:
            flops += self.norm.flops(Ho * Wo)
        return flops


class SuperPatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=SuperLayerNorm, scale=False):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(2 * dim)

        self.scale = scale
        # sampled
        self.sampled_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def set_sample_config(self, sampled_dim):
        self.sampled_dim = sampled_dim
        self.sampled_weight = torch.cat([self.reduction.weight[:2 * self.sampled_dim,i:4 * self.sampled_dim:4] for i in range(4)], dim=1)
        if self.reduction.bias is not None:
            self.sampled_bias = self.reduction.bias[:2 * self.sampled_dim]
        if self.scale:
            self.sampled_scale = self.dim / self.sampled_dim
        self.norm.sampled_embed_dim = 4 * self.sampled_dim
        self.norm.sampled_weight = torch.cat([self.norm.weight[i:4 * self.sampled_dim:4] for i in range(4)], dim=0)
        self.norm.sampled_bias = torch.cat([self.norm.bias[i:4 * self.sampled_dim:4] for i in range(4)], dim=0)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = F.linear(x, self.sampled_weight, self.sampled_bias) * (self.sampled_scale if self.scale else 1)
        x = self.norm(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def params(self):
        params = 0
        params += self.norm.params()
        params += self.sampled_weight.numel()
        if self.sampled_bias is not None:
            params += self.sampled_bias.numel()
        return params

    def flops(self):
        H, W = self.input_resolution
        flops = 0
        flops += self.norm.flops(H * W)
        flops += (H // 2) * (W // 2) * 4 * self.sampled_dim * 2 * self.sampled_dim
        return flops


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class SuperWindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
        pretrained_window_size (tuple[int]): The height and width of the window in pre-training.
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True,
                 attn_drop=0., proj_drop=0., pretrained_window_size=[0, 0], scale=False):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.pretrained_window_size = pretrained_window_size
        self.num_heads = num_heads

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.scale = scale
        # sampled
        self.sampled_proj_drop = None
        self.sampled_attn_drop = None
        self.sampled_dim = None
        self.sampled_qk_dim = None
        self.sampled_num_heads = None
        self.sampled_window_size = None
        self.sampled_logit_scale = None
        self.sampled_cpb_mlp_2_weight = None
        self.sampled_cpb_mlp_2_bias = None
        self.sampled_relative_coords_table = None
        self.sampled_relative_position_index = None
        self.sampled_qkv_weight = None
        self.sampled_q_bias = None
        self.sampled_v_bias = None
        self.sampled_proj_weight = None
        self.sampled_proj_bias = None
        self.sampled_scale = None
        self.sampled_cpb_scale = None

    def set_sample_config(self, sampled_dim=None, sampled_qk_dim=None, sampled_num_heads=None, sampled_proj_drop=None, sampled_attn_drop=None, sampled_window_size=None):
        self.sampled_proj_drop = sampled_proj_drop
        self.sampled_attn_drop = sampled_attn_drop
        self.sampled_dim = sampled_dim
        self.sampled_qk_dim = sampled_qk_dim
        self.sampled_num_heads = sampled_num_heads
        self.sampled_window_size = sampled_window_size

        self.sampled_logit_scale = self.logit_scale[:self.sampled_num_heads , :, :]

        self.sampled_cpb_mlp_2_weight = self.cpb_mlp[2].weight[:self.sampled_num_heads, :]
        if self.cpb_mlp[2].bias is not None:
            self.sampled_cpb_mlp_2_bias = self.cpb_mlp[2].bias[:self.sampled_num_heads]

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.sampled_window_size[0] - 1), self.sampled_window_size[0], dtype=torch.float32, device=next(self.parameters()).device)
        relative_coords_w = torch.arange(-(self.sampled_window_size[1] - 1), self.sampled_window_size[1], dtype=torch.float32, device=next(self.parameters()).device)
        self.sampled_relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if self.pretrained_window_size[0] > 0:
            self.sampled_relative_coords_table[:, :, :, 0] /= (self.pretrained_window_size[0] - 1)
            self.sampled_relative_coords_table[:, :, :, 1] /= (self.pretrained_window_size[1] - 1)
        else:
            self.sampled_relative_coords_table[:, :, :, 0] /= (self.sampled_window_size[0] - 1)
            self.sampled_relative_coords_table[:, :, :, 1] /= (self.sampled_window_size[1] - 1)
        self.sampled_relative_coords_table *= 8  # normalize to -8, 8
        self.sampled_relative_coords_table = torch.sign(self.sampled_relative_coords_table) * torch.log2(
            torch.abs(self.sampled_relative_coords_table) + 1.0) / np.log2(8)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.sampled_window_size[0], device=next(self.parameters()).device)
        coords_w = torch.arange(self.sampled_window_size[1], device=next(self.parameters()).device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.sampled_window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.sampled_window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.sampled_window_size[1] - 1
        self.sampled_relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww

        self.sampled_qkv_weight = torch.cat([self.qkv.weight[i:self.sampled_qk_dim*3:3, :self.sampled_dim] for i in range(3)], dim =0)
        if self.q_bias is not None:
            self.sampled_q_bias = self.q_bias[:self.sampled_qk_dim]
        if self.v_bias is not None:
            self.sampled_v_bias = self.v_bias[:self.sampled_qk_dim]
        self.sampled_proj_weight = self.proj.weight[:self.sampled_dim, :self.sampled_qk_dim]
        if self.proj.bias is not None:
            self.sampled_proj_bias = self.proj.bias[:self.sampled_dim]
        if self.scale:
            self.sampled_scale = self.dim / self.sampled_qk_dim
            self.sampled_cpb_scale = self.num_heads / self.sampled_num_heads
        self.attn_drop.p = self.sampled_attn_drop
        self.proj_drop.p = self.sampled_proj_drop

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv_bias = None
        if self.sampled_q_bias is not None:
            qkv_bias = torch.cat((self.sampled_q_bias, torch.zeros_like(self.sampled_v_bias, requires_grad=False), self.sampled_v_bias))
        qkv = F.linear(input=x, weight=self.sampled_qkv_weight, bias=qkv_bias) * (self.dim / self.sampled_qk_dim if self.scale else 1)
        qkv = qkv.reshape(B_, N, 3, self.sampled_num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        logit_scale = torch.clamp(self.sampled_logit_scale, max=torch.log(torch.tensor(1. / 0.01))).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp[:-1](self.sampled_relative_coords_table)
        relative_position_bias_table = F.linear(input=relative_position_bias_table, weight=self.sampled_cpb_mlp_2_weight, bias=self.sampled_cpb_mlp_2_bias) * (self.sampled_cpb_scale if self.scale else 1)
        relative_position_bias_table = relative_position_bias_table.reshape(-1, self.sampled_num_heads)
        relative_position_bias = relative_position_bias_table[self.sampled_relative_position_index.view(-1)].reshape(
            self.sampled_window_size[0] * self.sampled_window_size[1], self.sampled_window_size[0] * self.sampled_window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.sampled_num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.sampled_num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, -1) * (self.sampled_scale if self.scale else 1)
        x = F.linear(x, self.sampled_proj_weight, self.sampled_proj_bias) * (self.dim / self.sampled_dim if self.scale else 1)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, ' \
               f'pretrained_window_size={self.pretrained_window_size}, num_heads={self.num_heads}'

    def params(self):
        params = 0
        params += self.sampled_logit_scale.numel()
        params += self.cpb_mlp[0].weight.numel()
        if self.cpb_mlp[0].bias is not None:
            params += self.cpb_mlp[0].bias.numel()
        params += self.sampled_cpb_mlp_2_weight.numel()
        if self.sampled_cpb_mlp_2_bias is not None:
            params += self.sampled_cpb_mlp_2_bias.numel()
        params += self.sampled_qkv_weight.numel()
        if self.sampled_q_bias is not None:
            params += self.sampled_q_bias.numel()
        if self.sampled_v_bias is not None:
            params += self.sampled_v_bias.numel()
        params += self.sampled_proj_weight.numel()
        if self.sampled_proj_bias is not None:
            params += self.sampled_proj_bias.numel()
        return params

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.sampled_dim * 3 * self.sampled_qk_dim
        # attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        flops += self.sampled_num_heads * N * (self.sampled_qk_dim // self.sampled_num_heads) * N
        # relative_position_bias_table = self.cpb_mlp(relative_coords_table)
        flops += (2 * self.sampled_window_size[0] - 1) * (2 * self.sampled_window_size[1] - 1) * (2 * 512 + 512 * self.sampled_num_heads)
        # attn = attn + relative_position_bias.unsqueeze(0)
        flops += self.sampled_num_heads * N * N
        #  x = (attn @ v)
        flops += self.sampled_num_heads * N * N * (self.sampled_qk_dim // self.sampled_num_heads)
        # x = self.proj(x)
        flops += N * self.sampled_dim * self.sampled_qk_dim
        return flops


class SuperSwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        pretrained_window_size (int): Window size in pre-training.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=SuperLayerNorm, pretrained_window_size=0, scale=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = SuperWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            pretrained_window_size=to_2tuple(pretrained_window_size), scale=scale)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SuperMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, scale=scale)

        self.scale = scale
        # sampled
        self.is_identity_layer = None
        self.sampled_dim = None
        self.sampled_num_heads = None
        self.sampled_window_size = None
        self.sampled_mlp_ratio = None
        self.sampled_drop = None
        self.sampled_attn_drop = None
        self.sampled_drop_path = None
        self.sampled_shift_size = None
        self.sampled_attn_mask = None

    def set_sample_config(self, is_identity_layer, sampled_dim=None, sampled_num_heads=None, sampled_window_size=None, sampled_mlp_ratio=None, sampled_drop=None, sampled_attn_drop=None, sampled_drop_path=None):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False
        self.sampled_dim = sampled_dim
        self.sampled_num_heads = sampled_num_heads
        self.sampled_window_size = sampled_window_size
        self.sampled_mlp_ratio = sampled_mlp_ratio
        self.sampled_drop = sampled_drop
        self.sampled_attn_drop = sampled_attn_drop
        self.sampled_drop_path = sampled_drop_path
        self.sampled_shift_size = 0 if self.shift_size == 0 else self.sampled_window_size // 2
        if min(self.input_resolution) <= self.sampled_window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.sampled_shift_size = 0
            self.sampled_window_size = min(self.input_resolution)
        assert 0 <= self.sampled_shift_size < self.sampled_window_size, "shift_size must in 0-window_size"

        self.norm1.set_sample_config(self.sampled_dim)
        self.attn.set_sample_config(self.sampled_dim, self.sampled_num_heads, self.sampled_drop, self.sampled_attn_drop, to_2tuple(self.sampled_window_size))
        if isinstance(self.drop_path, DropPath):
            self.drop_path.drop_prob = self.sampled_drop_path
        self.norm2.set_sample_config(self.sampled_dim)
        self.sampled_mlp_hidden_dim = int(self.sampled_dim * self.sampled_mlp_ratio)
        self.mlp.set_sample_config(self.sampled_dim, self.sampled_mlp_hidden_dim, None, self.sampled_drop)

        if self.sampled_shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            Hp = int(np.ceil(H / self.sampled_window_size)) * self.sampled_window_size
            Wp = int(np.ceil(W / self.sampled_window_size)) * self.sampled_window_size
            img_mask = torch.zeros((1, Hp, Wp, 1), device=next(self.parameters()).device)  # 1 H W 1
            h_slices = (slice(0, -self.sampled_window_size),
                        slice(-self.sampled_window_size, -self.sampled_shift_size),
                        slice(-self.sampled_shift_size, None))
            w_slices = (slice(0, -self.sampled_window_size),
                        slice(-self.sampled_window_size, -self.sampled_shift_size),
                        slice(-self.sampled_shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.sampled_window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.sampled_window_size * self.sampled_window_size)
            self.sampled_attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            self.sampled_attn_mask = self.sampled_attn_mask.masked_fill(self.sampled_attn_mask != 0, float(-100.0)).masked_fill(self.sampled_attn_mask == 0, float(0.0))
        else:
            self.sampled_attn_mask = None

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        if self.is_identity_layer:
            return x

        shortcut = x
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_l = pad_t = 0
        pad_r = (self.sampled_window_size - W % self.sampled_window_size) % self.sampled_window_size
        pad_b = (self.sampled_window_size - H % self.sampled_window_size) % self.sampled_window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.sampled_shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.sampled_shift_size, -self.sampled_shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.sampled_window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.sampled_window_size * self.sampled_window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.sampled_attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.sampled_window_size, self.sampled_window_size, C)
        shifted_x = window_reverse(attn_windows, self.sampled_window_size, Hp, Wp)  # B H' W' C

        # reverse cyclic shift
        if self.sampled_shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.sampled_shift_size, self.sampled_shift_size), dims=(1, 2))
        else:
            x = shifted_x
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(self.norm1(x))

        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def params(self):
        params = 0
        if not self.is_identity_layer:
            params += self.norm1.params()
            params += self.attn.params()
            params += self.norm2.params()
            params += self.mlp.params()
        return params

    def flops(self):
        flops = 0
        if not self.is_identity_layer:
            H, W = self.input_resolution
            # norm1
            flops += self.norm1.flops(H * W)
            # W-MSA/SW-MSA
            nW = H * W / self.sampled_window_size / self.sampled_window_size
            flops += nW * self.attn.flops(self.sampled_window_size * self.sampled_window_size)
            # mlp
            flops += self.mlp.flops(H * W)
            # norm2
            flops += self.norm2.flops(H * W)
        return flops


class SuperBasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        pretrained_window_size (int): Local window size in pre-training.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=SuperLayerNorm, downsample=None, use_checkpoint=False,
                 pretrained_window_size=0, scale=False, shift=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SuperSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) or (not shift) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer,
                                 pretrained_window_size=pretrained_window_size,
                                 scale=scale)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer, scale=scale)
        else:
            self.downsample = None

        self.scale = scale
        # sampled
        self.sampled_depth = None
        self.sampled_dim = None
        self.sampled_mlp_ratio = None
        self.sampled_num_heads = None
        self.sampled_window_size = None
        self.sampled_drop = None
        self.sampled_attn_drop = None
        self.sampled_drop_path = None

    def set_sample_config(self, sampled_depth, sampled_dim, sampled_num_heads, sampled_window_size, sampled_mlp_ratio, sampled_drop, sampled_attn_drop, sampled_drop_path):
        self.sampled_depth = sampled_depth
        self.sampled_dim = sampled_dim
        self.sampled_mlp_ratio = sampled_mlp_ratio
        self.sampled_num_heads = sampled_num_heads
        self.sampled_window_size = sampled_window_size
        self.sampled_drop = sampled_drop
        self.sampled_attn_drop = sampled_attn_drop
        self.sampled_drop_path = sampled_drop_path
        for i, block in enumerate(self.blocks):
            if i < self.sampled_depth:
                block.set_sample_config(False, self.sampled_dim, self.sampled_num_heads[i], self.sampled_window_size[i], self.sampled_mlp_ratio[i], self.sampled_drop, self.sampled_attn_drop, self.sampled_drop_path[i])
            else:
                block.set_sample_config(True)

        if self.downsample is not None:
            self.downsample.set_sample_config(self.sampled_dim)

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def params(self):
        params = 0
        for blk in self.blocks:
            params += blk.params()
        if self.downsample is not None:
            params += self.downsample.params()
        return params

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def _init_respostnorm(self):
        for blk in self.blocks:
            nn.init.constant_(blk.norm1.bias, 0)
            nn.init.constant_(blk.norm1.weight, 0)
            nn.init.constant_(blk.norm2.bias, 0)
            nn.init.constant_(blk.norm2.weight, 0)


class SuperSwinTransformerV2(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        pretrained_window_sizes (tuple(int)): Pretrained window sizes of each layer.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=SuperLayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, pretrained_window_sizes=[0, 0, 0, 0], scale=False, shift=True, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        # split image into non-overlapping patches
        self.patch_embed = SuperPatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, scale=scale)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = SuperBasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=SuperPatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint,
                               pretrained_window_size=pretrained_window_sizes[i_layer],
                               scale=scale,
                               shift=shift)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)
        for bly in self.layers:
            bly._init_respostnorm()

        self.scale = scale
        # sampled
        self.sampled_depths = None
        self.sampled_num_layers = None
        self.sampled_embed_dim = None
        self.sampled_num_features = None
        self.sampled_mlp_ratio = None
        self.sampled_num_heads = None
        self.sampled_window_size = None
        self.sampled_dpr = None

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"cpb_mlp", "logit_scale", 'relative_position_bias_table'}

    def set_sample_config(self, config):
        self.sampled_depths = config.DEPTHS
        self.sampled_num_layers = len(self.sampled_depths)
        self.sampled_embed_dim = config.EMBED_DIM
        self.sampled_num_features = int(self.sampled_embed_dim * 2 ** (self.sampled_num_layers - 1))
        self.sampled_mlp_ratio = config.MLP_RATIO
        self.sampled_num_heads = config.NUM_HEADS
        self.sampled_window_size = config.WINDOW_SIZE

        self.patch_embed.set_sample_config(self.sampled_embed_dim)

        if self.ape:
            self.sampled_absolute_pos_embed = self.absolute_pos_embed[:, :, :self.sampled_embed_dim]

        self.pos_drop.p = self.pos_drop.p * self.sampled_embed_dim / self.embed_dim

        self.sampled_dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.sampled_depths))]

        cum_depths = [0] + np.cumsum(self.sampled_depths).tolist()
        for i_layer in range(self.sampled_num_layers):
            sampled_dim = int(self.sampled_embed_dim * 2 ** i_layer)
            dim = int(self.embed_dim * 2 ** i_layer)
            sampled_drop = self.drop_rate * sampled_dim / dim
            sampled_attn_drop = self.attn_drop_rate * sampled_dim / dim
            self.layers[i_layer].set_sample_config(
                self.sampled_depths[i_layer],
                sampled_dim,
                self.sampled_num_heads[cum_depths[i_layer]: cum_depths[i_layer+1]],
                self.sampled_window_size[cum_depths[i_layer]: cum_depths[i_layer+1]],
                self.sampled_mlp_ratio[cum_depths[i_layer]: cum_depths[i_layer+1]],
                sampled_drop,
                sampled_attn_drop,
                self.sampled_dpr[cum_depths[i_layer]: cum_depths[i_layer+1]])

        self.norm.set_sample_config(self.sampled_num_features)
        if self.num_classes > 0:
            self.sampled_head_weight = self.head.weight[:, :self.sampled_num_features]
            self.sampled_head_bias = self.head.bias

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.sampled_absolute_pos_embed
        x = self.pos_drop(x)

        for i_layer in range(self.sampled_num_layers):
            x = self.layers[i_layer](x)

        x = self.norm(x)  # B L C
        x = self.avgpool(x.transpose(1, 2))  # B C 1
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = F.linear(x, self.sampled_head_weight, self.sampled_head_bias)
        return x

    def params(self):
        params = 0
        params += self.patch_embed.params()
        if self.ape:
            params += self.sampled_absolute_pos_embed.numel()
        for i_layer in range(self.sampled_num_layers):
            params += self.layers[i_layer].params()
        params += self.norm.params()
        params += self.sampled_head_weight.numel()
        if self.sampled_head_bias is not None:
            params += self.sampled_head_bias.numel()
        return params

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        if self.ape:
            flops += self.sampled_absolute_pos_embed.numel()
        for i_layer in range(self.sampled_num_layers):
            flops += self.layers[i_layer].flops()
        flops += self.norm.flops(self.patches_resolution[0] * self.patches_resolution[1] // (2 ** (self.sampled_num_layers-1)) // (2 ** (self.sampled_num_layers-1)))
        flops += self.sampled_num_features * self.num_classes
        return flops
