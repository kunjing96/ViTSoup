import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec
#from natten import NATTENQKRPBFunction, NATTENAVFunction

def _get_norm_super(norm, out_channels):
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "LN": LayerNormSuper,
            "GN": lambda channels: GroupNormSuper(32, channels),
        }[norm]
    return norm(out_channels)


class LinearSuper(nn.Linear):
    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear='linear', scale=False):
        # super_in_dim and super_out_dim indicate the largest network!
        if isinstance(super_in_dim, list):
            self.mh, self.mw, self.dim = super_in_dim[0], super_in_dim[1], super_in_dim[2]
            self.super_in_dim = self.mh * self.mw * self.dim
        else:
            self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        super().__init__(self.super_in_dim, super_out_dim, bias=bias)

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self.scale = scale
        self._reset_parameters(bias, uniform_, non_linear)

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def set_sample_config_4merging(self, mh, mw, dim, sample_out_dim):
        self.sample_mh = mh
        self.sample_mw = mw
        self.sample_dim = dim
        self.sample_out_dim = sample_out_dim
        self.samples['weight'] = self.weight.reshape(self.super_out_dim, self.mh, self.mw, -1)[:self.sample_out_dim, :self.sample_mh, :self.sample_mw, :self.sample_dim].reshape(self.sample_out_dim, -1)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim / self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = self.bias.reshape(2, self.super_out_dim // 2)[:, :self.sample_dim].reshape(-1)
        return self.samples

    def _sample_parameters(self):
        self.samples['weight'] = linear_sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim / self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = linear_sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        return F.linear(x, self.samples['weight'], self.samples['bias']) * (self.sample_scale if self.scale else 1)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length *  np.prod(self.samples['weight'].size())
        return total_flops


def linear_sample_weight(weight, sample_in_dim, sample_out_dim):
    sample_weight = weight[:, :sample_in_dim]
    sample_weight = sample_weight[:sample_out_dim, :]

    return sample_weight


def linear_sample_bias(bias, sample_out_dim):
    sample_bias = bias[:sample_out_dim]

    return sample_bias


class LayerNormSuper(nn.LayerNorm):
    def __init__(self, super_embed_dim):
        # the largest embed dim
        if isinstance(super_embed_dim, list):
            self.mh, self.mw, self.dim = super_embed_dim[0], super_embed_dim[1], super_embed_dim[2]
            self.super_embed_dim = self.mh * self.mw * self.dim
        else:
            self.super_embed_dim = super_embed_dim

        super().__init__(self.super_embed_dim)

        # the current sampled embed dim
        self.sample_embed_dim = None

        self.samples = {}

    def _sample_parameters(self):
        self.samples['weight'] = self.weight[:self.sample_embed_dim]
        self.samples['bias'] = self.bias[:self.sample_embed_dim]
        return self.samples

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self._sample_parameters()

    def set_sample_config_4merging(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.samples['weight'] = self.weight.reshape(self.mh, self.mw, -1)[:self.sample_mh, :self.sample_mw, :].reshape(-1)
        self.samples['bias'] = self.bias.reshape(self.mh, self.mw, -1)[:self.sample_mh, :self.sample_mw, :].reshape(-1)

    def forward(self, x):
        return F.layer_norm(x, (self.sample_embed_dim,), weight=self.samples['weight'], bias=self.samples['bias'], eps=self.eps)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        assert 'bias' in self.samples.keys()
        return self.samples['weight'].numel() + self.samples['bias'].numel()

    def get_complexity(self, sequence_length):
        return sequence_length * self.sample_embed_dim


class GroupNormSuper(nn.GroupNorm):
    def __init__(self, num_groups, super_embed_dim):
        # the largest embed dim
        if isinstance(super_embed_dim, list):
            self.mh, self.mw, self.dim = super_embed_dim[0], super_embed_dim[1], super_embed_dim[2]
            self.super_embed_dim = self.mh * self.mw * self.dim
        else:
            self.super_embed_dim = super_embed_dim
        
        super().__init__(num_groups, self.super_embed_dim)

        # the current sampled embed dim
        self.sample_embed_dim = None

        self.samples = {}

    def _sample_parameters(self):
        self.samples['weight'] = self.weight[:self.sample_embed_dim]
        self.samples['bias'] = self.bias[:self.sample_embed_dim]

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self._sample_parameters()

    def set_sample_config_4merging(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.samples['weight'] = self.weight.reshape(self.mh, self.mw, -1)[:self.sample_mh, :self.sample_mw, :].reshape(-1)
        self.samples['bias'] = self.bias.reshape(self.mh, self.mw, -1)[:self.sample_mh, :self.sample_mw, :].reshape(-1)

    def forward(self, x):
        return F.group_norm(x, self.num_groups, weight=self.samples['weight'], bias=self.samples['bias'], eps=self.eps)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        assert 'bias' in self.samples.keys()
        return self.samples['weight'].numel() + self.samples['bias'].numel()

    def get_complexity(self, sequence_length):
        return sequence_length * self.sample_embed_dim


class qkv_super(nn.Linear):

    def __init__(self, super_in_dim, super_out_dim, bias=True, uniform_=None, non_linear='linear', scale=False):
        super().__init__(super_in_dim, super_out_dim, bias=bias)

        # super_in_dim and super_out_dim indicate the largest network!
        self.super_in_dim = super_in_dim
        self.super_out_dim = super_out_dim

        # input_dim and output_dim indicate the current sampled size
        self.sample_in_dim = None
        self.sample_out_dim = None

        self.samples = {}

        self.scale = scale
        # self._reset_parameters(bias, uniform_, non_linear)
        self.profiling = False

    def profile(self, mode=True):
        self.profiling = mode

    def sample_parameters(self, resample=False):
        if self.profiling or resample:
            return self._sample_parameters()
        return self.samples

    def _reset_parameters(self, bias, uniform_, non_linear):
        nn.init.xavier_uniform_(self.weight) if uniform_ is None else uniform_(
            self.weight, non_linear=non_linear)
        if bias:
            nn.init.constant_(self.bias, 0.)

    def set_sample_config(self, sample_in_dim, sample_out_dim):
        self.sample_in_dim = sample_in_dim
        self.sample_out_dim = sample_out_dim

        self._sample_parameters()

    def _sample_parameters(self):
        self.samples['weight'] = qkv_sample_weight(self.weight, self.sample_in_dim, self.sample_out_dim)
        self.samples['bias'] = self.bias
        self.sample_scale = self.super_out_dim/self.sample_out_dim
        if self.bias is not None:
            self.samples['bias'] = qkv_sample_bias(self.bias, self.sample_out_dim)
        return self.samples

    def forward(self, x):
        self.sample_parameters()
        return F.linear(x, self.samples['weight'], self.samples['bias']) * (self.sample_scale if self.scale else 1)

    def calc_sampled_param_num(self):
        assert 'weight' in self.samples.keys()
        weight_numel = self.samples['weight'].numel()

        if self.samples['bias'] is not None:
            bias_numel = self.samples['bias'].numel()
        else:
            bias_numel = 0

        return weight_numel + bias_numel

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += sequence_length *  np.prod(self.samples['weight'].size())
        return total_flops


def qkv_sample_weight(weight, sample_in_dim, sample_out_dim):
    sample_weight = weight[:, :sample_in_dim]
    sample_weight = torch.cat([sample_weight[i:sample_out_dim:3, :] for i in range(3)], dim =0)

    return sample_weight


def qkv_sample_bias(bias, sample_out_dim):
    sample_bias = torch.cat([bias[i:sample_out_dim:3] for i in range(3)], dim =0)

    return sample_bias


def softmax(x, dim, onnx_trace=False):
    if onnx_trace:
        return F.softmax(x.float(), dim=dim)
    else:
        return F.softmax(x, dim=dim, dtype=torch.float32)


class AttentionSuper(nn.Module):

    def __init__(self, super_embed_dim, num_heads=8, qkv_bias=False, qk_scale=False, attn_drop=0., proj_drop=0., normalization = False, relative_position = False, num_patches = None, max_relative_position=14, scale=False, change_qkv = False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = super_embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.super_embed_dim = super_embed_dim
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.num_heads = num_heads

        self.fc_scale = scale
        self.change_qkv = change_qkv
        if change_qkv:
            self.qkv = qkv_super(super_embed_dim, 3 * super_embed_dim, bias=qkv_bias)
        else:
            self.qkv = LinearSuper(super_embed_dim, 3 * super_embed_dim, bias=qkv_bias)

        self.relative_position = relative_position
        if self.relative_position:
            self.relative_h_position_bias_table = nn.Parameter(
                torch.zeros(num_heads, 2 * max_relative_position - 1)
            )  # 2*Wh-1, nH
            trunc_normal_(self.relative_h_position_bias_table, std=0.02)
            self.relative_v_position_bias_table = nn.Parameter(
                torch.zeros(num_heads, 2 * max_relative_position - 1)
            )  # 2*Wh-1, nH
            trunc_normal_(self.relative_v_position_bias_table, std=0.02)
        self.max_relative_position = max_relative_position

        self.sample_qk_embed_dim = None
        self.sample_v_embed_dim = None
        self.sample_num_heads = None
        self.sample_scale = None
        self.sample_in_embed_dim = None
        self.sample_relative_h_position_bias_table = None
        self.sample_relative_v_position_bias_table = None

        self.proj = LinearSuper(super_embed_dim, super_embed_dim)

        # self.attn_drop = nn.Dropout(attn_drop)
        # self.proj_drop = nn.Dropout(proj_drop)

    def set_sample_config(self, sample_q_embed_dim=None, sample_num_heads=None, sample_mask_ratio=None, sample_in_embed_dim=None, sample_dropout=None, sample_attn_dropout=None):
        self.sample_dropout = sample_dropout if sample_dropout else self.proj_drop
        self.sample_attn_dropout = sample_attn_dropout if sample_attn_dropout else self.attn_drop
        self.sample_in_embed_dim = sample_in_embed_dim
        self.sample_num_heads = sample_num_heads
        self.sample_mask_ratio = sample_mask_ratio
        if not self.change_qkv:
            self.sample_qk_embed_dim = self.super_embed_dim
            self.sample_scale = (self.sample_in_embed_dim // self.sample_num_heads) ** -0.5
        else:
            self.sample_qk_embed_dim = sample_q_embed_dim
            self.sample_scale = (self.sample_qk_embed_dim // self.sample_num_heads) ** -0.5

        self.qkv.set_sample_config(sample_in_dim=sample_in_embed_dim, sample_out_dim=3*self.sample_qk_embed_dim)
        self.proj.set_sample_config(sample_in_dim=self.sample_qk_embed_dim, sample_out_dim=sample_in_embed_dim)
        if self.relative_position:
            start = (self.relative_v_position_bias_table.size(1) - (2 * self.sample_mask_ratio - 1)) // 2
            self.sample_relative_h_position_bias_table = self.relative_h_position_bias_table[:self.sample_num_heads, start:start+2*self.sample_mask_ratio-1]
            self.sample_relative_v_position_bias_table = self.relative_v_position_bias_table[:self.sample_num_heads, start:start+2*self.sample_mask_ratio-1]

    def calc_sampled_param_num(self):
        return self.qkv.calc_sampled_param_num() + (self.sample_relative_h_position_bias_table.numel() + self.sample_relative_v_position_bias_table.numel()) if self.relative_position else 0 + self.proj.calc_sampled_param_num()

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.qkv.get_complexity(sequence_length)
        if self.sample_mask_ratio not in [7, 5, 3]:
            # attn
            total_flops += sequence_length * sequence_length * self.sample_qk_embed_dim
            # x
            total_flops += sequence_length * sequence_length * self.sample_qk_embed_dim
        else:
            num_tokens = int(self.sample_mask_ratio ** 2)
            # attn
            total_flops += sequence_length * num_tokens * self.sample_qk_embed_dim
            # x
            total_flops += sequence_length * num_tokens * self.sample_qk_embed_dim
        if self.relative_position:
            total_flops += 2 * sequence_length * num_tokens * self.sample_num_heads
        total_flops += self.proj.get_complexity(sequence_length)
        return total_flops

    def get_reference_points(self, H, W):
        device = next(self.parameters()).device
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(H, device=device)
        coords_w = torch.arange(W, device=device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += H - 1  # shift to start from 0
        relative_coords[:, :, 1] += W - 1
        return relative_coords[:, :, 0], relative_coords[:, :, 1]

    def forward(self, x, H, W):
        B, N, C = x.shape
        assert N == H * W, "input feature has wrong size"

        qkv = self.qkv(x).reshape(B, N, 3, self.sample_num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # make torchscript happy (cannot use tensor as tuple)
        q = q * self.sample_scale

        if self.sample_mask_ratio not in [7, 5, 3]:
            attn = (q @ k.transpose(-2, -1)).reshape(B, self.sample_num_heads, N, -1) # B H Nq Nk / B H Nq K
            if self.relative_position:
                relative_h_position_index, relative_v_position_index = self.get_reference_points(self.sample_mask_ratio, self.sample_mask_ratio)
                relative_h_position_bias = self.sample_relative_h_position_bias_table[:, relative_h_position_index.view(-1)].view(-1, H * W, H * W)  # nH,Wh*Ww,Wh*Ww
                relative_v_position_bias = self.sample_relative_v_position_bias_table[:, relative_v_position_index.view(-1)].view(-1, H * W, H * W)  # nH,Wh*Ww,Wh*Ww
                relative_position_bias = relative_h_position_bias.unsqueeze(0) + relative_v_position_bias.unsqueeze(0) # 1,nH,Wh*Ww,Wh*Ww
                relative_position_bias = F.interpolate(relative_position_bias, size=(H*W, H*W), mode='bilinear')
                attn = attn + relative_position_bias

            attn = attn.softmax(dim=-1) # B H Nq Nk
            #attn = self.attn_drop(attn)
            attn = F.dropout(attn, p=self.sample_attn_dropout, training=self.training)

            x = (attn @ v).transpose(1,2).reshape(B, N, -1)

        else:
            q = q.reshape(B, self.sample_num_heads, H, W, -1)
            k = k.reshape(B, self.sample_num_heads, H, W, -1)
            v = v.reshape(B, self.sample_num_heads, H, W, -1)

            if self.relative_position:
                relative_position_bias = self.sample_relative_v_position_bias_table[:, :, None] + self.sample_relative_h_position_bias_table[:, None, :]
            else:
                relative_position_bias = torch.zeros((self.sample_num_heads, 2 * self.sample_mask_ratio - 1, 2 * self.sample_mask_ratio - 1), device=q.device)
            attn = NATTENQKRPBFunction.apply(q, k, relative_position_bias)

            attn = attn.softmax(dim=-1)
            #attn = self.attn_drop(attn)
            attn = F.dropout(attn, p=self.sample_attn_dropout, training=self.training)

            x = NATTENAVFunction.apply(attn, v)
            x = x.permute(0, 2, 3, 1, 4).reshape(B, N, -1)

        if self.fc_scale:
            x = x * (self.super_embed_dim / self.sample_qk_embed_dim)
        x = self.proj(x)
        #x = self.proj_drop(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        return x


class TransformerBlockSuper(nn.Module):
    """Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        pre_norm=True,
        norm_layer='LN',
        scale=False,
        relative_position=False,
        change_qkv=False,
        max_relative_position=14,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.pre_norm = pre_norm
        self.scale = scale
        self.relative_position = relative_position

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_mask_ratio = None
        self.sample_num_heads_this_layer = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_attn_dropout = None
        self.is_identity_layer = None

        self.attn_layer_norm = _get_norm_super(norm_layer, dim)
        self.attn = AttentionSuper(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            scale=self.scale,
            relative_position=self.relative_position,
            change_qkv=change_qkv,
            max_relative_position=max_relative_position,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ffn_layer_norm = _get_norm_super(norm_layer, dim)
        self.fc1 = LinearSuper(super_in_dim=self.dim, super_out_dim=self.super_ffn_embed_dim_this_layer)
        self.fc2 = LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer, super_out_dim=self.dim)
        self.activation_fn = act_layer()

    def set_sample_config(self, is_identity_layer, sample_mask_ratio=None, sample_embed_dim=None, sample_mlp_ratio=None, sample_num_heads=None, sample_dropout=None, sample_attn_dropout=None, sample_out_dim=None):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_mask_ratio = sample_mask_ratio
        self.sample_num_heads_this_layer = sample_num_heads
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_ffn_embed_dim_this_layer = int(sample_mlp_ratio * sample_embed_dim)
        self.sample_scale = self.mlp_ratio / sample_mlp_ratio

        self.sample_dropout = sample_dropout
        self.sample_attn_dropout = sample_attn_dropout
        self.attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        self.attn.set_sample_config(sample_q_embed_dim=self.sample_num_heads_this_layer*(self.dim//self.num_heads), sample_num_heads=self.sample_num_heads_this_layer, sample_mask_ratio=self.sample_mask_ratio, sample_in_embed_dim=self.sample_embed_dim, sample_dropout=self.sample_dropout, sample_attn_dropout=self.sample_attn_dropout)

        self.fc1.set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_out_dim)

        self.ffn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

    def forward(self, x, H, W):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, patch_num , sample_embed_dim)`

        Returns:
            encoded output of shape `(batch, patch_num, sample_embed_dim)`
        """
        if self.is_identity_layer:
            return x

        residual = x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x, H=H, W=W)
        #x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)
        # compute the ffn
        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        if self.scale:
            x = x * (self.super_mlp_ratio / self.sample_mlp_ratio)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.pre_norm:
            return layer_norm(x)
        else:
            return x

    def calc_sampled_param_num(self):
        if self.is_identity_layer:
            return 0
        return self.attn_layer_norm.calc_sampled_param_num() + self.attn.calc_sampled_param_num() + self.ffn_layer_norm.calc_sampled_param_num() + self.fc1.calc_sampled_param_num() + self.fc2.calc_sampled_param_num()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.is_identity_layer:
            return total_flops
        total_flops += self.attn_layer_norm.get_complexity(sequence_length)
        total_flops += self.attn.get_complexity(sequence_length)
        total_flops += self.ffn_layer_norm.get_complexity(sequence_length)
        total_flops += self.fc1.get_complexity(sequence_length)
        total_flops += self.fc2.get_complexity(sequence_length)
        return total_flops


def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim


class PatchMergingSuper(nn.Module):
    """Patch Merging Layer
    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, merge_size=2, pre_norm=True, norm_layer='LN'):
        super().__init__()
        merge_size = to_2tuple(merge_size)
        self.merge_size = merge_size
        self.dim = dim
        self.reduction = LinearSuper([merge_size[0], merge_size[1], dim], 2 * dim, bias=False)
        self.pre_norm = pre_norm
        if pre_norm:
            self.norm = _get_norm_super(norm_layer, [merge_size[0], merge_size[1], dim])
        else:
            self.norm = _get_norm_super(norm_layer, 2 * dim)

        # sampled_
        self.sample_merge_size = None

    def set_sample_config(self, sample_merge_size, dim):
        self.sample_merge_size = to_2tuple(sample_merge_size)
        self.sample_dim = dim
        self.reduction.set_sample_config_4merging(self.sample_merge_size[0], self.sample_merge_size[1], dim, sample_out_dim=2 * dim)
        if self.pre_norm:
            self.norm.set_sample_config_4merging(self.sample_merge_size[0], self.sample_merge_size[1])
        else:
            self.norm.set_sample_config(sample_embed_dim=2 * dim)

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.pre_norm:
            return layer_norm(x)
        else:
            return x

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        if self.sample_merge_size[1] - 2 + W % 2 != 0:
            x = F.pad(x, (0, 0, 0, self.sample_merge_size[1] - 2 + W % 2))
        if self.sample_merge_size[0] - 2 + H % 2 != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.sample_merge_size[0] - 2 + H % 2))

        x_list = []
        for i in range(self.sample_merge_size[0]):
            for j in range(self.sample_merge_size[1]):
                x_list.append(x[:, i::2, j::2, :][:, :(H+1)//2, :(W+1)//2, :])  # B H/2 W/2 C
        x = torch.cat(x_list, -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, self.sample_merge_size[0] * self.sample_merge_size[1] * C)  # B H/2*W/2 4*C

        x = self.maybe_layer_norm(self.norm, x, before=True)
        x = self.reduction(x)
        x = self.maybe_layer_norm(self.norm, x, after=True)
        return x

    def calc_sampled_param_num(self):
        return self.reduction.calc_sampled_param_num() + self.norm.calc_sampled_param_num()

    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.reduction.get_complexity(sequence_length)
        total_flops += self.norm.get_complexity(sequence_length)
        return total_flops


class BasicLayerSuper(nn.Module):
    """A basic Transformer layer for one stage.
    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        window_size (int): Local window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        depth,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        pre_norm=True,
        norm_layer='LN',
        scale=False,
        relative_position=False,
        change_qkv=False,
        max_relative_position=14,
        merge_size=5,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.depth = depth
        self.dim = dim
        self.drop = drop
        self.attn_drop = attn_drop
        self.pre_norm = pre_norm
        self.use_checkpoint = use_checkpoint
        
        # configs for the sampled subTransformer
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        self.sample_dropout = None
        self.sample_output_dim = None

        # build blocks
        self.blocks = nn.ModuleList(
            [
                TransformerBlockSuper(
                    dim=dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    act_layer=act_layer,
                    pre_norm=pre_norm,
                    norm_layer=norm_layer,
                    scale=scale,
                    relative_position=relative_position,
                    change_qkv=change_qkv,
                    max_relative_position=max_relative_position,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, merge_size=merge_size, pre_norm=pre_norm, norm_layer=norm_layer)
        else:
            self.downsample = None

    def set_sample_config(self, sample_layer_num, sample_embed_dim=None, sample_mlp_ratio=None, sample_num_heads=None, sample_mask_ratio=None, sample_out_dim=None, sample_merge_size=None):
        for i, block in enumerate(self.blocks):
            self.sample_layer_num = sample_layer_num
            self.sample_embed_dim = sample_embed_dim
            self.sample_mlp_ratio = sample_mlp_ratio
            self.sample_num_heads = sample_num_heads
            self.sample_mask_ratio = sample_mask_ratio
            self.sample_out_dim = sample_out_dim
            self.sample_merge_size = sample_merge_size
            if i < self.sample_layer_num:
                sample_dropout = calc_dropout(self.drop, self.sample_embed_dim, self.dim)
                sample_attn_dropout = calc_dropout(self.attn_drop, self.sample_embed_dim, self.dim)
                block.set_sample_config(is_identity_layer=False,
                                        sample_mask_ratio=self.sample_mask_ratio[i],
                                        sample_embed_dim=self.sample_embed_dim,
                                        sample_mlp_ratio=self.sample_mlp_ratio[i],
                                        sample_num_heads=self.sample_num_heads[i],
                                        sample_dropout=sample_dropout,
                                        sample_attn_dropout=sample_attn_dropout,
                                        sample_out_dim=self.sample_embed_dim)
            # exceeds sample layer number
            else:
                block.set_sample_config(is_identity_layer=True)

        if self.downsample:
            self.downsample.set_sample_config(self.sample_merge_size, self.sample_embed_dim)

    def forward(self, x, H, W):
        """Forward function.
        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, H, W)
            else:
                x = blk(x, H, W)
        if self.downsample is not None:
            x_down = self.downsample(x, H, W)
            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            return x, H, W, x_down, Wh, Ww
        else:
            return x, H, W, x, H, W

    def calc_sampled_param_num(self):
        total_numels = 0
        for i, block in enumerate(self.blocks):
            if i < self.sample_layer_num:
                total_numels += block.calc_sampled_param_num()
        if self.downsample:
            total_numels += self.downsample.calc_sampled_param_num()
        return total_numels

    def get_complexity(self, sequence_length):
        total_flops = 0
        for i, block in enumerate(self.blocks):
            if i < self.sample_layer_num:
                total_flops += block.get_complexity(sequence_length)
        if self.downsample:
            total_flops += self.downsample.get_complexity(sequence_length)
        return total_flops


class PatchEmbedSuper(nn.Module):
    """Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, scale=False, patch_norm=True, norm_layer='LN'):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.scale = scale
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = _get_norm_super(norm_layer, embed_dim)
        self.patch_norm = patch_norm

        # sampled_
        self.sample_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def set_sample_config(self, sample_embed_dim):
        self.sample_embed_dim = sample_embed_dim
        self.sampled_weight = self.proj.weight[:sample_embed_dim, ...]
        self.sampled_bias = self.proj.bias[:self.sample_embed_dim, ...]
        self.norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)
        if self.scale:
            self.sampled_scale = self.super_embed_dim / sample_embed_dim

    def maybe_layer_norm(self, layer_norm, x):
        if self.patch_norm:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = layer_norm(x)
            x = x.transpose(1, 2).reshape(-1, self.sample_embed_dim, Wh, Ww)
            return x
        else:
            return x

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))
        x = F.conv2d(x, self.sampled_weight, self.sampled_bias, stride=self.patch_size, padding=self.proj.padding, dilation=self.proj.dilation) # B C Wh Ww
        if self.scale:
            x = x * self.sampled_scale
        x = self.maybe_layer_norm(self.norm, x)
        return x

    def calc_sampled_param_num(self):
        return  self.sampled_weight.numel() + self.sampled_bias.numel() + self.norm.calc_sampled_param_num()

    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.sampled_bias is not None:
             total_flops += self.sampled_bias.size(0)
        total_flops += sequence_length * np.prod(self.sampled_weight.size())
        total_flops += self.norm.get_complexity(sequence_length)
        return total_flops


class TransformerSuper(nn.Module):
    """Transformer backbone.
    Args:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        depths (tuple[int]): Depths of each Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        window_size (int): Window size. Default: 7.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        pretrain_img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        act_layer=nn.GELU,
        abs_pos=False,
        rel_pos=True,
        max_rel_pos=56,
        patch_norm=True,
        pre_norm=True,
        norm_layer='LN',
        scale=False,
        change_qkv=False,
        merge_size=5,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        use_checkpoint=False,
    ):
        super().__init__()
        self.pretrain_img_size = pretrain_img_size
        self.depths = depths
        self.num_layers = len(depths)
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.embed_dim = embed_dim
        self.abs_pos = abs_pos
        self.rel_pos = rel_pos
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.pre_norm = pre_norm
        self.patch_size = patch_size

        # configs for the sampled subTransformer
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        self.sample_mask_ratio = None
        self.sample_dropout = None
        self.sample_output_dim = None
        self.sample_merge_size = None

        # split image into non-overlapping patches
        self.patch_embed_super = PatchEmbedSuper(
            img_size=pretrain_img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            patch_norm=self.patch_norm,
            norm_layer=norm_layer,
        )

        # absolute position embedding
        if self.abs_pos:
            patches_resolution = self.patch_embed_super.patches_resolution
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # build layers
        self.layers = nn.ModuleList()
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ] # stochastic depth decay rule
        for i_layer in range(self.num_layers):
            layer = BasicLayerSuper(
                depth=depths[i_layer],
                dim=int(embed_dim * (2 ** i_layer)),
                num_heads=num_heads[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                act_layer=act_layer,
                pre_norm=pre_norm,
                norm_layer=norm_layer,
                scale=scale,
                relative_position=rel_pos,
                change_qkv=change_qkv,
                max_relative_position=max_rel_pos,
                merge_size=merge_size,
                downsample=PatchMergingSuper if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]

        if self.pre_norm:
            for i_layer in out_indices:
                layer = _get_norm_super(norm_layer, self.num_features[i_layer])
                layer_name = f"norm{i_layer}"
                self.add_module(layer_name, layer)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1 and self.abs_pos:
            self.absolute_pos_embed.requires_grad = False

        if self.frozen_stages >= 2:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages - 1):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'rel_pos_embed'}

    def set_sample_config(self, config):
        self.sample_layer_num = config.BACKBONE.DEPTHS
        self.sample_embed_dim = [config.BACKBONE.EMBED_DIM, config.BACKBONE.EMBED_DIM*2, config.BACKBONE.EMBED_DIM*4, config.BACKBONE.EMBED_DIM*8]
        self.sample_mlp_ratio = config.BACKBONE.MLP_RATIO
        self.sample_num_heads = config.BACKBONE.NUM_HEADS
        self.sample_mask_ratio = config.BACKBONE.MASK_RATIO
        self.sample_merge_size = config.BACKBONE.MERGE_SIZE
        self.sample_dropout = calc_dropout(self.drop_rate, self.sample_embed_dim[0], self.embed_dim)
        self.sample_output_dim = [out_dim for out_dim in self.sample_embed_dim[1:]] + [self.sample_embed_dim[-1]]
        self.sample_num_features = self.sample_embed_dim
        self.patch_embed_super.set_sample_config(self.sample_embed_dim[0])
        depths = [0] + np.cumsum(self.sample_layer_num).tolist()
        for i, layer in enumerate(self.layers):
            layer.set_sample_config(self.sample_layer_num[i],
                                    self.sample_embed_dim[i],
                                    self.sample_mlp_ratio[depths[i]:depths[i+1]],
                                    self.sample_num_heads[depths[i]:depths[i+1]],
                                    self.sample_mask_ratio[depths[i]:depths[i+1]],
                                    self.sample_output_dim[i],
                                    self.sample_merge_size[i])
        if self.pre_norm:
            for i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                norm_layer.set_sample_config(self.sample_num_features[i])

    def calc_sampled_param_num(self):
        total_numels = 0
        total_numels += self.patch_embed_super.calc_sampled_param_num()
        if self.abs_pos:
            total_numels += self.absolute_pos_embed[:, :self.sample_embed_dim[0], :, :].numel()
        for layer in self.layers:
            total_numels += layer.calc_sampled_param_num()
        if self.pre_norm:
            for i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                total_numels += norm_layer.calc_sampled_param_num()
        return total_numels

    def get_complexity(self, H, W):
        total_flops = 0
        total_flops += self.patch_embed_super.get_complexity((H // self.patch_size) * (W // self.patch_size))
        if self.abs_pos:
            total_flops += np.prod(self.absolute_pos_embed[:, :self.sample_embed_dim[0], :, :].size()) / 2.0
        for i, layer in enumerate(self.layers):
            total_flops += layer.get_complexity((H // self.patch_size // (2**i)) * (W // self.patch_size // (2**i)))
        if self.pre_norm:
            for i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                total_flops += norm_layer.get_complexity((H // self.patch_size // (2**i)) * (W // self.patch_size // (2**i)))
        return total_flops

    def forward(self, x):
        """Forward function."""
        x = self.patch_embed_super(x)
        B, C, H, W = x.size()
        if self.abs_pos:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed[:, :self.sample_embed_dim[0], :, :], size=(H, W), mode="bicubic")
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2) # B, H*W, C
        else:
            x = x.flatten(2).transpose(1, 2) # B, H*W, C
        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        outs = {}
        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, HH, WW, x, H, W = layer(x, H, W) # x, x_down
            if i in self.out_indices:
                if self.pre_norm:
                    norm_layer = getattr(self, f"norm{i}")
                    x_out = norm_layer(x_out)
                out = x_out.view(B, HH, WW, -1).permute(0, 3, 1, 2).contiguous()
                outs["res{}".format(i + 2)] = out

        return outs

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(TransformerSuper, self).train(mode)
        self._freeze_stages()


@BACKBONE_REGISTRY.register()
class D2TransformerSuper(TransformerSuper, Backbone):
    def __init__(self, cfg, input_shape):
        pretrain_img_size = cfg.MODEL.PRETRAIN_IMG_SIZE
        patch_size = cfg.MODEL.BACKBONE.PATCH_SIZE
        in_chans = 3
        embed_dim = cfg.MODEL.BACKBONE.EMBED_DIM
        depths = cfg.MODEL.BACKBONE.DEPTHS
        num_heads = cfg.MODEL.BACKBONE.NUM_HEADS
        mlp_ratio = cfg.MODEL.BACKBONE.MLP_RATIO
        qkv_bias = cfg.MODEL.BACKBONE.QKV_BIAS
        qk_scale = cfg.MODEL.BACKBONE.QK_SCALE
        drop_rate = cfg.MODEL.BACKBONE.DROP_RATE
        attn_drop_rate = cfg.MODEL.BACKBONE.ATTN_DROP_RATE
        drop_path_rate = cfg.MODEL.BACKBONE.DROP_PATH_RATE
        acts = {'gelu': nn.GELU, 'relu': nn.ReLU}
        act_layer = acts[cfg.MODEL.BACKBONE.ACT_LAYER]
        abs_pos = cfg.MODEL.BACKBONE.ABS_POS
        rel_pos = cfg.MODEL.BACKBONE.REL_POS
        max_rel_pos = cfg.MODEL.BACKBONE.MAX_REL_POS
        patch_norm = cfg.MODEL.BACKBONE.PATCH_NORM
        pre_norm = cfg.MODEL.BACKBONE.PRE_NORM
        norm_layer = cfg.MODEL.BACKBONE.NORM
        scale=cfg.MODEL.BACKBONE.SCALE
        change_qkv=cfg.MODEL.BACKBONE.CHANGE_QKV
        merge_size=cfg.MODEL.BACKBONE.MERGE_SIZE
        use_checkpoint = cfg.MODEL.BACKBONE.USE_CHECKPOINT

        super().__init__(
            pretrain_img_size,
            patch_size,
            in_chans,
            embed_dim,
            depths,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            act_layer,
            abs_pos,
            rel_pos,
            max_rel_pos,
            patch_norm,
            pre_norm,
            norm_layer,
            scale,
            change_qkv,
            merge_size,
            use_checkpoint=use_checkpoint,
        )

        self._out_features = cfg.MODEL.BACKBONE.OUT_FEATURES

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": self.num_features[0],
            "res3": self.num_features[1],
            "res4": self.num_features[2],
            "res5": self.num_features[3],
        }

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (N,C,H,W). H, W must be a multiple of ``self.size_divisibility``.
        Returns:
            dict[str->Tensor]: names and the corresponding features
        """
        assert (
            x.dim() == 4
        ), f"TransformerSuper takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        outputs = {}
        y = super().forward(x)
        for k in y.keys():
            if k in self._out_features:
                outputs[k] = y[k]
        return outputs

    def output_shape(self):
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name], stride=self._out_feature_strides[name]
            )
            for name in self._out_features
        }

    @property
    def size_divisibility(self):
        return 32
