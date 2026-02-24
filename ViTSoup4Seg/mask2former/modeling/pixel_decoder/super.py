import numpy as np
from typing import Callable, Dict, List, Optional, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..backbone.swin_super import SuperLinear, SuperLayerNorm

from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.transformer import _get_clones, _get_activation_fn
from .ops.modules import SuperMSDeformAttn


class SuperGroupNorm(nn.GroupNorm):
    def __init__(self, num_groups, embed_dim):
        super().__init__(num_groups, embed_dim)

        # sampled
        self.sampled_embed_dim = None
        self.sampled_weight = None
        self.sampled_bias = None

    def set_sample_config(self, sampled_embed_dim):
        self.sampled_embed_dim = sampled_embed_dim
        self.sampled_weight = self.weight[:self.sampled_embed_dim]
        self.sampled_bias = self.bias[:self.sampled_embed_dim]

    def forward(self, x):
        return F.group_norm(x, self.num_groups, self.sampled_weight, self.sampled_bias, self.eps)

    def params(self):
        params = 0
        params += self.sampled_weight.numel()
        params += self.sampled_bias.numel()
        return params

    def flops(self, N):
        flops = 0
        flops += N * self.sampled_embed_dim * 2
        return flops


def _get_norm_super(norm, out_channels):
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            "LN": SuperLayerNorm,
            "GN": lambda channels: SuperGroupNorm(32, channels),
        }[norm]
    return norm(out_channels)


class SuperPositionEmbeddingSine(PositionEmbeddingSine):

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__(num_pos_feats, temperature, normalize, scale)
        self.sampled_num_pos_feats = None

    def set_sample_config(self, sample_num_pos_feats):
        self.sampled_num_pos_feats = sample_num_pos_feats

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.sampled_num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.sampled_num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def params(self):
        return 0

    def flops(self, sequence_length):
        return 0


class SuperConv2d(Conv2d):

    def __init__(self, *args, **kwargs):
        self.scale = kwargs.pop("scale", None)
        super().__init__(*args, **kwargs)

        # sampled
        self.sampled_in_channels = None
        self.sampled_out_channels = None
        self.sampled_weight = None
        self.sampled_bias = None
        self.sampled_scale = None

    def set_sample_config(self, sample_in_channels, sample_out_channels):
        self.sampled_in_channels = sample_in_channels
        self.sampled_out_channels = sample_out_channels
        self.sampled_weight = self.weight[:self.sampled_out_channels, :self.sampled_in_channels, ...]
        if self.bias is not None:
            self.sampled_bias = self.bias[:self.sampled_out_channels, ...]
        if self.norm is not None:
            self.norm.set_sample_config(self.sampled_out_channels)
        if self.scale:
            self.sampled_scale = self.out_channels / self.sampled_out_channels

    def forward(self, x):
        # torchscript does not support SyncBatchNorm yet
        # https://github.com/pytorch/pytorch/issues/40507
        # and we skip these codes in torchscript since:
        # 1. currently we only support torchscript in evaluation mode
        # 2. features needed by exporting module to torchscript are added in PyTorch 1.6 or
        # later version, `Conv2d` in these PyTorch versions has already supported empty inputs.
        if not torch.jit.is_scripting():
            if x.numel() == 0 and self.training:
                # https://github.com/pytorch/pytorch/issues/12013
                assert not isinstance(
                    self.norm, torch.nn.SyncBatchNorm
                ), "SyncBatchNorm does not support empty inputs!"

        x = F.conv2d(
            x, self.sampled_weight, self.sampled_bias, self.stride, self.padding, self.dilation, self.groups
        ) * (self.sampled_scale if self.scale else 1)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def params(self):
        return self.sampled_weight.numel() + self.sampled_bias.numel() if self.bias is not None else 0 + self.norm.params() if self.norm is not None else 0

    def flops(self, H, W):
        flops = 0
        flops += H // self.stride[0] * W // self.stride[1] * np.prod(self.sampled_weight.size())
        if self.sampled_bias is not None:
             flops += H // self.stride[0] * W // self.stride[1] * np.prod(self.sampled_bias.size())
        if self.norm is not None:
            flops += self.norm.flops(H // self.stride[0] * W // self.stride[1])
        return flops


class SuperMSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, mlp_ratio=4.0,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4, 
                 pre_norm=True, norm="LN",
                 scale=False):
                                                            
        super().__init__()
        self.d_model = d_model
        self.d_ffn = int(mlp_ratio * d_model)
        self.pre_norm = pre_norm
        self.dropout = dropout

        # self attention
        self.self_attn = SuperMSDeformAttn(d_model, n_levels, n_heads, n_points, scale=scale)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = _get_norm_super(norm, d_model)

        # ffn
        self.linear1 = SuperLinear(d_model, self.d_ffn, scale=scale)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = SuperLinear(self.d_ffn, d_model, scale=scale)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = _get_norm_super(norm, d_model)

        self.scale = scale
        # sampled
        self.is_identity_layer = None
        self.sampled_d_model = None
        self.sampled_d_ffn = None
        self.sampled_n_levels = None
        self.sampled_n_heads = None
        self.sampled_n_points = None

    def set_sample_config(self, is_identity_layer, d_model=None, transformer_nlevels=None, transformer_enc_num_points=None, transformer_nheads=None, transformer_mlp_ratio=None):
        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False
        self.sampled_d_model = d_model
        self.sampled_d_ffn = int(transformer_mlp_ratio * self.sampled_d_model)
        self.sampled_n_levels = transformer_nlevels
        self.sampled_n_heads = transformer_nheads
        self.sampled_n_points = transformer_enc_num_points
        self.self_attn.set_sample_config(d_model=self.sampled_d_model,
                                         q_dim = self.sampled_n_heads*32,
                                         transformer_nlevel=self.sampled_n_levels,
                                         transformer_enc_num_points=self.sampled_n_points,
                                         transformer_nheads=self.sampled_n_heads,)
        self.dropout1.p = self.dropout * self.sampled_d_model / self.d_model
        self.norm1.set_sample_config(self.sampled_d_model)
        self.linear1.set_sample_config(self.sampled_d_model, self.sampled_d_ffn)
        self.dropout2.p = self.dropout * self.sampled_d_model / self.d_model
        self.linear2.set_sample_config(self.sampled_d_ffn, self.sampled_d_model)
        self.dropout3.p = self.dropout * self.sampled_d_model / self.d_model
        self.norm2.set_sample_config(self.sampled_d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.pre_norm:
            return layer_norm(x)
        else:
            return x

    def forward_ffn(self, src):
        residual = src
        src = self.maybe_layer_norm(self.norm2, src, before=True)
        src = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = residual + self.dropout3(src)
        src = self.maybe_layer_norm(self.norm2, src, after=True)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        if self.is_identity_layer:
            return src

        # self attention
        residual = src
        src = self.maybe_layer_norm(self.norm1, src, before=True)
        src = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = residual + self.dropout1(src)
        src = self.maybe_layer_norm(self.norm1, src, after=True)

        # ffn
        src = self.forward_ffn(src)

        return src

    def params(self):
        if self.is_identity_layer:
            return 0
        params = 0
        params += self.self_attn.params()
        params += self.norm1.params()
        params += self.linear1.params()
        params += self.linear2.params()
        params += self.norm2.params()
        return params

    def flops(self, sequence_length):
        if self.is_identity_layer:
            return 0
        flops = 0
        flops += self.self_attn.flops(sequence_length)
        flops += self.norm1.flops(sequence_length)
        flops += self.linear1.flops(sequence_length)
        flops += self.linear2.flops(sequence_length)
        flops += self.norm2.flops(sequence_length)
        return flops


class SuperMSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, use_checkpoint):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.use_checkpoint = use_checkpoint

        self.sampled_transformer_enc_layers = None

    def set_sample_config(self, d_model=None, transformer_nlevels=None,transformer_enc_layers=None, transformer_enc_num_points=None, transformer_nheads=None, transformer_mlp_ratio=None):
        self.sampled_transformer_enc_layers = transformer_enc_layers
        for i, layer in enumerate(self.layers):
            if i < self.sampled_transformer_enc_layers:
                layer.set_sample_config(is_identity_layer=False,
                                        d_model=d_model,
                                        transformer_nlevels=transformer_nlevels,
                                        transformer_enc_num_points=transformer_enc_num_points[i],
                                        transformer_nheads=transformer_nheads[i],
                                        transformer_mlp_ratio=transformer_mlp_ratio[i])
            # exceeds sample layer number
            else:
                layer.set_sample_config(is_identity_layer=True)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            if self.use_checkpoint:
                output = checkpoint.checkpoint(layer, output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)
            else:
                output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output

    def params(self):
        params = 0
        for i, layer in enumerate(self.layers):
            params += layer.params()
        return params

    def flops(self, sequence_length):
        flops = 0
        for i, layer in enumerate(self.layers):
            flops += layer.flops(sequence_length)
        return flops


# Super MSDeformAttn Transformer encoder in deformable detr
class SuperMSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, mlp_ratio=4.0, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
                 pre_norm=True, norm="LN", scale=False, use_checkpoint=False,
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.num_feature_levels = num_feature_levels

        encoder_layer = SuperMSDeformAttnTransformerEncoderLayer(d_model, mlp_ratio,
                                                             dropout, activation,
                                                             num_feature_levels, nhead, enc_n_points,
                                                             pre_norm=pre_norm,
                                                             norm=norm, scale=scale)
        self.encoder = SuperMSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers, use_checkpoint=use_checkpoint)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

        self.scale = scale
        # sampled
        self.sampled_d_model = None
        self.sampled_num_feature_levels = None
        self.sampled_transformer_enc_layers = None
        self.sampled_transformer_enc_num_points = None
        self.sampled_transformer_nheads = None
        self.sampled_transformer_mlp_ratio = None
        self.sampled_level_embed = None

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, SuperMSDeformAttn):
                m._reset_parameters()
        nn.init.normal_(self.level_embed)

    def set_sample_config(self, d_model=None, transformer_input_features=None, transformer_enc_layers=None, transformer_enc_num_points=None, transformer_nheads=None, transformer_mlp_ratio=None):
        self.sampled_d_model = d_model
        self.sampled_num_feature_levels = len(transformer_input_features)
        self.sampled_transformer_enc_layers = transformer_enc_layers
        self.sampled_transformer_enc_num_points = transformer_enc_num_points
        self.sampled_transformer_nheads = transformer_nheads
        self.sampled_transformer_mlp_ratio = transformer_mlp_ratio
        self.sampled_level_embed = self.level_embed[:self.sampled_num_feature_levels, :self.sampled_d_model]
        self.encoder.set_sample_config(d_model=self.sampled_d_model,
                                       transformer_nlevels=self.sampled_num_feature_levels,
                                       transformer_enc_layers=self.sampled_transformer_enc_layers,
                                       transformer_enc_num_points=self.sampled_transformer_enc_num_points,
                                       transformer_nheads=self.sampled_transformer_nheads,
                                       transformer_mlp_ratio=self.sampled_transformer_mlp_ratio,)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, pos_embeds):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.sampled_level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)

        return memory, spatial_shapes, level_start_index

    def params(self):
        return self.encoder.params() + self.sampled_num_feature_levels * self.sampled_d_model

    def flops(self, sequence_length):
        return self.encoder.flops(sequence_length) + self.sampled_num_feature_levels * self.sampled_d_model / 2.0


@SEM_SEG_HEADS_REGISTRY.register()
class SuperMSDeformAttnPixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_mlp_ratio: int,
        transformer_enc_layers: int,
        conv_dim: int,
        mask_dim: int,
        norm_layer: Optional[Union[str, Callable]] = None,
        transformer_pre_norm=True,
        transformer_act_layer=None,
        transformer_norm_layer=None,
        transformer_enc_n_points=None,
        # deformable transformer encoder args
        transformer_in_features: List[str],
        maskformer_in_features: List[str],
        scale=False,
        use_checkpoint=False,
        common_stride: int,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()
        self.input_shape_dict = input_shape
        transformer_input_shape = {
            k: v for k, v in input_shape.items() if k in transformer_in_features
        }

        # this is the input shape of pixel decoder
        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.input_shape = input_shape
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        self.feature_strides = [v.stride for k, v in input_shape]
        self.feature_channels = [v.channels for k, v in input_shape]
        
        # this is the input shape of transformer encoder (could use less features than pixel decoder
        transformer_input_shape = sorted(transformer_input_shape.items(), key=lambda x: x[1].stride)
        self.transformer_input_shape = transformer_input_shape
        self.transformer_in_features = [k for k, v in transformer_input_shape]  # starting from "res2" to "res5"
        self.transformer_in_channels = [v.channels for k, v in transformer_input_shape]
        self.transformer_feature_strides = [v.stride for k, v in transformer_input_shape]  # to decide extra FPN layers
        self.transformer_pre_norm = transformer_pre_norm

        self.transformer_num_feature_levels = len(self.transformer_in_features)
        self.input_proj = nn.ModuleDict()
        # from low resolution to high resolution (res5 -> res2)
        for k, v in self.transformer_input_shape:
            proj_norm = _get_norm_super(norm_layer, conv_dim)
            self.input_proj[k] = SuperConv2d(v.channels, conv_dim, kernel_size=1, norm=proj_norm, scale=scale)

            nn.init.xavier_uniform_(self.input_proj[k].weight, gain=1)
            nn.init.constant_(self.input_proj[k].bias, 0)

        self.transformer = SuperMSDeformAttnTransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            mlp_ratio=transformer_mlp_ratio,
            num_encoder_layers=transformer_enc_layers,
            num_feature_levels=self.transformer_num_feature_levels,
            activation=transformer_act_layer,
            enc_n_points=transformer_enc_n_points,
            pre_norm=transformer_pre_norm,
            norm=transformer_norm_layer,
            scale=scale,
            use_checkpoint=use_checkpoint
        )
        N_steps = conv_dim // 2
        self.pe_layer = SuperPositionEmbeddingSine(N_steps, normalize=True)

        self.mask_dim = mask_dim
        # use 1x1 conv instead
        self.mask_features = SuperConv2d(
            conv_dim,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            scale=scale
        )
        weight_init.c2_xavier_fill(self.mask_features)
        
        self.maskformer_in_features = maskformer_in_features
        self.maskformer_num_feature_levels = len(maskformer_in_features)
        self.common_stride = common_stride

        # extra fpn levels
        stride = max(self.feature_strides)
        self.num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        self.lateral_convs = nn.ModuleDict()
        self.output_convs = nn.ModuleDict()

        use_bias = norm_layer == ""
        for idx, k in enumerate(self.in_features[:self.num_fpn_levels]):
            in_channels = self.input_shape_dict[k].channels

            lateral_norm = _get_norm_super(norm_layer, conv_dim)
            output_norm = _get_norm_super(norm_layer, conv_dim)

            lateral_conv = SuperConv2d(
                in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm, scale=scale
            )
            output_conv = SuperConv2d(
                conv_dim,
                conv_dim,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
                norm=output_norm,
                activation=F.relu,
                scale=scale
            )
            weight_init.c2_xavier_fill(lateral_conv)
            weight_init.c2_xavier_fill(output_conv)
            # self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            # self.add_module("layer_{}".format(idx + 1), output_conv)

            self.lateral_convs[k] = lateral_conv
            self.output_convs[k] = output_conv
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.

        self.scale = scale
        # sampled
        self.sampled_transformer_in_features = None
        self.sampled_maskformer_in_features = None
        self.sampled_conv_dim = None
        self.sampled_transformer_nheads = None
        self.sampled_transformer_enc_num_points = None
        self.sampled_transformer_mlp_ratio = None
        self.sampled_transformer_enc_layers = None
        self.sampled_mask_dim = None
        self.sampled_transformer_num_feature_levels = None
        self.sampled_num_fpn_levels = None
        self.sampled_maskformer_num_feature_levels = None
        self.sampled_scale = None

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        cls.original_input_shape = input_shape
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm_layer"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        ret["transformer_pre_norm"] = cfg.MODEL.SEM_SEG_HEAD.PRE_NORM
        ret["transformer_act_layer"] = cfg.MODEL.SEM_SEG_HEAD.ACT_LAYER
        ret["transformer_norm_layer"] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_NORM
        ret["transformer_enc_n_points"] = cfg.MODEL.SEM_SEG_HEAD.ENC_N_POINTS
        ret["transformer_dropout"] = cfg.MODEL.SEM_SEG_HEAD.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.SEM_SEG_HEAD.ENC_N_HEADS
        ret["transformer_mlp_ratio"] = cfg.MODEL.SEM_SEG_HEAD.ENC_MLP_RATIO
        ret["transformer_enc_layers"] = cfg.MODEL.SEM_SEG_HEAD.ENC_DEPTHS
        ret["transformer_in_features"] = sorted(cfg.MODEL.SEM_SEG_HEAD.ENC_IN_FEATURES, key=lambda x: int(x[3:]))
        ret["maskformer_in_features"] = sorted(cfg.MODEL.MASK_FORMER.DEC_IN_FEATURES, key=lambda x: int(x[3:]))
        ret["scale"] = cfg.MODEL.SEM_SEG_HEAD.SCALE
        ret["use_checkpoint"] = cfg.MODEL.SEM_SEG_HEAD.USE_CHECKPOINT
        ret["common_stride"] = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        return ret

    def set_sample_config(self, config: dict):
        self.sampled_transformer_in_features = sorted(config.SEM_SEG_HEAD.ENC_IN_FEATURES, key=lambda x: int(x[3:]))
        self.sampled_maskformer_in_features = sorted(config.MASK_FORMER.DEC_IN_FEATURES, key=lambda x: int(x[3:]))
        self.sampled_conv_dim = config.SEM_SEG_HEAD.CONVS_DIM
        self.sampled_transformer_nheads = config.SEM_SEG_HEAD.ENC_N_HEADS
        self.sampled_transformer_enc_num_points = config.SEM_SEG_HEAD.ENC_N_POINTS
        self.sampled_transformer_mlp_ratio = config.SEM_SEG_HEAD.ENC_MLP_RATIO
        self.sampled_transformer_enc_layers = config.SEM_SEG_HEAD.ENC_DEPTHS
        self.sampled_mask_dim = config.SEM_SEG_HEAD.MASK_DIM
        sampled_transformer_feature_strides = [v.stride for k, v in self.input_shape if k in self.sampled_transformer_in_features]
        sampled_transformer_in_channels = {'res2': config.BACKBONE.EMBED_DIM, 'res3': config.BACKBONE.EMBED_DIM*2, 'res4': config.BACKBONE.EMBED_DIM*4, 'res5': config.BACKBONE.EMBED_DIM*8}

        self.sampled_transformer_num_feature_levels = len(self.sampled_transformer_in_features)
        for k, proj in self.input_proj.items():
            if k in self.sampled_transformer_in_features:
                proj.set_sample_config(sampled_transformer_in_channels[k], self.sampled_conv_dim)

        self.transformer.set_sample_config(d_model=self.sampled_conv_dim, transformer_input_features=self.sampled_transformer_in_features, transformer_enc_layers=self.sampled_transformer_enc_layers, transformer_enc_num_points=self.sampled_transformer_enc_num_points, transformer_nheads=self.sampled_transformer_nheads, transformer_mlp_ratio=self.sampled_transformer_mlp_ratio)

        sampled_N_steps = self.sampled_conv_dim // 2
        self.pe_layer.set_sample_config(sampled_N_steps)

        self.mask_features.set_sample_config(self.sampled_conv_dim, self.sampled_mask_dim)

        self.maskformer_num_feature_levels = len(self.sampled_maskformer_in_features)

        stride = min(sampled_transformer_feature_strides)
        self.sampled_num_fpn_levels = int(np.log2(stride) - np.log2(self.common_stride))

        for k, conv in self.lateral_convs.items():
            if k in self.in_features[:self.sampled_num_fpn_levels]:
                conv.set_sample_config(sampled_transformer_in_channels[k], self.sampled_conv_dim)

        for k, conv in self.output_convs.items():
            if k in self.in_features[:self.sampled_num_fpn_levels]:
                conv.set_sample_config(self.sampled_conv_dim, self.sampled_conv_dim)

    @autocast(enabled=False)
    def forward_features(self, features):
        srcs = []
        pos = []
        # Reverse feature maps into top-down order (from low to high resolution)
        for k in self.sampled_transformer_in_features[::-1]:
            x = features[k].float()  # deformable detr does not support half precision
            srcs.append(self.input_proj[k](x))
            pos.append(self.pe_layer(x))

        y, spatial_shapes, level_start_index = self.transformer(srcs, pos)
        bs = y.shape[0]

        split_size_or_sections = [None] * self.sampled_transformer_num_feature_levels
        for i in range(self.sampled_transformer_num_feature_levels):
            if i < self.sampled_transformer_num_feature_levels - 1:
                split_size_or_sections[i] = level_start_index[i + 1] - level_start_index[i]
            else:
                split_size_or_sections[i] = y.shape[1] - level_start_index[i]
        y = torch.split(y, split_size_or_sections, dim=1)

        out = {}
        multi_scale_features = {}
        for i, z in enumerate(y):
            out[self.sampled_transformer_in_features[::-1][i]] = z.transpose(1, 2).view(bs, -1, spatial_shapes[i][0], spatial_shapes[i][1])

        # append `out` with extra FPN levels
        # Reverse feature maps into top-down order (from low to high resolution)
        for k in self.in_features[:self.sampled_num_fpn_levels][::-1]:
            x = features[k].float()
            lateral_conv = self.lateral_convs[k]
            output_conv = self.output_convs[k]
            cur_fpn = lateral_conv(x)
            # Following FPN implementation, we use nearest upsampling here
            ind = sorted(out.keys(), key=lambda x: int(x[3:]))[0] # use the image with the higner resolution
            y = cur_fpn + F.interpolate(out[ind], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = output_conv(y)
            out[k] = y

        for k in out.keys():
            if k in self.sampled_maskformer_in_features:
                multi_scale_features[k] = out[k]

        sorted_key_list = sorted(out.keys(), key=lambda x: int(x[3:]), reverse=True) # res5 -> res2

        return self.mask_features(out[sorted_key_list[-1]]), out[sorted_key_list[0]], multi_scale_features

    def params(self):
        params = 0
        for k, proj in self.input_proj.items():
            if k in self.sampled_transformer_in_features:
                params += proj.params()
        params += self.pe_layer.params()
        params += self.transformer.params()
        for k in self.in_features[:self.sampled_num_fpn_levels]:
            params += self.lateral_convs[k].params()
            params += self.output_convs[k].params()
        params += self.mask_features.params()
        return params

    def flops(self, H, W, shapes):
        flops = 0
        for k, proj in self.input_proj.items():
            if k in self.sampled_transformer_in_features:
                flops += proj.flops((H // shapes[k].stride), (W // shapes[k].stride))
                flops += self.pe_layer.flops((H // shapes[k].stride) * (W // shapes[k].stride))
        sequence_length = 0
        for k in self.sampled_transformer_in_features:
            sequence_length += (H // shapes[k].stride) * (W // shapes[k].stride)
        flops += self.transformer.flops(sequence_length)
        for k in self.in_features[:self.sampled_num_fpn_levels]:
            flops += self.lateral_convs[k].flops((H // shapes[k].stride), (W // shapes[k].stride))
            flops += self.output_convs[k].flops((H // shapes[k].stride), (W // shapes[k].stride))
        k = self.in_features[0]
        flops += self.mask_features.flops((H // shapes[k].stride), (W // shapes[k].stride))
        return flops
