# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..backbone.vit import Norm2d

# This is a modified FPN decoder with extra Transformer encoder that processes the lowest-resolution feature map.
@SEM_SEG_HEADS_REGISTRY.register()
class FPN(nn.Module):
    @configurable
    def __init__(self,
                 input_shape,
                 out_channels,
                 mask_dim,
                 num_outs,
                 *,
                 no_norm_on_lateral=False,
                 upsample_cfg=dict(mode='nearest'),
    ):
        r"""Feature Pyramid Network.

        This is an implementation of paper `Feature Pyramid Networks for Object
        Detection <https://arxiv.org/abs/1612.03144>`_.

        Args:
            in_channels (List[int]): Number of input channels per scale.
            out_channels (int): Number of output channels (used at each scale)
            num_outs (int): Number of output scales.
            start_level (int): Index of the start input backbone level used to
                build the feature pyramid. Default: 0.
            end_level (int): Index of the end input backbone level (exclusive) to
                build the feature pyramid. Default: -1, which means the last level.
            add_extra_convs (bool | str): If bool, it decides whether to add conv
                layers on top of the original feature maps. Default to False.
                If True, it is equivalent to `add_extra_convs='on_input'`.
                If str, it specifies the source feature map of the extra convs.
                Only the following options are allowed

                - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
                - 'on_lateral':  Last feature map after lateral convs.
                - 'on_output': The last output feature map after fpn convs.
            relu_before_extra_convs (bool): Whether to apply relu before the extra
                conv. Default: False.
            no_norm_on_lateral (bool): Whether to apply norm on lateral.
                Default: False.
            conv_cfg (dict): Config dict for convolution layer. Default: None.
            norm_cfg (dict): Config dict for normalization layer. Default: None.
            act_cfg (str): Config dict for activation layer in ConvModule.
                Default: None.
            upsample_cfg (dict): Config dict for interpolate layer.
                Default: `dict(mode='nearest')`
        """
        super(FPN, self).__init__()
        self.input_shape = input_shape
        self.out_channels = out_channels
        self.mask_dim = mask_dim
        self.num_outs = num_outs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()

        self.lateral_convs = nn.ModuleDict()
        self.fpn_convs = nn.ModuleDict()
        for k in self.input_shape:
            self.lateral_convs[k] = nn.Sequential([
                Conv2d(input_shape[k].channels, out_channels, kernel_size=1, stride=1, padding=0),
                Norm2d(out_channels),
            ])
            self.fpn_convs[k] = nn.Sequential([
                Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
                Norm2d(out_channels),
            ])
        self.mask_features = Conv2d(
            out_channels,
            mask_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["num_outs"] = cfg.MODEL.SEM_SEG_HEAD.NUM_OUTS
        ret["out_channels"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["no_norm_on_lateral"] = cfg.MODEL.SEM_SEG_HEAD.NO_NORM_ON_LATERAL
        ret["upsample_cfg"] = cfg.MODEL.SEM_SEG_HEAD.UPSAMPLE_CFG
        return ret

    @autocast(enabled=False)
    def forward_features(self, inputs):
        """Forward function."""
        out = []
        multi_scale_features = []
        for k in list(self.input_shape.keys())[::-1]:
            x = self.lateral_convs(inputs[k])
            x = self.fpn_convs(x)
            out.append(x)
        multi_scale_features = out[:self.num_outs]
        return self.mask_features(out[-1]), out[0], multi_scale_features
