# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu, Yutong Lin, Yixuan Wei
# --------------------------------------------------------

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation/blob/main/mmseg/models/backbones/swin_transformer.py

import torch
import torch.nn as nn
import timm

from detectron2.modeling import BACKBONE_REGISTRY, Backbone, ShapeSpec


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


@BACKBONE_REGISTRY.register()
class VisionTransformer(Backbone):
    def __init__(self, cfg, input_shape):

        model_name = cfg.MODEL.BACKBONE.MODEL_NAME
        pretrained = cfg.MODEL.BACKBONE.PRETRAINED

        super().__init__()

        self.vit = timm.create_model(model_name, pretrained=pretrained)
        embed_dim = self.vit.embed_dim

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            Norm2d(embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )

        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
        )

        self.fpn3 = nn.Identity()

        self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self._out_features = cfg.MODEL.VIT.OUT_FEATURES

        self._out_feature_strides = {
            "res2": 4,
            "res3": 8,
            "res4": 16,
            "res5": 32,
        }
        self._out_feature_channels = {
            "res2": embed_dim,
            "res3": embed_dim,
            "res4": embed_dim,
            "res5": embed_dim,
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
        ), f"SwinTransformer takes an input of shape (N, C, H, W). Got {x.shape} instead!"
        N, C, H, W = x.size()
        Hp, Wp = H // self.vit.patch_embed.proj.stride[0], W // self.vit.patch_embed.proj.stride[1]
        outputs = {}
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(x.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        if self.vit.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.vit.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.vit.pos_drop(x + self.vit.pos_embed)
        x = self.vit.blocks(x)
        x = self.vit.norm(x)
        if self.dist_token is None:
            y = x[:, 1:]
        else:
            y = x[:, 2:]
        # FPN
        y = y.permute(0, 2, 1).reshape(N, -1, Hp, Wp)
        ops = {"res2": self.fpn1, "res3": self.fpn2, "res4": self.fpn3, "res5": self.fpn4}
        for k in ops.keys():
            if k in self._out_features:
                outputs[k] = ops[k](y)
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
