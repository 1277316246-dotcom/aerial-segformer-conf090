# Copyright (c) OpenMMLab. All rights reserved.
"""Scale-aware SegFormer decode head for aerial semantic segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from ..utils import resize
from .segformer_head import SegformerHead


class ChannelAttention(nn.Module):
    """CBAM-style channel attention initialized as an identity mapping."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden_channels = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        avg_descriptor = F.adaptive_avg_pool2d(x, output_size=1)
        max_descriptor = F.adaptive_max_pool2d(x, output_size=1)
        # Multiplication by two makes the zero-initialized gate equal to one.
        weight = 2.0 * torch.sigmoid(
            self.mlp(avg_descriptor) + self.mlp(max_descriptor))
        return x * weight


class SpatialAttention(nn.Module):
    """CBAM-style spatial attention initialized as an identity mapping."""

    def __init__(self, kernel_size=7):
        super().__init__()
        if kernel_size not in (3, 7):
            raise ValueError('spatial_kernel_size must be 3 or 7')
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=True)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True).values
        weight = 2.0 * torch.sigmoid(
            self.conv(torch.cat([avg_map, max_map], dim=1)))
        return x * weight


@MODELS.register_module()
class AttentionSegformerHead(SegformerHead):
    """SegFormer head with source-aware scale and CBAM attention.

    The standard SegFormer projections are retained so that checkpoints from
    ``SegformerHead`` can initialize most parameters.  Each projected scale is
    pooled separately, the four descriptors are concatenated, and a shared
    MLP predicts a per-scale, per-channel softmax weight.  Channel and spatial
    attention are then applied after the original fusion convolution.

    New attention layers are initialized to identity, preventing a randomly
    initialized gate from destroying a converged SegFormer representation at
    the beginning of fine-tuning.
    """

    def __init__(self,
                 attention_reduction=16,
                 spatial_kernel_size=7,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_scales = len(self.in_channels)
        descriptor_channels = self.channels * self.num_scales
        hidden_channels = max(descriptor_channels // attention_reduction, 16)

        self.scale_reduce = nn.Sequential(
            nn.Conv2d(
                descriptor_channels, hidden_channels,
                kernel_size=1, bias=False),
            nn.ReLU(inplace=True))
        self.scale_expand = nn.Conv2d(
            hidden_channels, descriptor_channels,
            kernel_size=1, bias=True)
        # Zero logits -> uniform softmax. Multiplying by num_scales below then
        # gives every scale an initial weight of exactly one.
        nn.init.zeros_(self.scale_expand.weight)
        nn.init.zeros_(self.scale_expand.bias)

        self.channel_attention = ChannelAttention(
            self.channels, reduction=attention_reduction)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, inputs):
        inputs = self._transform_inputs(inputs)
        projected_features = []
        target_size = inputs[0].shape[2:]
        for feature, projection in zip(inputs, self.convs):
            projected_features.append(
                resize(
                    input=projection(feature),
                    size=target_size,
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        descriptors = [
            F.adaptive_avg_pool2d(feature, output_size=1)
            for feature in projected_features
        ]
        descriptor = torch.cat(descriptors, dim=1)
        scale_logits = self.scale_expand(self.scale_reduce(descriptor))
        batch_size = scale_logits.shape[0]
        scale_logits = scale_logits.reshape(
            batch_size, self.num_scales, self.channels, 1, 1)
        scale_weights = (
            torch.softmax(scale_logits, dim=1) * self.num_scales)

        weighted_features = [
            feature * scale_weights[:, index]
            for index, feature in enumerate(projected_features)
        ]
        output = self.fusion_conv(torch.cat(weighted_features, dim=1))
        output = self.channel_attention(output)
        output = self.spatial_attention(output)
        return self.cls_seg(output)
