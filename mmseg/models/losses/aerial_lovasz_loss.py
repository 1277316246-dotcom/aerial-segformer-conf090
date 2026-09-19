"""FP32 adapter for MMSegmentation's Lovasz-Softmax implementation.

The loss formula is the upstream lovasz_softmax_flat implementation. This
adapter preserves the pixel dimension when exactly one pixel is valid and
returns a differentiable scalar zero when no pixels are valid. No upstream
file or CE implementation is modified.
"""

from contextlib import nullcontext

import torch
from torch import nn

from mmseg.registry import MODELS
from .lovasz_loss import lovasz_softmax_flat


@MODELS.register_module()
class AerialLovaszLoss(nn.Module):
    """Batch-wise, present-class Lovasz-Softmax, evaluated in FP32.

Only the settings used by this experiment are accepted. Pixel samplers,
additional pixel weights, and external averaging factors are not supported.
"""

    def __init__(self, loss_type='multi_class', classes='present',
                 per_image=False, reduction='none', loss_weight=0.5,
                 loss_name='loss_lovasz'):
        super().__init__()
        if (loss_type != 'multi_class' or classes != 'present' or per_image
                or reduction != 'none'):
            raise ValueError('Expected multi_class/present/per_image=False/reduction=none')
        if not 0 < float(loss_weight) < float('inf'):
            raise ValueError('loss_weight must be finite and positive')
        if not loss_name.startswith('loss_'):
            raise ValueError('loss_name must start with loss_')
        self.loss_type = loss_type
        self.classes = classes
        self.per_image = per_image
        self.reduction = reduction
        self.loss_weight = float(loss_weight)
        self._loss_name = loss_name

    def forward(self, cls_score, label, weight=None, avg_factor=None,
                reduction_override=None, ignore_index=255, **kwargs):
        if weight is not None or avg_factor is not None:
            raise ValueError('AerialLovaszLoss does not support pixel weights or avg_factor')
        if reduction_override not in (None, 'none'):
            raise ValueError('Batch-wise Lovasz already returns one scalar')
        if (cls_score.ndim != 4 or label.ndim != 3
                or cls_score.shape[0] != label.shape[0]
                or cls_score.shape[2:] != label.shape[1:]):
            raise ValueError('Expected logits NCHW and labels NHW with matching shapes')
        if label.dtype not in (torch.uint8, torch.int8, torch.int16,
                               torch.int32, torch.int64):
            raise ValueError('Labels must be integer training IDs')
        if cls_score.device != label.device or cls_score.shape[1] < 2:
            raise ValueError('Expected multiclass logits and labels on the same device')

        context = (torch.autocast(device_type=cls_score.device.type, enabled=False)
                   if cls_score.device.type in ('cuda', 'cpu') else nullcontext())
        with context:
            scores = cls_score.float()
            if not torch.isfinite(scores).all():
                raise FloatingPointError('Non-finite segmentation logits before Lovasz')
            labels = label.reshape(-1).long()
            valid = labels != ignore_index
            labels = labels[valid]
            if labels.numel() == 0:
                return scores.sum() * 0.0
            if (labels < 0).any() or (labels >= scores.shape[1]).any():
                raise ValueError('Label outside training IDs 0..C-1 and ignore_index')
            probabilities = scores.softmax(dim=1)
            probabilities = probabilities.permute(0, 2, 3, 1).reshape(-1, scores.shape[1])
            probabilities = probabilities[valid]  # Retains shape (1, C) for one pixel.
            loss = lovasz_softmax_flat(probabilities, labels, classes=self.classes)
            return self.loss_weight * loss

    @property
    def loss_name(self):
        return self._loss_nam