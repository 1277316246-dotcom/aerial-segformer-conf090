"""Preflight B's registered CE/Lovasz losses; does not load model weights."""

import argparse
from contextlib import nullcontext
from pathlib import Path
import sys


def run_checks(cfg, device):
    import torch
    from mmseg.registry import MODELS

    losses = cfg.model.decode_head.loss_decode
    ce_cfg = next(item for item in losses if item.type == 'CrossEntropyLoss')
    lovasz_cfg = next(item for item in losses if item.type == 'AerialLovaszLoss')
    ce, lovasz = (MODELS.build(item).to(device) for item in (ce_cfg, lovasz_cfg))
    native = MODELS.build(dict(type='LovaszLoss', loss_type='multi_class',
                              classes='present', per_image=False, reduction='none',
                              loss_weight=lovasz_cfg.loss_weight)).to(device)
    torch.manual_seed(3407)
    target = torch.randint(0, 8, (2, 8, 8), device=device)
    target[:, :2] = 255
    logits = torch.randn(2, 8, 8, 8, device=device, requires_grad=True)
    value = lovasz(logits, target, ignore_index=255)
    reference = native(logits, target, ignore_index=255)
    if not torch.allclose(value, reference, atol=1e-6, rtol=1e-5):
        raise AssertionError('FP32 result differs from installed native Lovasz')
    grad = torch.autograd.grad(value, logits, retain_graph=True)[0]
    ref_grad = torch.autograd.grad(reference, logits, retain_graph=True)[0]
    if not torch.allclose(grad, ref_grad, atol=1e-6, rtol=1e-5):
        raise AssertionError('Gradient differs from installed native Lovasz')
    if torch.count_nonzero(grad[:, :, :2]).item() != 0:
        raise AssertionError('Ignored pixels have nonzero Lovasz gradients')
    total = ce(logits, target, ignore_index=255) + value
    total.backward()
    if not torch.isfinite(total) or not torch.isfinite(logits.grad).all():
        raise AssertionError('CE + Lovasz loss or gradients are non-finite')
    print('[PASS] CE + Lovasz前向/反向；Lovasz数值与梯度和原生实现一致。')

    for valid_pixels in (0, 1):
        edge_logits = torch.randn(1, 8, 3, 3, device=device, requires_grad=True)
        edge_target = torch.full((1, 3, 3), 255, dtype=torch.long, device=device)
        if valid_pixels:
            edge_target[0, 1, 1] = 4
        loss = lovasz(edge_logits, edge_target, ignore_index=255)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise AssertionError('Edge case did not produce a finite scalar')
        loss.backward()
        if not torch.isfinite(edge_logits.grad).all():
            raise AssertionError('Edge-case gradient is non-finite')
        if not valid_pixels and (loss.item() != 0 or edge_logits.grad.abs().sum().item() != 0):
            raise AssertionError('All-Ignore Lovasz must return zero loss/gradients')
    print('[PASS] Lovasz全Ignore和单有效像素边界情况。')

    half_logits = torch.randn(2, 8, 8, 8, device=device, dtype=torch.float16, requires_grad=True)
    context = (torch.autocast(device_type='cuda', dtype=torch.float16)
               if torch.device(device).type == 'cuda' else nullcontext())
    with context:
        half_loss = lovasz(half_logits, target, ignore_index=255)
    half_loss.backward()
    if (half_loss.dtype != torch.float32 or not torch.isfinite(half_loss)
            or not torch.isfinite(half_logits.grad).all()):
        raise AssertionError('FP16 input was not safely processed in FP32')
    print('[PASS] FP16输入，Lovasz内部FP32计算和有限梯度。')
    print('损失自检完成；未加载checkpoint，未修改数据，未运行完整模型训练。')
    print('注意：全Ignore测试仅针对Lovasz分支，原CE保持不变；完整训练仍需检查loss。')


def main():
    parser = argparse.ArgumentParser(description='B方案损失数值与梯度自检')
    parser.add_argument('config')
    parser.add_argument('--device', default='cpu', help='cpu或cuda:0；不加载模型权重')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from mmengine.config import Config
    from mmseg.utils import register_all_modules
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)
    if cfg.train_dataloader.dataset.data_prefix.seg_map_path != 'SegmentationClass':
        raise ValueError('This preflight expects standalone B, not A+B')
    run_checks(cfg, args.device)


if __name__ == '__main__':
    main()