# Copyright (c) OpenMMLab. All rights reserved.
"""Generate high-quality pseudo labels for the second self-training stage.

The output PNG uses the raw dataset label IDs: 0 is ignore and 1--8 are
classes.  Therefore it can be loaded directly with ``reduce_zero_label=True``:
valid pixels become training IDs 0--7 and ignored pixels become 255.

Unlike a plain argmax pseudo label, a pixel is retained only when both its
mean TTA confidence and its cross-TTA vote agreement satisfy the requested
thresholds.  Class-specific confidence thresholds should be obtained from
``tools/search_pseudo_thresholds.py`` on the validation set.
"""

from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path
from typing import Sequence

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import revert_sync_batchnorm
from mmengine.utils import mkdir_or_exist
from PIL import Image

from mmseg.apis import inference_model, init_model

try:
    # Used when imported by ``tools/search_pseudo_thresholds.py``.
    from demo.infer_gray_label import (EXPECTED_NUM_CLASSES, IMG_EXTENSIONS,
                                       check_readable_image, collect_images,
                                       scaled_shape)
except ModuleNotFoundError:
    # Used when launched directly: ``python demo/infer_pseudo_label_v2.py``.
    from infer_gray_label import (EXPECTED_NUM_CLASSES, IMG_EXTENSIONS,
                                  check_readable_image, collect_images,
                                  scaled_shape)


CLASS_NAMES = (
    'Background', 'Building', 'Road', 'Water', 'Barren', 'Vegetation',
    'Agricultural', 'Vehicle')


def predict_probabilities_and_votes(model, image: np.ndarray, scales,
                                    rotations, hflip: bool, use_amp: bool):
    """Average TTA probabilities and count the winning class of every view.

    The returned tensors are CPU tensors. ``vote_counts[c, y, x]`` is the
    number of TTA views that predicted class ``c`` for pixel ``(y, x)``.
    """
    height, width = image.shape[:2]
    num_classes = int(model.decode_head.num_classes)
    probability_sum = None
    vote_counts = None
    variant_count = 0
    model_device = next(model.parameters()).device
    amp_enabled = use_amp and model_device.type == 'cuda'

    for scale in scales:
        new_height, new_width = scaled_shape(height, width, scale)
        resized = mmcv.imresize(
            image, (new_width, new_height), interpolation='bilinear')
        for rotation in rotations:
            rotated = np.ascontiguousarray(np.rot90(resized, k=rotation))
            flip_states = (False, True) if hflip else (False, )
            for flipped in flip_states:
                augmented = (np.ascontiguousarray(rotated[:, ::-1])
                             if flipped else rotated)
                amp_context = (
                    torch.autocast(device_type='cuda', dtype=torch.float16)
                    if amp_enabled else nullcontext())
                with amp_context:
                    result = inference_model(model, augmented)
                if not hasattr(result, 'seg_logits'):
                    raise RuntimeError(
                        '模型结果中没有seg_logits，无法进行伪标签TTA。')

                logits = result.seg_logits.data.float()
                if flipped:
                    logits = torch.flip(logits, dims=(2, ))
                if rotation:
                    logits = torch.rot90(logits, k=-rotation, dims=(1, 2))

                probabilities = logits.softmax(dim=0).unsqueeze(0)
                probabilities = F.interpolate(
                    probabilities,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False).squeeze(0).cpu()

                probability_sum = (
                    probabilities if probability_sum is None else
                    probability_sum + probabilities)

                view_prediction = probabilities.argmax(dim=0)
                if vote_counts is None:
                    vote_counts = torch.zeros(
                        (num_classes, height, width), dtype=torch.uint8)
                # Avoid allocating an HxWxC int64 one-hot tensor for every
                # view.  The largest supported number of views is 255.
                for class_id in range(num_classes):
                    vote_counts[class_id].add_(
                        view_prediction.eq(class_id).to(torch.uint8))
                variant_count += 1

    if variant_count == 0 or variant_count > 255:
        raise RuntimeError('TTA组合数量必须位于1到255之间。')
    return probability_sum / variant_count, vote_counts, variant_count


def build_pseudo_label(probabilities: torch.Tensor, vote_counts: torch.Tensor,
                       thresholds: Sequence[float], min_vote_ratio: float):
    """Return raw-ID pseudo label, retain mask, confidence and agreement."""
    if len(thresholds) != EXPECTED_NUM_CLASSES:
        raise ValueError('必须提供8个类别置信度阈值。')
    if any(value <= 0.0 or value > 1.0 for value in thresholds):
        raise ValueError('所有类别阈值必须在(0, 1]范围内。')
    if not 0.0 <= min_vote_ratio <= 1.0:
        raise ValueError('--min-vote-ratio必须在[0, 1]范围内。')

    confidence, prediction = probabilities.max(dim=0)
    winning_votes = vote_counts.gather(0, prediction.unsqueeze(0)).squeeze(0)
    variant_count = int(vote_counts.sum(dim=0).max().item())
    min_votes = max(1, int(np.ceil(variant_count * min_vote_ratio)))

    threshold_tensor = torch.tensor(
        thresholds, dtype=confidence.dtype, device=confidence.device)
    per_pixel_threshold = threshold_tensor[prediction]
    keep = (confidence >= per_pixel_threshold) & (winning_votes >= min_votes)

    # Dataset raw ID 0 means ignore.  ``reduce_zero_label=True`` maps this to
    # 255, while raw IDs 1--8 map back to model IDs 0--7.
    pseudo_label = torch.where(
        keep, prediction.to(torch.uint8) + 1,
        torch.zeros_like(prediction, dtype=torch.uint8))
    agreement = winning_votes.float() / variant_count
    return pseudo_label.numpy(), keep.numpy(), confidence.numpy(), agreement.numpy(), min_votes


def save_pseudo_label(label: np.ndarray, out_path: Path, expected_size: int):
    """Save a single-channel raw-ID pseudo label and verify it."""
    if label.ndim != 2:
        raise ValueError(f'伪标签必须为HxW，实际为{label.shape}')
    if expected_size > 0 and label.shape != (expected_size, expected_size):
        raise ValueError(
            f'伪标签尺寸为{label.shape[1]}x{label.shape[0]}，'
            f'要求为{expected_size}x{expected_size}')
    valid_ids = set(np.unique(label).tolist())
    if not valid_ids.issubset(set(range(9))):
        raise ValueError(f'伪标签只能包含0-8，实际为{sorted(valid_ids)}')

    Image.fromarray(label, mode='L').save(out_path, format='PNG')
    with Image.open(out_path) as image:
        if image.mode != 'L' or image.format != 'PNG':
            raise ValueError(f'伪标签保存格式异常: {out_path}')
        if expected_size > 0 and image.size != (expected_size, expected_size):
            raise ValueError(f'伪标签保存尺寸异常: {out_path}')


def parse_args():
    parser = ArgumentParser(description='生成第二轮自训练的稳定伪标签')
    parser.add_argument('img', help='测试图片目录或单张图片')
    parser.add_argument('config', help='教师模型配置文件')
    parser.add_argument('checkpoint', help='教师模型权重文件')
    parser.add_argument('--out-dir', default='outputs/pseudo_labels_v2')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--recursive', action='store_true')
    parser.add_argument('--expected-size', type=int, default=1024)
    parser.add_argument(
        '--scales', type=float, nargs='+', default=[0.75, 1.0, 1.25])
    parser.add_argument('--hflip', action='store_true')
    parser.add_argument(
        '--rotations', type=int, nargs='+', default=[0], choices=[0, 1, 2, 3])
    parser.add_argument(
        '--thresholds', type=float, nargs=EXPECTED_NUM_CLASSES,
        default=[0.90] * EXPECTED_NUM_CLASSES,
        metavar=('BG', 'BLD', 'ROAD', 'WATER', 'BARREN', 'VEG', 'AGRI', 'VEH'),
        help='按训练类别0-7排列的8个置信度阈值')
    parser.add_argument(
        '--min-vote-ratio', type=float, default=0.833333,
        help='获胜类别在所有TTA视图中的最低票数比例；6视图时约为5票')
    parser.add_argument(
        '--amp', action='store_true',
        help='CUDA下使用FP16；阈值附近的像素建议默认不用')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.expected_size < 0:
        raise ValueError('--expected-size不能为负数。')

    out_dir = Path(args.out_dir)
    mkdir_or_exist(str(out_dir))
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device.startswith('cpu'):
        model = revert_sync_batchnorm(model)
    if int(model.decode_head.num_classes) != EXPECTED_NUM_CLASSES:
        raise ValueError('当前脚本仅支持本赛题的8类别模型。')

    images = collect_images(args.img, recursive=args.recursive)
    variant_count = len(args.scales) * len(args.rotations) * (2 if args.hflip else 1)
    print(f'TTA组合数: {variant_count}; 阈值: {args.thresholds}; '
          f'最小投票比例: {args.min_vote_ratio:.3f}')

    retained_pixels = 0
    total_pixels = 0
    class_pixels = np.zeros(EXPECTED_NUM_CLASSES, dtype=np.int64)
    confidence_sum = 0.0
    agreement_sum = 0.0
    failures = 0

    for index, image_path in enumerate(images, start=1):
        try:
            check_readable_image(image_path, args.expected_size)
            image = mmcv.imread(str(image_path), channel_order='bgr')
            if image is None:
                raise ValueError('MMCV无法读取该图片。')
            probabilities, votes, _ = predict_probabilities_and_votes(
                model, image, args.scales, args.rotations, args.hflip, args.amp)
            label, keep, confidence, agreement, min_votes = build_pseudo_label(
                probabilities, votes, args.thresholds, args.min_vote_ratio)
            save_pseudo_label(label, out_dir / f'{image_path.stem}.png',
                              args.expected_size)

            retained_pixels += int(keep.sum())
            total_pixels += int(keep.size)
            confidence_sum += float(confidence.sum())
            agreement_sum += float(agreement.sum())
            class_pixels += np.bincount(
                label[keep].astype(np.int64) - 1,
                minlength=EXPECTED_NUM_CLASSES)
            print(f'[OK {index}/{len(images)}] {image_path.name}: '
                  f'保留={keep.mean():.1%}, '
                  f'平均置信度={confidence.mean():.3f}, '
                  f'平均一致性={agreement.mean():.3f}, '
                  f'最少票数={min_votes}')
        except Exception as exc:
            failures += 1
            print(f'[FAIL] {image_path}: {exc}')

    if failures:
        raise SystemExit(f'有{failures}张图片生成失败，已停止后续训练。')
    if total_pixels == 0:
        raise SystemExit('没有生成有效像素。')

    print(f'\n完成: {len(images)}张伪标签')
    print(f'总体保留像素占比: {retained_pixels / total_pixels:.2%}')
    print(f'总体平均最大概率: {confidence_sum / total_pixels:.4f}')
    print(f'总体平均TTA一致性: {agreement_sum / total_pixels:.4f}')
    print('保留像素的类别分布:')
    for class_name, count in zip(CLASS_NAMES, class_pixels):
        ratio = count / max(retained_pixels, 1)
        print(f'  {class_name:<12} {count:>10d} ({ratio:.2%})')


if __name__ == '__main__':
    main()