# Copyright (c) OpenMMLab. All rights reserved.
"""Search class-specific pseudo-label confidence thresholds on ``val.txt``.

The teacher inference, scales, rotations, flip setting and vote rule must be
identical to those used later by ``demo/infer_pseudo_label_v2.py``.  The
script measures per-class precision of pseudo labels on the validation set and
chooses the lowest threshold reaching the requested precision, preserving as
many reliable pseudo pixels as possible.
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import mmcv
import numpy as np
from mmengine.model import revert_sync_batchnorm
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.infer_gray_label import EXPECTED_NUM_CLASSES, check_readable_image
from demo.infer_pseudo_label_v2 import (CLASS_NAMES,
                                        predict_probabilities_and_votes)
from mmseg.apis import init_model


def load_ground_truth(path: Path, expected_size: int):
    """Load raw IDs and convert them to model IDs 0--7 / ignore 255."""
    with Image.open(path) as image:
        raw = np.asarray(image.convert('L'))
    if expected_size > 0 and raw.shape != (expected_size, expected_size):
        raise ValueError(f'标签尺寸异常: {path} ({raw.shape})')
    ground_truth = np.full(raw.shape, 255, dtype=np.uint8)
    valid = (raw >= 1) & (raw <= EXPECTED_NUM_CLASSES)
    ground_truth[valid] = raw[valid] - 1
    return ground_truth


def choose_thresholds(total_bins, correct_bins, gt_class_pixels,
                      candidate_step, min_threshold, max_threshold,
                      target_precision, min_pixels):
    """Choose lowest threshold reaching the precision target for each class."""
    bin_count = total_bins.shape[1]
    candidates = np.arange(
        min_threshold, max_threshold + candidate_step / 2, candidate_step)
    results = []

    for class_id, class_name in enumerate(CLASS_NAMES):
        counts = np.cumsum(total_bins[class_id][::-1])[::-1]
        correct = np.cumsum(correct_bins[class_id][::-1])[::-1]
        entries = []
        for threshold in candidates:
            start_bin = min(
                bin_count - 1,
                max(0, int(np.ceil(threshold * bin_count))))
            selected = int(counts[start_bin])
            true_positive = int(correct[start_bin])
            precision = (true_positive / selected) if selected else 0.0
            entries.append((float(threshold), selected, true_positive, precision))

        qualified = [item for item in entries
                     if item[1] >= min_pixels and item[3] >= target_precision]
        if qualified:
            selected_entry = qualified[0]
            status = 'target_precision_reached'
        else:
            usable = [item for item in entries if item[1] >= min_pixels]
            if usable:
                selected_entry = max(usable, key=lambda item: (item[3], -item[0]))
                status = 'target_not_reached_best_available'
            else:
                selected_entry = entries[-1]
                status = 'insufficient_validation_pixels'

        threshold, selected, true_positive, precision = selected_entry
        results.append(dict(
            class_id=class_id,
            class_name=class_name,
            threshold=threshold,
            selected_pixels=selected,
            correct_pixels=true_positive,
            precision=precision,
            selected_over_gt_pixels=(selected / max(int(gt_class_pixels[class_id]), 1)),
            status=status))
    return results


def parse_args():
    parser = ArgumentParser(description='在验证集搜索分类别伪标签阈值')
    parser.add_argument('config')
    parser.add_argument('checkpoint')
    parser.add_argument('--data-root', default='data/mydata')
    parser.add_argument('--ann-file', default='val.txt')
    parser.add_argument('--img-dir', default='JPEGImages')
    parser.add_argument('--seg-dir', default='SegmentationClass')
    parser.add_argument('--out', default='work_dirs/pseudo_thresholds_v2.json')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--expected-size', type=int, default=1024)
    parser.add_argument(
        '--scales', type=float, nargs='+', default=[0.75, 1.0, 1.25])
    parser.add_argument('--hflip', action='store_true')
    parser.add_argument(
        '--rotations', type=int, nargs='+', default=[0], choices=[0, 1, 2, 3])
    parser.add_argument('--min-vote-ratio', type=float, default=0.833333)
    parser.add_argument('--target-precision', type=float, default=0.92)
    parser.add_argument('--min-pixels', type=int, default=2000)
    parser.add_argument('--candidate-step', type=float, default=0.01)
    parser.add_argument('--min-threshold', type=float, default=0.50)
    parser.add_argument('--max-threshold', type=float, default=0.99)
    parser.add_argument('--amp', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 <= args.min_vote_ratio <= 1.0:
        raise ValueError('--min-vote-ratio必须在[0, 1]范围内。')
    if not 0.0 < args.target_precision <= 1.0:
        raise ValueError('--target-precision必须在(0, 1]范围内。')
    if not 0.0 < args.candidate_step <= 0.1:
        raise ValueError('--candidate-step必须在(0, 0.1]范围内。')
    if not 0.0 <= args.min_threshold <= args.max_threshold <= 1.0:
        raise ValueError('阈值搜索范围不合法。')

    data_root = Path(args.data_root)
    ann_path = data_root / args.ann_file
    image_dir = data_root / args.img_dir
    seg_dir = data_root / args.seg_dir
    names = [line.strip() for line in ann_path.read_text(encoding='utf-8').splitlines()
             if line.strip()]
    if not names:
        raise ValueError(f'验证列表为空: {ann_path}')

    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device.startswith('cpu'):
        model = revert_sync_batchnorm(model)
    if int(model.decode_head.num_classes) != EXPECTED_NUM_CLASSES:
        raise ValueError('当前脚本仅支持8类别比赛模型。')

    bin_count = int(round(1.0 / args.candidate_step))
    if not np.isclose(bin_count * args.candidate_step, 1.0):
        raise ValueError('--candidate-step必须能整除1，例如0.01或0.005。')
    total_bins = np.zeros((EXPECTED_NUM_CLASSES, bin_count), dtype=np.int64)
    correct_bins = np.zeros_like(total_bins)
    gt_class_pixels = np.zeros(EXPECTED_NUM_CLASSES, dtype=np.int64)
    variant_count = len(args.scales) * len(args.rotations) * (2 if args.hflip else 1)
    print(f'验证样本={len(names)}, TTA组合={variant_count}, '
          f'最小投票比例={args.min_vote_ratio:.3f}, '
          f'目标精确率={args.target_precision:.2%}')

    for index, name in enumerate(names, start=1):
        image_path = image_dir / f'{name}.png'
        label_path = seg_dir / f'{name}.png'
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f'缺少验证样本: {image_path} 或 {label_path}')
        check_readable_image(image_path, args.expected_size)
        image = mmcv.imread(str(image_path), channel_order='bgr')
        if image is None:
            raise ValueError(f'无法读取图片: {image_path}')
        gt = load_ground_truth(label_path, args.expected_size)
        probabilities, votes, _ = predict_probabilities_and_votes(
            model, image, args.scales, args.rotations, args.hflip, args.amp)
        confidence, prediction = probabilities.max(dim=0)
        winning_votes = votes.gather(0, prediction.unsqueeze(0)).squeeze(0)

        confidence = confidence.numpy()
        prediction = prediction.numpy().astype(np.int64)
        vote_ok = (winning_votes.numpy() >=
                   max(1, int(np.ceil(variant_count * args.min_vote_ratio))))
        valid = (gt != 255) & vote_ok
        gt_class_pixels += np.bincount(
            gt[gt != 255].astype(np.int64), minlength=EXPECTED_NUM_CLASSES)
        confidence_bin = np.minimum(
            (confidence * bin_count).astype(np.int64), bin_count - 1)
        flattened = prediction * bin_count + confidence_bin

        selected_flat = flattened[valid]
        total_bins += np.bincount(
            selected_flat, minlength=EXPECTED_NUM_CLASSES * bin_count).reshape(
                EXPECTED_NUM_CLASSES, bin_count)
        correct = valid & (prediction == gt)
        correct_bins += np.bincount(
            flattened[correct],
            minlength=EXPECTED_NUM_CLASSES * bin_count).reshape(
                EXPECTED_NUM_CLASSES, bin_count)
        if index % 25 == 0 or index == len(names):
            print(f'已完成 {index}/{len(names)}')

    results = choose_thresholds(
        total_bins, correct_bins, gt_class_pixels, args.candidate_step,
        args.min_threshold, args.max_threshold, args.target_precision,
        args.min_pixels)
    print('\nClass          threshold  precision  selected_pixels  status')
    print('---------------------------------------------------------------')
    for item in results:
        print(f'{item["class_name"]:<14} {item["threshold"]:>8.3f} '
              f'{item["precision"]:>9.2%} {item["selected_pixels"]:>16d} '
              f'{item["status"]}')
    thresholds = [item['threshold'] for item in results]
    print('\n用于生成V2伪标签的阈值:')
    print(' '.join(f'{value:.2f}' for value in thresholds))

    payload = dict(
        config=str(args.config),
        checkpoint=str(args.checkpoint),
        validation_list=str(ann_path),
        tta=dict(scales=args.scales, rotations=args.rotations,
                 hflip=args.hflip, min_vote_ratio=args.min_vote_ratio,
                 variants=variant_count),
        search=dict(target_precision=args.target_precision,
                    min_pixels=args.min_pixels,
                    candidate_step=args.candidate_step,
                    min_threshold=args.min_threshold,
                    max_threshold=args.max_threshold),
        thresholds=thresholds,
        classes=results)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n结果已保存: {out_path}')


if __name__ == '__main__':
    main()