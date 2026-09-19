# Copyright (c) OpenMMLab. All rights reserved.
"""Analyze suspicious pixel annotations with multi-view teacher consensus.

This script never modifies the source annotations.  It compares the raw
competition labels (0=ignore, 1--8=classes) with a teacher model and writes:

1. per-image statistics that support interrupted/resumed runs;
2. class-wise disagreement and confusion statistics;
3. confidence-valued boundary/interior candidate masks;
4. visualizations for the images with the most boundary candidates.

The conservative cleaning stage should normally use boundary candidates only.
Interior candidates are collected for diagnosis and are disabled by default in
``tools/build_noise_clean_labels.py``.
"""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import mmcv
import numpy as np
import torch
from mmengine.model import revert_sync_batchnorm
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.infer_pseudo_label_v2 import (  # noqa: E402
    predict_probabilities_and_votes)
from mmseg.apis import init_model  # noqa: E402


NUM_CLASSES = 8
IGNORE_INDEX = 255
CLASS_NAMES = (
    'Background', 'Building', 'Road', 'Water', 'Barren', 'Vegetation',
    'Agricultural', 'Vehicle')
PALETTE = np.asarray([
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(
        description='使用教师模型和TTA分析真实训练标签中的疑似噪声')
    parser.add_argument('config', help='教师模型配置文件')
    parser.add_argument('checkpoint', help='教师模型checkpoint')
    parser.add_argument('--data-root', default='data/mydata')
    parser.add_argument('--ann-file', default='train.txt')
    parser.add_argument('--image-dir', default='JPEGImages')
    parser.add_argument('--seg-dir', default='SegmentationClass')
    parser.add_argument('--img-suffix', default='.png')
    parser.add_argument('--seg-suffix', default='.png')
    parser.add_argument(
        '--out-dir', default='outputs/noise_analysis_v1',
        help='分析结果目录；不会修改原标签')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--scales', type=float, nargs='+', default=[0.75, 1.0, 1.25])
    parser.add_argument(
        '--rotations', type=int, nargs='+', default=[0],
        choices=[0, 1, 2, 3])
    parser.add_argument('--hflip', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument(
        '--min-vote-ratio', type=float, default=5.0 / 6.0,
        help='预测视图的最低投票一致率，6-TTA推荐5/6')
    parser.add_argument(
        '--boundary-confidence', type=float, default=0.95,
        help='边界分歧像素的最低教师置信度')
    parser.add_argument(
        '--interior-confidence', type=float, default=0.98,
        help='内部分歧像素的最低教师置信度；只用于诊断')
    parser.add_argument(
        '--boundary-radius', type=int, default=2,
        help='原始1024标签上的边界膨胀半径')
    parser.add_argument(
        '--expected-size', type=int, default=1024,
        help='要求的图像和标签边长；0表示不检查')
    parser.add_argument(
        '--max-visualizations', type=int, default=20,
        help='输出疑似噪声比例最高的可视化数量；0表示关闭')
    parser.add_argument(
        '--limit', type=int, default=0,
        help='只处理前N张，用于冒烟测试；0表示处理全部')
    parser.add_argument(
        '--resume', action='store_true',
        help='复用相同参数已经生成的逐图记录和候选掩码')
    return parser.parse_args()


def validate_args(args):
    if not 0.0 < args.boundary_confidence <= 1.0:
        raise ValueError('--boundary-confidence必须在(0, 1]范围内')
    if not 0.0 < args.interior_confidence <= 1.0:
        raise ValueError('--interior-confidence必须在(0, 1]范围内')
    if not 0.0 <= args.min_vote_ratio <= 1.0:
        raise ValueError('--min-vote-ratio必须在[0, 1]范围内')
    if args.boundary_radius < 0:
        raise ValueError('--boundary-radius不能为负数')
    if args.expected_size < 0 or args.limit < 0:
        raise ValueError('--expected-size和--limit不能为负数')
    if args.max_visualizations < 0:
        raise ValueError('--max-visualizations不能为负数')
    if any(scale <= 0 for scale in args.scales):
        raise ValueError('--scales中的数值必须大于0')


def read_split(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'训练列表不存在: {path}')
    names = []
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped:
            names.append(stripped.split()[0])
    if not names:
        raise ValueError(f'训练列表为空: {path}')
    if len(names) != len(set(names)):
        raise ValueError(
            f'{path}中存在重复样本。噪声分析必须使用不重复的train.txt，'
            '不能使用包含重复伪样本的train_pseudo列表。')
    return names


def sample_path(root: Path, directory: str, name: str, suffix: str):
    relative = Path(name)
    if relative.suffix:
        return root / directory / relative
    return root / directory / relative.with_suffix(suffix)


def output_path(root: Path, name: str, suffix: str):
    relative = Path(name)
    if relative.suffix:
        relative = relative.with_suffix(suffix)
    else:
        relative = relative.with_suffix(suffix)
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_raw_label(path: Path, expected_size: int):
    if not path.is_file():
        raise FileNotFoundError(f'标签不存在: {path}')
    with Image.open(path) as image:
        label = np.asarray(image)
    if label.ndim != 2:
        raise ValueError(f'标签必须是单通道图，实际形状为{label.shape}: {path}')
    if expected_size > 0 and label.shape != (expected_size, expected_size):
        raise ValueError(
            f'标签尺寸为{label.shape[1]}x{label.shape[0]}，要求为'
            f'{expected_size}x{expected_size}: {path}')
    values = np.unique(label)
    allowed = (values == 0) | (values == 255) | (
        (values >= 1) & (values <= NUM_CLASSES))
    if not np.all(allowed):
        raise ValueError(
            f'标签只能包含0、1-8或255，实际包含{values.tolist()}: {path}')
    return label.astype(np.uint8, copy=False)


def raw_to_train_ids(raw_label: np.ndarray):
    train_label = np.full(raw_label.shape, IGNORE_INDEX, dtype=np.uint8)
    valid = (raw_label >= 1) & (raw_label <= NUM_CLASSES)
    train_label[valid] = raw_label[valid] - 1
    return train_label


def dilate_boolean(mask: np.ndarray, radius: int):
    result = mask.astype(bool, copy=True)
    for _ in range(radius):
        padded = np.pad(result, 1, mode='constant', constant_values=False)
        expanded = np.zeros_like(result)
        for dy in range(3):
            for dx in range(3):
                expanded |= padded[dy:dy + result.shape[0],
                                   dx:dx + result.shape[1]]
        result = expanded
    return result


def label_boundary(train_label: np.ndarray, radius: int):
    valid = train_label != IGNORE_INDEX
    padded = np.pad(
        train_label, 1, mode='constant', constant_values=IGNORE_INDEX)
    boundary = np.zeros_like(valid)
    height, width = train_label.shape
    for dy in range(3):
        for dx in range(3):
            if dy == 1 and dx == 1:
                continue
            neighbour = padded[dy:dy + height, dx:dx + width]
            boundary |= (
                valid & (neighbour != IGNORE_INDEX) &
                (neighbour != train_label))
    return dilate_boolean(boundary, radius) & valid


def class_counts(mask: np.ndarray, train_label: np.ndarray):
    if not np.any(mask):
        return np.zeros(NUM_CLASSES, dtype=np.int64)
    return np.bincount(
        train_label[mask].astype(np.int64), minlength=NUM_CLASSES
    )[:NUM_CLASSES]


def build_settings(args, variant_count):
    payload = dict(
        config=str(Path(args.config)),
        checkpoint=str(Path(args.checkpoint)),
        data_root=str(Path(args.data_root)),
        ann_file=args.ann_file,
        image_dir=args.image_dir,
        seg_dir=args.seg_dir,
        img_suffix=args.img_suffix,
        seg_suffix=args.seg_suffix,
        scales=[float(value) for value in args.scales],
        rotations=[int(value) for value in args.rotations],
        hflip=bool(args.hflip),
        amp=bool(args.amp),
        variants=int(variant_count),
        min_vote_ratio=float(args.min_vote_ratio),
        boundary_confidence=float(args.boundary_confidence),
        interior_confidence=float(args.interior_confidence),
        boundary_radius=int(args.boundary_radius),
        expected_size=int(args.expected_size))
    encoded = json.dumps(payload, sort_keys=True).encode('utf-8')
    payload['settings_id'] = hashlib.sha256(encoded).hexdigest()[:16]
    return payload


def empty_aggregate():
    return dict(
        images=0,
        valid_pixels=0,
        disagreement_pixels=0,
        high_vote_disagreement_pixels=0,
        boundary_candidate_pixels=0,
        interior_candidate_pixels=0,
        gt_pixels=np.zeros(NUM_CLASSES, dtype=np.int64),
        disagreement=np.zeros(NUM_CLASSES, dtype=np.int64),
        high_vote_disagreement=np.zeros(NUM_CLASSES, dtype=np.int64),
        boundary_pixels=np.zeros(NUM_CLASSES, dtype=np.int64),
        boundary_candidates=np.zeros(NUM_CLASSES, dtype=np.int64),
        interior_candidates=np.zeros(NUM_CLASSES, dtype=np.int64),
        confusion=np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64))


def add_record(aggregate, record):
    aggregate['images'] += 1
    for key in (
            'valid_pixels', 'disagreement_pixels',
            'high_vote_disagreement_pixels', 'boundary_candidate_pixels',
            'interior_candidate_pixels'):
        aggregate[key] += int(record[key])
    for key in (
            'gt_pixels', 'disagreement', 'high_vote_disagreement',
            'boundary_pixels', 'boundary_candidates',
            'interior_candidates'):
        aggregate[key] += np.asarray(record[key], dtype=np.int64)
    aggregate['confusion'] += np.asarray(
        record['confusion'], dtype=np.int64).reshape(NUM_CLASSES, NUM_CLASSES)


def safe_rate(numerator, denominator):
    return float(numerator / denominator) if denominator else 0.0


def analyze_one(model, image, train_label, args, variant_count):
    probabilities, vote_counts, returned_count = (
        predict_probabilities_and_votes(
            model, image, args.scales, args.rotations, args.hflip, args.amp))
    if returned_count != variant_count:
        raise RuntimeError(
            f'TTA组合数不一致: 预期{variant_count}，实际{returned_count}')

    confidence, prediction = probabilities.max(dim=0)
    winning_votes = vote_counts.gather(
        0, prediction.unsqueeze(0)).squeeze(0)
    agreement = winning_votes.float() / float(variant_count)

    confidence_np = confidence.numpy()
    prediction_np = prediction.numpy().astype(np.uint8)
    agreement_np = agreement.numpy()
    valid = train_label != IGNORE_INDEX
    disagreement = valid & (prediction_np != train_label)
    high_vote_disagreement = disagreement & (
        agreement_np >= args.min_vote_ratio)
    boundary = label_boundary(train_label, args.boundary_radius)
    boundary_candidate = (
        high_vote_disagreement & boundary &
        (confidence_np >= args.boundary_confidence))
    interior_candidate = (
        high_vote_disagreement & (~boundary) &
        (confidence_np >= args.interior_confidence))

    confidence_u8 = np.clip(
        np.rint(confidence_np * 255.0), 1, 255).astype(np.uint8)
    boundary_scores = np.where(
        boundary_candidate, confidence_u8, 0).astype(np.uint8)
    interior_scores = np.where(
        interior_candidate, confidence_u8, 0).astype(np.uint8)

    valid_gt = train_label[valid].astype(np.int64)
    valid_prediction = prediction_np[valid].astype(np.int64)
    confusion = np.bincount(
        valid_gt * NUM_CLASSES + valid_prediction,
        minlength=NUM_CLASSES * NUM_CLASSES).reshape(
            NUM_CLASSES, NUM_CLASSES)

    valid_pixels = int(valid.sum())
    record = dict(
        valid_pixels=valid_pixels,
        disagreement_pixels=int(disagreement.sum()),
        high_vote_disagreement_pixels=int(high_vote_disagreement.sum()),
        boundary_candidate_pixels=int(boundary_candidate.sum()),
        interior_candidate_pixels=int(interior_candidate.sum()),
        disagreement_fraction=safe_rate(disagreement.sum(), valid_pixels),
        boundary_candidate_fraction=safe_rate(
            boundary_candidate.sum(), valid_pixels),
        interior_candidate_fraction=safe_rate(
            interior_candidate.sum(), valid_pixels),
        gt_pixels=class_counts(valid, train_label).tolist(),
        disagreement=class_counts(disagreement, train_label).tolist(),
        high_vote_disagreement=class_counts(
            high_vote_disagreement, train_label).tolist(),
        boundary_pixels=class_counts(boundary, train_label).tolist(),
        boundary_candidates=class_counts(
            boundary_candidate, train_label).tolist(),
        interior_candidates=class_counts(
            interior_candidate, train_label).tolist(),
        confusion=confusion.reshape(-1).tolist())
    return record, boundary_scores, interior_scores, prediction_np


def colorize_train_label(train_label: np.ndarray):
    color = np.zeros((*train_label.shape, 3), dtype=np.uint8)
    valid = train_label != IGNORE_INDEX
    color[valid] = PALETTE[train_label[valid]]
    return color


def make_visualization(image_bgr, train_label, prediction, candidate_scores,
                       output_path: Path):
    image_rgb = image_bgr[..., ::-1]
    gt_color = colorize_train_label(train_label)
    pred_color = PALETTE[prediction]
    overlay = image_rgb.astype(np.float32)
    candidate = candidate_scores > 0
    overlay[candidate] = (
        0.30 * overlay[candidate] +
        0.70 * np.asarray([255, 0, 0], dtype=np.float32))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    mask_panel = np.zeros_like(image_rgb)
    mask_panel[candidate] = np.asarray([255, 255, 255], dtype=np.uint8)

    panels = []
    for panel in (image_rgb, gt_color, pred_color, overlay, mask_panel):
        panels.append(mmcv.imresize(
            panel, (320, 320), interpolation='nearest'))
    canvas = np.concatenate(panels, axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas).save(output_path, format='JPEG', quality=92)


def write_outputs(out_dir: Path, settings, aggregate, records):
    classes = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        gt_pixels = int(aggregate['gt_pixels'][class_id])
        boundary_pixels = int(aggregate['boundary_pixels'][class_id])
        classes.append(dict(
            class_id=class_id,
            class_name=class_name,
            gt_pixels=gt_pixels,
            disagreement_pixels=int(aggregate['disagreement'][class_id]),
            disagreement_rate=safe_rate(
                aggregate['disagreement'][class_id], gt_pixels),
            high_vote_disagreement_pixels=int(
                aggregate['high_vote_disagreement'][class_id]),
            high_vote_disagreement_rate=safe_rate(
                aggregate['high_vote_disagreement'][class_id], gt_pixels),
            boundary_pixels=boundary_pixels,
            boundary_candidate_pixels=int(
                aggregate['boundary_candidates'][class_id]),
            boundary_candidate_rate=safe_rate(
                aggregate['boundary_candidates'][class_id], gt_pixels),
            boundary_candidate_rate_within_boundary=safe_rate(
                aggregate['boundary_candidates'][class_id], boundary_pixels),
            interior_candidate_pixels=int(
                aggregate['interior_candidates'][class_id]),
            interior_candidate_rate=safe_rate(
                aggregate['interior_candidates'][class_id], gt_pixels)))

    totals = dict(
        images=int(aggregate['images']),
        valid_pixels=int(aggregate['valid_pixels']),
        disagreement_pixels=int(aggregate['disagreement_pixels']),
        disagreement_rate=safe_rate(
            aggregate['disagreement_pixels'], aggregate['valid_pixels']),
        high_vote_disagreement_pixels=int(
            aggregate['high_vote_disagreement_pixels']),
        high_vote_disagreement_rate=safe_rate(
            aggregate['high_vote_disagreement_pixels'],
            aggregate['valid_pixels']),
        boundary_candidate_pixels=int(
            aggregate['boundary_candidate_pixels']),
        boundary_candidate_rate=safe_rate(
            aggregate['boundary_candidate_pixels'],
            aggregate['valid_pixels']),
        interior_candidate_pixels=int(
            aggregate['interior_candidate_pixels']),
        interior_candidate_rate=safe_rate(
            aggregate['interior_candidate_pixels'],
            aggregate['valid_pixels']))

    summary = dict(
        settings=settings,
        totals=totals,
        classes=classes,
        confusion_gt_rows_pred_columns=aggregate['confusion'].tolist())
    (out_dir / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    with (out_dir / 'images.csv').open(
            'w', encoding='utf-8-sig', newline='') as file:
        fieldnames = [
            'name', 'valid_pixels', 'disagreement_pixels',
            'disagreement_fraction', 'high_vote_disagreement_pixels',
            'boundary_candidate_pixels', 'boundary_candidate_fraction',
            'interior_candidate_pixels', 'interior_candidate_fraction']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(
                records, key=lambda item: item['boundary_candidate_fraction'],
                reverse=True):
            writer.writerow({key: record[key] for key in fieldnames})

    print('\n类别噪声统计（按人工标签类别统计）')
    print('Class          GT pixels   disagree  boundary cand.  interior cand.')
    print('--------------------------------------------------------------------')
    for item in classes:
        print(
            f'{item["class_name"]:<14} '
            f'{item["gt_pixels"]:>11d} '
            f'{item["disagreement_rate"]:>9.2%} '
            f'{item["boundary_candidate_rate"]:>14.3%} '
            f'{item["interior_candidate_rate"]:>14.3%}')
    print('\n总体统计')
    print(f'  图像数量: {totals["images"]}')
    print(f'  教师与人工标签分歧率: {totals["disagreement_rate"]:.2%}')
    print(
        '  高一致性边界候选占有效像素: '
        f'{totals["boundary_candidate_rate"]:.3%}')
    print(
        '  高一致性内部候选占有效像素: '
        f'{totals["interior_candidate_rate"]:.3%}')
    print(f'  汇总结果: {out_dir / "summary.json"}')
    print(f'  逐图结果: {out_dir / "images.csv"}')


def main():
    args = parse_args()
    validate_args(args)

    data_root = Path(args.data_root)
    ann_path = data_root / args.ann_file
    names = read_split(ann_path)
    if args.limit:
        names = names[:args.limit]

    out_dir = Path(args.out_dir)
    boundary_dir = out_dir / 'boundary_candidates'
    interior_dir = out_dir / 'interior_candidates'
    records_dir = out_dir / 'records'
    if out_dir.exists() and any(out_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f'输出目录非空: {out_dir}。请换一个目录，或确认参数完全相同后'
            '增加--resume。脚本不会自动删除已有结果。')
    boundary_dir.mkdir(parents=True, exist_ok=True)
    interior_dir.mkdir(parents=True, exist_ok=True)
    records_dir.mkdir(parents=True, exist_ok=True)

    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device.startswith('cpu'):
        model = revert_sync_batchnorm(model)
    if int(model.decode_head.num_classes) != NUM_CLASSES:
        raise ValueError(
            f'模型必须是{NUM_CLASSES}类，实际为'
            f'{int(model.decode_head.num_classes)}类')

    variant_count = (
        len(args.scales) * len(args.rotations) * (2 if args.hflip else 1))
    if variant_count <= 0 or variant_count > 255:
        raise ValueError('TTA组合数量必须位于1到255之间')
    settings = build_settings(args, variant_count)
    print(
        f'真实训练样本={len(names)}, TTA组合={variant_count}, '
        f'边界半径={args.boundary_radius}, '
        f'边界阈值={args.boundary_confidence:.3f}, '
        f'内部阈值={args.interior_confidence:.3f}, '
        f'最低投票比例={args.min_vote_ratio:.3f}')
    print('注意：本脚本只分析并输出候选掩码，不修改任何原始标签。')

    aggregate = empty_aggregate()
    all_records = []
    for index, name in enumerate(names, start=1):
        image_path = sample_path(
            data_root, args.image_dir, name, args.img_suffix)
        label_path = sample_path(
            data_root, args.seg_dir, name, args.seg_suffix)
        boundary_path = output_path(boundary_dir, name, '.png')
        interior_path = output_path(interior_dir, name, '.png')
        record_path = output_path(records_dir, name, '.json')

        record = None
        if args.resume and boundary_path.is_file() and \
                interior_path.is_file() and record_path.is_file():
            cached = json.loads(record_path.read_text(encoding='utf-8'))
            if cached.get('settings_id') != settings['settings_id']:
                raise RuntimeError(
                    f'已有记录参数不同，不能续跑: {record_path}。'
                    '请使用新的--out-dir。')
            record = cached

        if record is None:
            if not image_path.is_file():
                raise FileNotFoundError(f'图像不存在: {image_path}')
            image = mmcv.imread(str(image_path), channel_order='bgr')
            if image is None:
                raise ValueError(f'MMCV无法读取图像: {image_path}')
            if args.expected_size > 0 and image.shape[:2] != (
                    args.expected_size, args.expected_size):
                raise ValueError(
                    f'图像尺寸为{image.shape[1]}x{image.shape[0]}，要求为'
                    f'{args.expected_size}x{args.expected_size}: {image_path}')
            raw_label = load_raw_label(label_path, args.expected_size)
            if raw_label.shape != image.shape[:2]:
                raise ValueError(
                    f'图像和标签尺寸不一致: {image_path}, {label_path}')
            train_label = raw_to_train_ids(raw_label)
            record, boundary_scores, interior_scores, _ = analyze_one(
                model, image, train_label, args, variant_count)
            record['name'] = name
            record['settings_id'] = settings['settings_id']
            Image.fromarray(boundary_scores, mode='L').save(boundary_path)
            Image.fromarray(interior_scores, mode='L').save(interior_path)
            record_path.write_text(
                json.dumps(record, ensure_ascii=False), encoding='utf-8')

        add_record(aggregate, record)
        all_records.append(record)
        if index % 25 == 0 or index == len(names):
            print(
                f'已完成 {index}/{len(names)}，当前边界候选占比='
                f'{safe_rate(aggregate["boundary_candidate_pixels"], aggregate["valid_pixels"]):.3%}')

    write_outputs(out_dir, settings, aggregate, all_records)

    visualization_count = min(args.max_visualizations, len(all_records))
    if visualization_count:
        print(f'\n正在为候选比例最高的{visualization_count}张图生成可视化...')
        top_records = sorted(
            all_records,
            key=lambda item: item['boundary_candidate_fraction'],
            reverse=True)[:visualization_count]
        vis_dir = out_dir / 'visualizations'
        for index, record in enumerate(top_records, start=1):
            name = record['name']
            image_path = sample_path(
                data_root, args.image_dir, name, args.img_suffix)
            label_path = sample_path(
                data_root, args.seg_dir, name, args.seg_suffix)
            boundary_path = output_path(boundary_dir, name, '.png')
            image = mmcv.imread(str(image_path), channel_order='bgr')
            train_label = raw_to_train_ids(
                load_raw_label(label_path, args.expected_size))
            probabilities, _, _ = predict_probabilities_and_votes(
                model, image, args.scales, args.rotations,
                args.hflip, args.amp)
            prediction = probabilities.argmax(dim=0).numpy().astype(np.uint8)
            with Image.open(boundary_path) as candidate_image:
                candidate_scores = np.asarray(candidate_image).copy()
            vis_path = output_path(vis_dir, name, '.jpg')
            make_visualization(
                image, train_label, prediction, candidate_scores, vis_path)
            print(f'  可视化 {index}/{visualization_count}: {vis_path}')


if __name__ == '__main__':
    main()