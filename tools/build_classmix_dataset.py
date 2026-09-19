# Copyright (c) OpenMMLab. All rights reserved.
"""Build an offline DACS-Lite ClassMix dataset.

Each mixed sample combines semantic regions from one labeled source image with
one pseudo-labeled target/test image.  Source and target IDs are inferred from
``train.txt`` and ``train_pseudo.txt``; the 500 IDs present only in the latter
are treated as target-domain pseudo samples.

The builder never modifies the original images or labels.  Mixed images are
stored as high-quality JPEG files to reduce data-disk usage, while mixed labels
remain raw-ID PNG files (0=ignore, 1--8=classes).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


NUM_CLASSES = 8


def parse_args():
    parser = argparse.ArgumentParser(
        description='构建真实标签与测试域伪标签的离线ClassMix数据集')
    parser.add_argument('--data-root', default='data/mydata')
    parser.add_argument('--real-list', default='train.txt')
    parser.add_argument('--combined-list', default='train_pseudo.txt')
    parser.add_argument('--image-dir', default='JPEGImages')
    parser.add_argument('--seg-dir', default='SegmentationClass')
    parser.add_argument('--img-suffix', default='.png')
    parser.add_argument('--seg-suffix', default='.png')
    parser.add_argument('--output-image-dir', default='ClassMixImages')
    parser.add_argument('--output-seg-dir', default='ClassMixLabels')
    parser.add_argument('--output-list', default='classmix_train.txt')
    parser.add_argument('--num-mixes', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument(
        '--min-source-fraction', type=float, default=0.15,
        help='源域语义区域在混合图中的最低像素占比')
    parser.add_argument(
        '--max-source-fraction', type=float, default=0.75,
        help='源域语义区域在混合图中的最高像素占比')
    parser.add_argument(
        '--max-attempts', type=int, default=20,
        help='为每张混合图寻找合适源域类别组合的最大次数')
    parser.add_argument(
        '--resume', action='store_true',
        help='按相同参数复用已经成功生成的混合样本')
    return parser.parse_args()


def validate_args(args):
    if args.num_mixes <= 0:
        raise ValueError('--num-mixes必须为正整数')
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError('--jpeg-quality必须在[1, 100]范围内')
    if not 0.0 <= args.min_source_fraction < args.max_source_fraction <= 1.0:
        raise ValueError(
            '必须满足0 <= min-source-fraction < max-source-fraction <= 1')
    if args.max_attempts <= 0:
        raise ValueError('--max-attempts必须为正整数')


def read_list(path: Path, allow_duplicates: bool):
    if not path.is_file():
        raise FileNotFoundError(f'列表不存在: {path}')
    names = []
    for line in path.read_text(encoding='utf-8').splitlines():
        stripped = line.strip()
        if stripped:
            names.append(stripped.split()[0])
    if not names:
        raise ValueError(f'列表为空: {path}')
    if not allow_duplicates and len(names) != len(set(names)):
        raise ValueError(f'列表不允许重复样本: {path}')
    return names


def sample_path(root: Path, directory: str, name: str, suffix: str):
    relative = Path(name)
    if relative.suffix:
        return root / directory / relative
    return root / directory / relative.with_suffix(suffix)


def load_rgb(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'图像不存在: {path}')
    with Image.open(path) as image:
        image.load()
        return np.asarray(image.convert('RGB')).copy()


def load_label(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f'标签不存在: {path}')
    with Image.open(path) as image:
        label = np.asarray(image)
    if label.ndim != 2:
        raise ValueError(f'标签必须为单通道，实际为{label.shape}: {path}')
    values = np.unique(label)
    allowed = (values == 0) | (values == 255) | (
        (values >= 1) & (values <= NUM_CLASSES))
    if not np.all(allowed):
        raise ValueError(
            f'标签只能包含0、1-8或255，实际为{values.tolist()}: {path}')
    return label.astype(np.uint8, copy=False)


def choose_source_mask(source_label: np.ndarray, rng,
                       min_fraction: float, max_fraction: float,
                       max_attempts: int):
    valid_classes = np.unique(source_label)
    valid_classes = valid_classes[
        (valid_classes >= 1) & (valid_classes <= NUM_CLASSES)]
    if valid_classes.size == 0:
        raise ValueError('源域标签没有任何1-8有效类别')

    valid_pixels = (source_label >= 1) & (source_label <= NUM_CLASSES)
    best = None
    best_distance = float('inf')
    target_fraction = 0.5 * (min_fraction + max_fraction)
    for _ in range(max_attempts):
        shuffled = rng.permutation(valid_classes)
        class_count = max(1, int(np.ceil(valid_classes.size / 2.0)))
        selected_classes = np.sort(shuffled[:class_count])
        mask = np.isin(source_label, selected_classes) & valid_pixels
        fraction = float(mask.mean())
        distance = abs(fraction - target_fraction)
        if distance < best_distance:
            best = (mask, selected_classes, fraction)
            best_distance = distance
        if min_fraction <= fraction <= max_fraction:
            return mask, selected_classes, fraction

    # Some images contain one dominant class and cannot meet both limits.
    # Returning the closest valid semantic mask is safer than using a random
    # rectangle or discarding the sample silently.
    return best


def build_mix(source_image: np.ndarray, source_label: np.ndarray,
              target_image: np.ndarray, target_label: np.ndarray, rng,
              args):
    if source_image.shape != target_image.shape:
        raise ValueError(
            f'源域和目标域图像尺寸不同: {source_image.shape}, '
            f'{target_image.shape}')
    if source_label.shape != target_label.shape:
        raise ValueError(
            f'源域和目标域标签尺寸不同: {source_label.shape}, '
            f'{target_label.shape}')
    if source_image.shape[:2] != source_label.shape:
        raise ValueError('图像和标签尺寸不一致')

    mask, selected_classes, source_fraction = choose_source_mask(
        source_label, rng, args.min_source_fraction,
        args.max_source_fraction, args.max_attempts)
    mixed_image = target_image.copy()
    mixed_label = target_label.copy()
    mixed_image[mask] = source_image[mask]
    mixed_label[mask] = source_label[mask]
    valid_fraction = float(np.mean(
        (mixed_label >= 1) & (mixed_label <= NUM_CLASSES)))
    return mixed_image, mixed_label, selected_classes, source_fraction, \
        valid_fraction


def main():
    args = parse_args()
    validate_args(args)

    data_root = Path(args.data_root)
    real_names = read_list(data_root / args.real_list, allow_duplicates=False)
    combined_names = read_list(
        data_root / args.combined_list, allow_duplicates=True)
    real_set = set(real_names)
    target_names = sorted(set(combined_names) - real_set)
    if not target_names:
        raise ValueError(
            '没有找到测试域伪样本。combined-list必须同时包含真实样本和'
            '额外的测试伪标签样本。')
    reserved_names = real_set | set(target_names)
    collisions = {
        f'classmix_{index:06d}'
        for index in range(args.num_mixes)
    } & reserved_names
    if collisions:
        raise ValueError(
            f'ClassMix输出名与原始样本冲突: '
            f'{sorted(collisions)[:5]}')

    image_out_dir = data_root / args.output_image_dir
    label_out_dir = data_root / args.output_seg_dir
    metadata_dir = data_root / f'{args.output_seg_dir}Meta'
    output_list = data_root / args.output_list
    output_dirs = (image_out_dir, label_out_dir, metadata_dir)
    if any(path.exists() and any(path.iterdir()) for path in output_dirs) \
            and not args.resume:
        raise FileExistsError(
            'ClassMix输出目录非空。请换输出目录，或确认参数完全相同后'
            '增加--resume。脚本不会自动删除已有数据。')
    if output_list.exists() and not args.resume:
        raise FileExistsError(
            f'输出列表已经存在: {output_list}。请使用--resume或换名称。')
    for path in output_dirs:
        path.mkdir(parents=True, exist_ok=True)

    settings = dict(
        real_list=args.real_list,
        combined_list=args.combined_list,
        image_dir=args.image_dir,
        seg_dir=args.seg_dir,
        num_mixes=args.num_mixes,
        seed=args.seed,
        jpeg_quality=args.jpeg_quality,
        min_source_fraction=args.min_source_fraction,
        max_source_fraction=args.max_source_fraction,
        max_attempts=args.max_attempts,
        real_samples=len(real_names),
        target_samples=len(target_names))
    settings_text = json.dumps(settings, sort_keys=True)

    disk_usage = shutil.disk_usage(data_root)
    print(
        f'真实样本={len(real_names)}，测试域伪样本={len(target_names)}，'
        f'计划生成ClassMix={args.num_mixes}')
    print(f'数据盘当前可用空间={disk_usage.free / 1024**3:.2f} GiB')
    print('原始JPEGImages和SegmentationClass不会被修改。')

    generated_names = []
    source_fractions = []
    valid_fractions = []
    for index in range(args.num_mixes):
        mixed_name = f'classmix_{index:06d}'
        image_out = image_out_dir / f'{mixed_name}.jpg'
        label_out = label_out_dir / f'{mixed_name}.png'
        metadata_out = metadata_dir / f'{mixed_name}.json'

        if args.resume and image_out.is_file() and label_out.is_file() \
                and metadata_out.is_file():
            metadata = json.loads(metadata_out.read_text(encoding='utf-8'))
            if metadata.get('settings') != settings_text:
                raise RuntimeError(
                    f'已有样本参数不一致，不能续跑: {metadata_out}')
        else:
            rng = np.random.default_rng(args.seed + index)
            source_name = real_names[int(rng.integers(len(real_names)))]
            # Every target is used three times when num_mixes=1500 and there
            # are 500 target images.  The seed-dependent offset avoids always
            # pairing the same target order with the same source pattern.
            target_offset = args.seed % len(target_names)
            target_name = target_names[
                (index + target_offset) % len(target_names)]

            source_image_path = sample_path(
                data_root, args.image_dir, source_name, args.img_suffix)
            source_label_path = sample_path(
                data_root, args.seg_dir, source_name, args.seg_suffix)
            target_image_path = sample_path(
                data_root, args.image_dir, target_name, args.img_suffix)
            target_label_path = sample_path(
                data_root, args.seg_dir, target_name, args.seg_suffix)
            source_image = load_rgb(source_image_path)
            source_label = load_label(source_label_path)
            target_image = load_rgb(target_image_path)
            target_label = load_label(target_label_path)
            mixed_image, mixed_label, classes, source_fraction, \
                valid_fraction = build_mix(
                    source_image, source_label, target_image, target_label,
                    rng, args)

            Image.fromarray(mixed_image, mode='RGB').save(
                image_out, format='JPEG', quality=args.jpeg_quality,
                subsampling=0, optimize=True)
            Image.fromarray(mixed_label, mode='L').save(
                label_out, format='PNG', optimize=True)
            metadata = dict(
                settings=settings_text,
                name=mixed_name,
                source_name=source_name,
                target_name=target_name,
                selected_source_raw_ids=[int(value) for value in classes],
                source_fraction=source_fraction,
                valid_fraction=valid_fraction)
            metadata_out.write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding='utf-8')

        generated_names.append(mixed_name)
        source_fractions.append(float(metadata['source_fraction']))
        valid_fractions.append(float(metadata['valid_fraction']))
        if (index + 1) % 100 == 0 or index + 1 == args.num_mixes:
            print(f'已完成 {index + 1}/{args.num_mixes}')

    output_list.write_text(
        '\n'.join(generated_names) + '\n', encoding='utf-8')
    report = dict(
        settings=settings,
        output=dict(
            image_dir=str(image_out_dir),
            label_dir=str(label_out_dir),
            list=str(output_list)),
        statistics=dict(
            generated=len(generated_names),
            mean_source_fraction=float(np.mean(source_fractions)),
            min_source_fraction=float(np.min(source_fractions)),
            max_source_fraction=float(np.max(source_fractions)),
            mean_valid_fraction=float(np.mean(valid_fractions))))
    report_path = data_root / 'classmix_build_report.json'
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\nClassMix构建完成')
    print(f'  混合样本: {len(generated_names)}')
    print(f'  平均源域占比: {report["statistics"]["mean_source_fraction"]:.2%}')
    print(f'  平均有效标签占比: {report["statistics"]["mean_valid_fraction"]:.2%}')
    print(f'  图像目录: {image_out_dir}')
    print(f'  标签目录: {label_out_dir}')
    print(f'  训练列表: {output_list}')
    print(f'  构建报告: {report_path}')


if __name__ == '__main__':
    main()