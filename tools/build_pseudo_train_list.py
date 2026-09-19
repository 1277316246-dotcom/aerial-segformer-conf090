# Copyright (c) OpenMMLab. All rights reserved.
"""Install V2 pseudo labels and build a repeated self-training list safely."""

import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from PIL import Image


def read_names(path: Path):
    names = [line.strip() for line in path.read_text(encoding='utf-8').splitlines()
             if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError(f'列表中存在重复名称: {path}')
    return names


def check_pseudo_label(path: Path):
    with Image.open(path) as image:
        if image.mode != 'L':
            raise ValueError(f'伪标签不是单通道灰度图: {path}')
        labels = np.asarray(image)
    if labels.ndim != 2 or not set(np.unique(labels).tolist()).issubset(set(range(9))):
        raise ValueError(f'伪标签值异常（必须为0-8）: {path}')


def parse_args():
    parser = ArgumentParser(description='安装V2伪标签并生成重复采样训练列表')
    parser.add_argument('--data-root', default='data/mydata')
    parser.add_argument('--train-list', default='train.txt')
    parser.add_argument('--pseudo-dir', default='outputs/pseudo_labels_v2')
    parser.add_argument('--test-image-dir', default='test_data')
    parser.add_argument('--out-list', default='train_pseudo_v2_x3.txt')
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument(
        '--install', action='store_true',
        help='验证完成后复制测试图和V2伪标签到data_root对应目录')
    parser.add_argument(
        '--backup-dir', default='outputs/pseudo_labels_v1_backup',
        help='安装时备份已有同名伪标签；设为空字符串可关闭备份')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.repeat <= 0:
        raise ValueError('--repeat必须为正整数。')

    data_root = Path(args.data_root)
    train_list = data_root / args.train_list
    out_list = data_root / args.out_list
    image_dir = data_root / 'JPEGImages'
    seg_dir = data_root / 'SegmentationClass'
    pseudo_dir = Path(args.pseudo_dir)
    test_image_dir = Path(args.test_image_dir)

    real_names = read_names(train_list)
    pseudo_paths = sorted(pseudo_dir.glob('*.png'))
    if not pseudo_paths:
        raise FileNotFoundError(f'未找到伪标签: {pseudo_dir}')
    pseudo_names = [path.stem for path in pseudo_paths]
    if len(pseudo_names) != len(set(pseudo_names)):
        raise ValueError('伪标签文件名重复。')
    overlap = set(real_names) & set(pseudo_names)
    if overlap:
        raise ValueError(f'伪标签与真标签训练集重名: {sorted(overlap)[:5]}')

    source_images = {path.stem: path for path in test_image_dir.glob('*.png')}
    missing_images = sorted(set(pseudo_names) - set(source_images))
    if missing_images:
        raise FileNotFoundError(f'测试图缺失，例如: {missing_images[:5]}')
    for path in pseudo_paths:
        check_pseudo_label(path)

    if args.install:
        image_dir.mkdir(parents=True, exist_ok=True)
        seg_dir.mkdir(parents=True, exist_ok=True)
        backup_dir = Path(args.backup_dir) if args.backup_dir else None
        if backup_dir is not None:
            backup_dir.mkdir(parents=True, exist_ok=True)
        for pseudo_path in pseudo_paths:
            name = pseudo_path.stem
            destination_image = image_dir / f'{name}.png'
            destination_label = seg_dir / f'{name}.png'
            if destination_label.is_file() and backup_dir is not None:
                shutil.copy2(destination_label, backup_dir / destination_label.name)
            shutil.copy2(source_images[name], destination_image)
            shutil.copy2(pseudo_path, destination_label)
        print(f'已安装{len(pseudo_paths)}张测试图和V2伪标签。')
    else:
        missing_installed = [
            name for name in pseudo_names
            if not (image_dir / f'{name}.png').is_file()
            or not (seg_dir / f'{name}.png').is_file()]
        if missing_installed:
            raise FileNotFoundError(
                '数据目录中缺少待训练的伪样本；请增加--install。'
                f'例如: {missing_installed[:5]}')

    list_names = real_names + pseudo_names * args.repeat
    out_list.write_text('\n'.join(list_names) + '\n', encoding='utf-8')
    print(f'真标签样本: {len(real_names)}')
    print(f'伪标签样本: {len(pseudo_names)} x {args.repeat}')
    print(f'训练列表总行数: {len(list_names)}')
    print(f'伪样本图片采样比例: '
          f'{len(pseudo_names) * args.repeat / len(list_names):.2%}')
    print(f'训练列表已生成: {out_list}')


if __name__ == '__main__':
    main()
