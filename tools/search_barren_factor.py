"""在验证集上搜索 Barren 类别的概率校准系数。

模型内部类别顺序为：
0 Background, 1 Building, 2 Road, 3 Water, 4 Barren,
5 Vegetation, 6 Agricultural, 7 Vehicle。

示例：
python tools/search_barren_factor.py CONFIG CHECKPOINT \
    --factors 0.80 0.85 0.90 0.95 1.00 \
    --scales 0.75 1.0 1.25 --hflip
"""

import json
from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.config import Config
from PIL import Image

from mmseg.apis import inference_model, init_model


NUM_CLASSES = 8
BARREN_INDEX = 4


def parse_args():
    parser = ArgumentParser(
        description='在验证集上搜索Barren概率校准系数')
    parser.add_argument('config', help='MMSegmentation配置文件')
    parser.add_argument('checkpoint', help='模型权重')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument(
        '--factors',
        type=float,
        nargs='+',
        default=[0.80, 0.85, 0.90, 0.95, 1.00],
        help='Barren概率候选乘数；小于1抑制Barren预测')
    parser.add_argument(
        '--scales',
        type=float,
        nargs='+',
        default=[0.75, 1.0, 1.25],
        help='TTA尺度；使用 --scales 1.0 可关闭多尺度')
    parser.add_argument(
        '--hflip', action='store_true', help='增加水平翻转TTA')
    parser.add_argument(
        '--rotations',
        type=int,
        nargs='+',
        default=[0],
        choices=[0, 1, 2, 3],
        help='0/1/2/3分别代表0/90/180/270度旋转')
    parser.add_argument(
        '--amp', action='store_true', help='使用FP16推理')
    parser.add_argument(
        '--output-json',
        default='work_dirs/barren_factor_search.json',
        help='搜索结果JSON保存位置')
    return parser.parse_args()


def scaled_shape(height, width, scale):
    if scale <= 0:
        raise ValueError('scale必须大于0')
    new_height = max(32, int(round(height * scale / 32.0)) * 32)
    new_width = max(32, int(round(width * scale / 32.0)) * 32)
    return new_height, new_width


def predict_probabilities(model, image, scales, rotations, hflip, use_amp):
    """执行与提交脚本一致的概率平均TTA。"""
    height, width = image.shape[:2]
    probability_sum = None
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
                augmented = (
                    np.ascontiguousarray(rotated[:, ::-1])
                    if flipped else rotated)
                amp_context = (
                    torch.autocast(device_type='cuda', dtype=torch.float16)
                    if amp_enabled else nullcontext())

                with amp_context:
                    result = inference_model(model, augmented)

                if not hasattr(result, 'seg_logits'):
                    raise RuntimeError('预测结果中没有seg_logits')

                logits = result.seg_logits.data.float()
                if flipped:
                    logits = torch.flip(logits, dims=(2, ))
                if rotation:
                    logits = torch.rot90(
                        logits, k=-rotation, dims=(1, 2))

                probabilities = logits.softmax(dim=0).unsqueeze(0)
                probabilities = F.interpolate(
                    probabilities,
                    size=(height, width),
                    mode='bilinear',
                    align_corners=False).squeeze(0).cpu()

                probability_sum = (
                    probabilities if probability_sum is None else
                    probability_sum + probabilities)
                variant_count += 1

    return probability_sum / variant_count


def get_dataset_paths(config_path):
    """从最终合并后的配置中读取验证集路径。"""
    cfg = Config.fromfile(config_path)
    dataset_cfg = cfg.val_dataloader.dataset

    # 兼容少量使用dataset wrapper的配置。
    while 'dataset' in dataset_cfg and 'data_prefix' not in dataset_cfg:
        dataset_cfg = dataset_cfg.dataset

    data_root = Path(dataset_cfg.get('data_root', ''))
    ann_file = Path(dataset_cfg.ann_file)
    if not ann_file.is_absolute():
        ann_file = data_root / ann_file

    data_prefix = dataset_cfg.data_prefix
    image_dir = data_root / data_prefix.img_path
    mask_dir = data_root / data_prefix.seg_map_path
    image_suffix = dataset_cfg.get('img_suffix', '.png')
    mask_suffix = dataset_cfg.get('seg_map_suffix', '.png')

    return ann_file, image_dir, mask_dir, image_suffix, mask_suffix


def add_suffix_if_needed(name, suffix):
    path = Path(name)
    return path if path.suffix else Path(f'{name}{suffix}')


def collect_validation_pairs(config_path):
    ann_file, image_dir, mask_dir, image_suffix, mask_suffix = \
        get_dataset_paths(config_path)

    if not ann_file.is_file():
        raise FileNotFoundError(f'验证集列表不存在: {ann_file}')

    pairs = []
    with ann_file.open('r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            value = line.strip()
            if not value:
                continue
            sample_id = value.split()[0]
            image_path = image_dir / add_suffix_if_needed(
                sample_id, image_suffix)
            mask_path = mask_dir / add_suffix_if_needed(
                sample_id, mask_suffix)
            if not image_path.is_file():
                raise FileNotFoundError(
                    f'第{line_number}行图像不存在: {image_path}')
            if not mask_path.is_file():
                raise FileNotFoundError(
                    f'第{line_number}行标签不存在: {mask_path}')
            pairs.append((image_path, mask_path))

    if not pairs:
        raise RuntimeError(f'验证集列表为空: {ann_file}')
    return pairs


def load_training_id_mask(mask_path):
    """把原始标签ID 1-8转成训练ID 0-7，0作为Ignore。"""
    with Image.open(mask_path) as image:
        raw_mask = np.asarray(image)
    if raw_mask.ndim != 2:
        raise ValueError(f'标签不是单通道图像: {mask_path}')

    valid = (raw_mask >= 1) & (raw_mask <= NUM_CLASSES)
    train_mask = np.full(raw_mask.shape, 255, dtype=np.uint8)
    train_mask[valid] = raw_mask[valid].astype(np.uint8) - 1
    return train_mask


def update_confusion(confusion, ground_truth, prediction):
    valid = ground_truth != 255
    encoded = (
        ground_truth[valid].astype(np.int64) * NUM_CLASSES +
        prediction[valid].astype(np.int64))
    confusion += np.bincount(
        encoded, minlength=NUM_CLASSES * NUM_CLASSES).reshape(
            NUM_CLASSES, NUM_CLASSES)


def calculate_metrics(confusion):
    intersection = np.diag(confusion).astype(np.float64)
    gt_area = confusion.sum(axis=1).astype(np.float64)
    pred_area = confusion.sum(axis=0).astype(np.float64)
    union = gt_area + pred_area - intersection

    iou = np.divide(
        intersection,
        union,
        out=np.full(NUM_CLASSES, np.nan),
        where=union > 0)
    accuracy = np.divide(
        intersection,
        gt_area,
        out=np.full(NUM_CLASSES, np.nan),
        where=gt_area > 0)
    total_accuracy = intersection.sum() / max(gt_area.sum(), 1.0)

    return {
        'mIoU': float(np.nanmean(iou)),
        'mAcc': float(np.nanmean(accuracy)),
        'aAcc': float(total_accuracy),
        'barren_iou': float(iou[BARREN_INDEX]),
        'barren_acc': float(accuracy[BARREN_INDEX]),
        'per_class_iou': iou.tolist(),
        'per_class_acc': accuracy.tolist(),
    }


def main():
    args = parse_args()
    factors = sorted(set(float(value) for value in args.factors))
    if not factors or any(value <= 0 for value in factors):
        raise ValueError('--factors必须全部大于0')

    pairs = collect_validation_pairs(args.config)
    model = init_model(
        args.config, args.checkpoint, device=args.device)
    if int(model.decode_head.num_classes) != NUM_CLASSES:
        raise ValueError('该脚本只支持当前8类模型')

    confusions = {
        factor: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        for factor in factors
    }
    variants = (
        len(args.scales) * len(args.rotations) *
        (2 if args.hflip else 1))
    print(f'验证图像数: {len(pairs)}，每张TTA组合数: {variants}')
    print(f'搜索系数: {factors}')

    with torch.inference_mode():
        for index, (image_path, mask_path) in enumerate(pairs, start=1):
            image = mmcv.imread(str(image_path), channel_order='bgr')
            if image is None:
                raise ValueError(f'无法读取图像: {image_path}')
            ground_truth = load_training_id_mask(mask_path)
            probabilities = predict_probabilities(
                model,
                image,
                args.scales,
                args.rotations,
                args.hflip,
                args.amp)

            if tuple(probabilities.shape[1:]) != ground_truth.shape:
                raise ValueError(
                    f'预测与标签尺寸不一致: {image_path.name}, '
                    f'{tuple(probabilities.shape[1:])} vs '
                    f'{ground_truth.shape}')

            for factor in factors:
                calibrated = probabilities.clone()
                calibrated[BARREN_INDEX].mul_(factor)
                prediction = calibrated.argmax(dim=0).numpy().astype(np.uint8)
                update_confusion(
                    confusions[factor], ground_truth, prediction)

            if index % 25 == 0 or index == len(pairs):
                print(f'[{index}/{len(pairs)}] 已完成')

    results = []
    print('\n factor |  mIoU  | Barren IoU | Barren Acc |  aAcc')
    print('--------+--------+------------+------------+-------')
    for factor in factors:
        metrics = calculate_metrics(confusions[factor])
        row = {'factor': factor, **metrics}
        results.append(row)
        print(
            f' {factor:6.3f} | {metrics["mIoU"] * 100:6.2f} | '
            f'{metrics["barren_iou"] * 100:10.2f} | '
            f'{metrics["barren_acc"] * 100:10.2f} | '
            f'{metrics["aAcc"] * 100:5.2f}')

    best = max(results, key=lambda item: item['mIoU'])
    print(
        f'\n最佳系数={best["factor"]:.3f}，'
        f'mIoU={best["mIoU"] * 100:.2f}，'
        f'Barren IoU={best["barren_iou"] * 100:.2f}')

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as file:
        json.dump(
            {
                'config': args.config,
                'checkpoint': args.checkpoint,
                'scales': args.scales,
                'rotations': args.rotations,
                'hflip': args.hflip,
                'amp': args.amp,
                'results': results,
                'best': best,
            },
            file,
            ensure_ascii=False,
            indent=2)
    print(f'结果已保存: {output_path}')


if __name__ == '__main__':
    main()