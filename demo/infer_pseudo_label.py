# Copyright (c) OpenMMLab. All rights reserved.
"""生成用于自训练（半监督）的伪标签。

与 infer_gray_label.py 的区别：
1. 输出是"训练格式"而非提交格式：高置信度像素 = argmax + 1 (1-8)，
   低置信度像素 = 0；配合 reduce_zero_label=True 加载后分别变成
   0-7 训练ID 和 255(ignore)。
2. 依赖 softmax 置信度过滤，不把低置信度像素硬塞进某个类别，
   避免把 68.4 模型的错误固化进训练集。

用法示例：
    python demo/infer_pseudo_label.py data/test_images \
        configs/segformer/segformer_mit-b3_...py \
        work_dirs/segformer_b3_mydata/best_mIoU_iter_XXXX.pth \
        --out-dir data/mydata/SegmentationClass \
        --conf-thresh 0.90 \
        --scales 0.75 1.0 1.25 --rotations 0 1 2 3 --hflip --amp

生成后：
    1. 把测试图像拷到 data/mydata/JPEGImages/
    2. train_pseudo.txt = train.txt + 测试图主文件名
    3. 配置里 ann_file 改成 train_pseudo.txt，load_from 68.4 权重微调
"""

from argparse import ArgumentParser
from contextlib import nullcontext
from pathlib import Path

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.model import revert_sync_batchnorm
from mmengine.utils import mkdir_or_exist
from PIL import Image

from mmseg.apis import inference_model, init_model


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
EXPECTED_NUM_CLASSES = 8
# 伪标签 PNG 的合法值：0 = ignore（加载后变 255），1-8 = 有效类别
VALID_PSEUDO_IDS = set(range(0, 9))


def collect_images(path, recursive=False):
    """收集单张图片或目录中的图片，并检查输出文件名冲突。"""
    input_path = Path(path)
    if input_path.is_file():
        if input_path.suffix.lower() not in IMG_EXTENSIONS:
            raise ValueError(f'不支持的图片格式: {input_path}')
        images = [input_path]
    elif input_path.is_dir():
        iterator = input_path.rglob('*') if recursive else input_path.iterdir()
        images = sorted(
            item for item in iterator
            if item.is_file() and item.suffix.lower() in IMG_EXTENSIONS)
        if not images:
            raise FileNotFoundError(f'目录中未找到图片: {input_path}')
    else:
        raise FileNotFoundError(f'输入路径不存在: {input_path}')

    output_names = [f'{image.stem}.png' for image in images]
    if len(output_names) != len(set(output_names)):
        raise ValueError('存在主文件名相同的图片，输出 PNG 时会发生覆盖。')
    return images


def check_readable_image(img_path, expected_size):
    """检查图片是否损坏以及输入尺寸是否符合要求。"""
    try:
        with Image.open(img_path) as image:
            image.verify()
        with Image.open(img_path) as image:
            image.load()
            size = image.size
    except Exception as exc:
        raise ValueError(f'图片无法读取或已经损坏: {exc}') from exc

    if expected_size > 0 and size != (expected_size, expected_size):
        raise ValueError(
            f'输入尺寸为 {size[0]}x{size[1]}，题目要求为 '
            f'{expected_size}x{expected_size}')


def scaled_shape(height, width, scale):
    """按比例缩放并对齐到32的倍数。"""
    if scale <= 0:
        raise ValueError('--scales中的数值必须大于0')
    new_height = max(32, int(round(height * scale / 32.0)) * 32)
    new_width = max(32, int(round(width * scale / 32.0)) * 32)
    return new_height, new_width


def predict_probabilities(model, image, scales, rotations, hflip, use_amp):
    """对多尺度、旋转和翻转结果做 softmax 概率平均。"""
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
                    raise RuntimeError('模型结果中没有 seg_logits，无法执行 TTA')

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
                variant_count += 1

    return probability_sum / variant_count, variant_count


def predict_pseudo_label(model, image, scales, rotations, hflip, use_amp,
                         conf_thresh):
    """生成训练格式的伪标签。

    返回 (pseudo_png, stats):
      pseudo_png: HxW uint8
          高置信度像素 = argmax + 1 (1-8)，加载时经 reduce_zero_label
          变回 0-7；低置信度像素 = 0，加载后变 255(ignore)。
      stats: 字典，含保留像素占比和平均置信度，用于判断阈值是否合理。
    """
    if not 0.0 < conf_thresh <= 1.0:
        raise ValueError('conf_thresh 必须在 (0, 1] 范围内')

    prob, _ = predict_probabilities(model, image, scales, rotations,
                                    hflip, use_amp)
    max_prob, argmax = prob.max(dim=0)              # argmax: 0-7 训练ID
    max_prob = max_prob.numpy()
    argmax = argmax.numpy().astype(np.uint8)

    confident = max_prob >= conf_thresh
    # 训练ID + 1 → PNG/加载ID (1-8)；低置信度 → 0 (加载后变 255 ignore)
    pseudo_png = np.where(confident, argmax + 1, 0).astype(np.uint8)

    stats = {
        'conf_ratio': float(confident.mean()),   # 保留为有效类别的像素占比
        'mean_conf': float(max_prob.mean()),     # 全图平均最大 softmax
    }
    return pseudo_png, stats


def save_pseudo_label(label, out_path, expected_size):
    """保存并重新打开检查，确保是单通道灰度 PNG。"""
    unique_ids = set(np.unique(label).tolist())
    if not unique_ids.issubset(VALID_PSEUDO_IDS):
        raise ValueError(f'伪标签只能包含 0-8，实际为 {sorted(unique_ids)}')

    Image.fromarray(label, mode='L').save(out_path, format='PNG')

    with Image.open(out_path) as image:
        if image.format != 'PNG':
            raise ValueError(f'输出文件不是 PNG: {out_path}')
        if image.mode != 'L':
            raise ValueError(
                f'输出必须为单通道灰度 L 模式，实际为 {image.mode}')
        if expected_size > 0 and image.size != (expected_size, expected_size):
            raise ValueError(
                f'输出尺寸为 {image.size}，要求为 '
                f'{expected_size}x{expected_size}')


def parse_args():
    parser = ArgumentParser(description='生成自训练用的置信度过滤伪标签')
    parser.add_argument('img', help='单张图片或测试图片目录')
    parser.add_argument('config', help='MMSegmentation 配置文件')
    parser.add_argument('checkpoint', help='训练得到的模型权重')
    parser.add_argument(
        '--out-dir', default='outputs/pseudo_labels',
        help='伪标签保存目录；可直接指到 data/mydata/SegmentationClass')
    parser.add_argument(
        '--conf-thresh', type=float, default=0.90,
        help='最大 softmax 置信度阈值；低于该值的像素设为 ignore(255)')
    parser.add_argument(
        '--device', default='cuda:0', help='推理设备，例如 cuda:0 或 cpu')
    parser.add_argument(
        '--recursive', action='store_true', help='递归搜索输入目录中的图片')
    parser.add_argument(
        '--expected-size', type=int, default=1024,
        help='严格检查输入和输出边长；设为 0 可关闭')
    parser.add_argument(
        '--scales', type=float, nargs='+', default=[1.0],
        help='多尺度TTA，例如 --scales 0.75 1.0 1.25')
    parser.add_argument(
        '--hflip', action='store_true', help='增加水平翻转TTA')
    parser.add_argument(
        '--rotations', type=int, nargs='+', default=[0],
        choices=[0, 1, 2, 3],
        help='旋转TTA：0/1/2/3分别代表旋转0/90/180/270度')
    parser.add_argument(
        '--amp', action='store_true', help='CUDA推理时使用FP16降低显存')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.expected_size < 0:
        raise ValueError('--expected-size 不能为负数')

    out_dir = Path(args.out_dir)
    mkdir_or_exist(str(out_dir))

    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device.startswith('cpu'):
        model = revert_sync_batchnorm(model)

    num_classes = int(model.decode_head.num_classes)
    if num_classes != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f'当前脚本只接受 8 类比赛模型，实际为 {num_classes} 类')

    image_list = collect_images(args.img, recursive=args.recursive)
    variant_count = (
        len(args.scales) * len(args.rotations) * (2 if args.hflip else 1))
    print(f'TTA组合数: {variant_count}，尺度={args.scales}，'
          f'旋转={args.rotations}，水平翻转={args.hflip}，'
          f'置信度阈值={args.conf_thresh}')

    written = []
    conf_ratios = []
    mean_confs = []

    for index, img_path in enumerate(image_list, start=1):
        try:
            check_readable_image(img_path, args.expected_size)
            image = mmcv.imread(str(img_path), channel_order='bgr')
            if image is None:
                raise ValueError('MMCV无法读取该图片')

            pseudo_png, stats = predict_pseudo_label(
                model, image, args.scales, args.rotations,
                args.hflip, args.amp, args.conf_thresh)

            out_path = out_dir / f'{img_path.stem}.png'
            save_pseudo_label(pseudo_png, out_path, args.expected_size)
            written.append(out_path)

            conf_ratios.append(stats['conf_ratio'])
            mean_confs.append(stats['mean_conf'])
            print(
                f'[OK {index}/{len(image_list)}] {img_path.name} -> '
                f'{out_path.name}, 保留占比={stats["conf_ratio"]:.1%}, '
                f'平均置信度={stats["mean_conf"]:.3f}')
        except ValueError as exc:
            print(f'[SKIP] {img_path}\n       原因: {exc}')
        except Exception as exc:
            print(f'[FAIL] 推理失败: {img_path}\n       原因: {exc}')

    if not written:
        raise SystemExit('没有成功生成任何伪标签。')

    print(f'\n完成: 成功 {len(written)}, 总计 {len(image_list)}')
    if conf_ratios:
        overall_ratio = float(np.mean(conf_ratios))
        overall_conf = float(np.mean(mean_confs))
        print(f'平均保留像素占比: {overall_ratio:.1%}')
        print(f'平均最大 softmax:  {overall_conf:.3f}')
        # 阈值过高会导致大量像素变 ignore，训练信号不足
        if overall_ratio < 0.60:
            warnings.warn(
                f'保留像素占比仅 {overall_ratio:.1%}，低于 60%。'
                '建议把 --conf-thresh 降到 0.85 左右，否则有效监督太少。')


if __name__ == '__main__':
    main()
