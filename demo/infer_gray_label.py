# Copyright (c) OpenMMLab. All rights reserved.
"""语义分割推理并生成竞赛要求的灰度标签图。

提交标签要求：
1. 输出文件名与输入图片主文件名一致，格式固定为 PNG；
2. 输出为 1024x1024 的单通道 uint8 灰度图；
3. 模型内部预测 ID 0-7，保存时还原为竞赛标签 ID 1-8；
4. ZIP 内直接放 PNG，不增加额外目录。

用法示例：
    python demo/infer_gray_label.py data/test_images \
        configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512.py \
        work_dirs/deeplabv3plus_mydata/best_mIoU_iter_XXXX.pth \
        --out-dir outputs/pred --zip-file submission.zip --device cuda:0

如需肉眼查看对比，可增加：
    --vis-dir outputs/visualization

如需启用单模型多尺度和水平翻转TTA，可增加：
    --scales 0.75 1.0 1.25 --hflip --amp

注意：vis-dir 中的图像会把类别线性拉伸到 0-255，只能用于查看，
绝对不能作为比赛提交标签。
"""

import zipfile
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


IMG_EXTENSIONS = (
    '.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp')
EXPECTED_NUM_CLASSES = 8


def collect_images(path: str, recursive: bool = False):
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
        raise ValueError(
            '存在主文件名相同的图片，输出 PNG 时会发生覆盖。'
            '请确保测试图片主文件名唯一。')
    return images


def check_readable_image(img_path: Path, expected_size: int):
    """检查图片是否损坏以及输入尺寸是否符合题目要求。"""
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


def convert_to_submission_label(seg_map: np.ndarray, num_classes: int,
                                expected_size: int):
    """把模型训练 ID 0-7 转换为竞赛提交 ID 1-8。"""
    if num_classes != EXPECTED_NUM_CLASSES:
        raise ValueError(
            f'模型类别数应为 {EXPECTED_NUM_CLASSES}，实际为 {num_classes}')
    if seg_map.ndim != 2:
        raise ValueError(f'预测结果应为 HxW，实际形状为 {seg_map.shape}')
    if expected_size > 0 and seg_map.shape != (expected_size, expected_size):
        raise ValueError(
            f'预测尺寸为 {seg_map.shape[1]}x{seg_map.shape[0]}，题目要求为 '
            f'{expected_size}x{expected_size}')

    min_id = int(seg_map.min())
    max_id = int(seg_map.max())
    if min_id < 0 or max_id >= num_classes:
        raise ValueError(
            f'模型预测值应为 0-{num_classes - 1}，实际范围为 '
            f'{min_id}-{max_id}')

    # reduce_zero_label=True 时，原始 ID 1-8 在训练内部变成 0-7。
    # 因此提交时必须执行 +1；原始 ID 0 是 Ignore，不是模型类别。
    return seg_map.astype(np.uint8) + 1


def save_submission_label(label: np.ndarray, out_path: Path,
                          expected_size: int):
    """保存并重新打开检查，确保是无调色板的单通道 PNG。"""
    unique_ids = np.unique(label)
    if unique_ids.min() < 1 or unique_ids.max() > 8:
        raise ValueError(f'提交标签只能包含 ID 1-8，实际为 {unique_ids.tolist()}')

    Image.fromarray(label).save(out_path, format='PNG')

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


def save_visualization(seg_map: np.ndarray, out_path: Path,
                       num_classes: int):
    """另存用于肉眼查看的 0-255 拉伸图，不能用于提交。"""
    if num_classes <= 1:
        gray = np.zeros_like(seg_map, dtype=np.uint8)
    else:
        gray = np.round(
            seg_map.astype(np.float32) * 255.0 /
            (num_classes - 1)).astype(np.uint8)
    Image.fromarray(gray).save(out_path, format='PNG')


def scaled_shape(height: int, width: int, scale: float):
    """按比例缩放并对齐到32的倍数，适配DeepLab骨干网络。"""
    if scale <= 0:
        raise ValueError('--scales中的数值必须大于0')
    new_height = max(32, int(round(height * scale / 32.0)) * 32)
    new_width = max(32, int(round(width * scale / 32.0)) * 32)
    return new_height, new_width


def predict_probabilities(model, image: np.ndarray, scales, rotations,
                          hflip: bool, use_amp: bool):
    """对多尺度、旋转和翻转结果做softmax概率平均。"""
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
                    raise RuntimeError(
                        '模型结果中没有seg_logits，无法执行概率平均TTA')

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


def create_submission_zip(label_paths, zip_path: Path):
    """把本次成功生成的标签放到 ZIP 根目录。"""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
            zip_path, mode='w', compression=zipfile.ZIP_DEFLATED) as archive:
        for label_path in label_paths:
            archive.write(label_path, arcname=label_path.name)

    with zipfile.ZipFile(zip_path) as archive:
        corrupt_name = archive.testzip()
        if corrupt_name is not None:
            raise RuntimeError(f'ZIP 文件损坏，异常成员为: {corrupt_name}')
        names = archive.namelist()
        if len(names) != len(label_paths):
            raise RuntimeError('ZIP 内文件数量与成功预测数量不一致')
        if any('/' in name or '\\' in name for name in names):
            raise RuntimeError('ZIP 内 PNG 必须直接位于根目录')


def parse_args():
    parser = ArgumentParser(description='生成竞赛要求的单通道灰度标签图')
    parser.add_argument('img', help='单张图片或测试图片目录')
    parser.add_argument('config', help='MMSegmentation 配置文件')
    parser.add_argument('checkpoint', help='训练得到的模型权重')
    parser.add_argument(
        '--out-dir', default='outputs/gray_label', help='提交标签保存目录')
    parser.add_argument(
        '--zip-file', help='可选：全部成功后生成提交 ZIP，例如 submission.zip')
    parser.add_argument(
        '--vis-dir', help='可选：保存 0-255 拉伸的可视化图，不能用于提交')
    parser.add_argument(
        '--device', default='cuda:0', help='推理设备，例如 cuda:0 或 cpu')
    parser.add_argument(
        '--recursive', action='store_true', help='递归搜索输入目录中的图片')
    parser.add_argument(
        '--expected-size',
        type=int,
        default=1024,
        help='严格检查输入和输出边长；比赛数据默认 1024，设为 0 可关闭')
    parser.add_argument(
        '--scales',
        type=float,
        nargs='+',
        default=[1.0],
        help='多尺度TTA，例如 --scales 0.75 1.0 1.25')
    parser.add_argument(
        '--hflip', action='store_true', help='增加水平翻转TTA')
    parser.add_argument(
        '--rotations',
        type=int,
        nargs='+',
        default=[0],
        choices=[0, 1, 2, 3],
        help='旋转TTA：0/1/2/3分别代表旋转0/90/180/270度')
    parser.add_argument(
        '--amp', action='store_true', help='在CUDA推理时使用FP16以降低显存占用')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.expected_size < 0:
        raise ValueError('--expected-size 不能为负数')

    out_dir = Path(args.out_dir)
    mkdir_or_exist(str(out_dir))
    vis_dir = Path(args.vis_dir) if args.vis_dir else None
    if vis_dir is not None:
        mkdir_or_exist(str(vis_dir))

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
          f'旋转={args.rotations}，水平翻转={args.hflip}')
    written_labels = []
    skip_count = 0
    fail_count = 0

    for index, img_path in enumerate(image_list, start=1):
        try:
            check_readable_image(img_path, args.expected_size)
            image = mmcv.imread(str(img_path), channel_order='bgr')
            if image is None:
                raise ValueError('MMCV无法读取该图片')
            probabilities, _ = predict_probabilities(
                model,
                image,
                args.scales,
                args.rotations,
                args.hflip,
                args.amp)
            seg_map = probabilities.argmax(dim=0).numpy().astype(np.uint8)
            submission_label = convert_to_submission_label(
                seg_map, num_classes, args.expected_size)

            out_path = out_dir / f'{img_path.stem}.png'
            save_submission_label(
                submission_label, out_path, args.expected_size)
            written_labels.append(out_path)

            if vis_dir is not None:
                save_visualization(
                    seg_map, vis_dir / f'{img_path.stem}.png', num_classes)

            unique_ids = np.unique(submission_label).tolist()
            print(
                f'[OK {index}/{len(image_list)}] {img_path.name} -> '
                f'{out_path.name}, shape={seg_map.shape}, labels={unique_ids}')
        except ValueError as exc:
            skip_count += 1
            print(f'[SKIP] {img_path}\n       原因: {exc}')
        except Exception as exc:
            fail_count += 1
            print(f'[FAIL] 推理失败: {img_path}\n       原因: {exc}')

    ok_count = len(written_labels)
    print(
        f'\n完成: 成功 {ok_count}, 跳过 {skip_count}, 失败 {fail_count}, '
        f'总计 {len(image_list)}')

    # 只要有一张失败或损坏，就不生成一个看似正常但内容不完整的 ZIP。
    if ok_count != len(image_list):
        raise SystemExit(
            '存在未成功生成的标签，已停止创建 ZIP。请修复后重新运行。')

    if args.zip_file:
        zip_path = Path(args.zip_file)
        create_submission_zip(written_labels, zip_path)
        print(f'[ZIP] 已生成 {zip_path}，包含 {len(written_labels)} 张标签图')


if __name__ == '__main__':
    main()