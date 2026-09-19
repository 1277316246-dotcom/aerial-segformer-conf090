"""Generate confidence-filtered raw-ID pseudo labels with the existing TTA.

This script reuses infer_gray_label.predict_probabilities, but writes 0 for
pixels below the confidence threshold. These PNGs are TRAINING labels only.
Heavy dependencies are imported inside main so --help and CPU unit tests work.
"""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = ('Background', 'Building', 'Road', 'Water', 'Barren',
           'Vegetation', 'Agricultural', 'Vehicle')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def filter_probabilities(probabilities, threshold):
    """Return raw label, raw argmax, confidence, and per-class counts."""
    probabilities = np.asarray(probabilities, dtype=np.float32)
    if probabilities.ndim != 3 or probabilities.shape[0] != 8:
        raise ValueError('Expected probabilities of shape (8, H, W)')
    if not 0 < threshold <= 1 or not np.isfinite(probabilities).all():
        raise ValueError('Invalid threshold or non-finite model probabilities')
    if probabilities.min() < -1e-6 or probabilities.max() > 1.00001:
        raise ValueError('Input is not a probability array')
    if not np.allclose(probabilities.sum(axis=0), 1, atol=2e-4):
        raise ValueError('Class probabilities do not sum to 1')
    hard = probabilities.argmax(axis=0).astype(np.uint8) + 1
    confidence = probabilities.max(axis=0)
    keep = confidence >= threshold
    label = np.where(keep, hard, 0).astype(np.uint8)
    before = np.bincount(hard.ravel(), minlength=9)[1:9]
    after = np.bincount(label.ravel(), minlength=9)[1:9]
    return label, hard, confidence, before, after


def read_reference(path, expected_size):
    with Image.open(path) as image:
        array = np.asarray(image).copy()
    if array.shape != (expected_size, expected_size):
        raise ValueError('Reference label shape mismatch: {}'.format(path))
    if not np.issubdtype(array.dtype, np.integer) or array.min() < 1 or array.max() > 8:
        raise ValueError('Reference must be the unfiltered raw IDs 1-8: {}'.format(path))
    return array.astype(np.uint8)


def filter_reference(probabilities, reference, threshold):
    """Mask low confidence without replacing any existing class annotation."""
    label, hard, confidence, _, _ = filter_probabilities(probabilities, threshold)
    if reference.shape != hard.shape or reference.min() < 1 or reference.max() > 8:
        raise ValueError('Reference must have matching shape and raw IDs 1-8')
    # A disagreement is not silently relabelled: retain only confident matches.
    label = np.where(hard == reference, label, 0).astype(np.uint8)
    before = np.bincount(reference.ravel(), minlength=9)[1:9]
    after = np.bincount(label.ravel(), minlength=9)[1:9]
    return label, hard, confidence, before, after


def write_json_new(path, data):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description='生成0.90置信度过滤伪标签并核对原硬标签')
    parser.add_argument('img', help='500张测试图片的目录')
    parser.add_argument('config', help='生成原硬伪标签时使用的教师配置')
    parser.add_argument('checkpoint', help='同一个基础教师checkpoint，不是V1学生')
    parser.add_argument('--out-dir', default='outputs/pseudo_labels_v1_conf090')
    parser.add_argument('--reference-dir', default='data/mydata/SegmentationClass')
    parser.add_argument('--expected-fingerprint', help='audit_v1_pseudo打印的像素指纹')
    parser.add_argument('--conf-thresh', type=float, default=0.90)
    parser.add_argument('--scales', type=float, nargs='+', default=[0.75, 1.0, 1.25])
    parser.add_argument('--rotations', type=int, nargs='+', choices=[0, 1, 2, 3], default=[0])
    parser.add_argument('--hflip', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--expected-count', type=int, default=500)
    parser.add_argument('--expected-size', type=int, default=1024)
    parser.add_argument('--resume', action='store_true', help='同参数续跑，通过哈希核验已完成样本')
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0 < args.conf_thresh <= 1 or args.expected_count <= 0 or args.expected_size <= 0:
        raise ValueError('Invalid confidence/count/size')
    if any(not np.isfinite(s) or s <= 0 for s in args.scales):
        raise ValueError('Scales must be finite and positive')
    if len(set(args.scales)) != len(args.scales) or len(set(args.rotations)) != len(args.rotations):
        raise ValueError('Duplicate TTA scales/rotations are not allowed')
    checkpoint = Path(args.checkpoint).resolve(strict=True)
    config_path = Path(args.config).resolve(strict=True)
    inference_path = Path(__file__).with_name('infer_gray_label.py').resolve(strict=True)

    from mmengine.config import Config
    from mmengine.model import revert_sync_batchnorm
    import mmcv
    import torch
    from mmseg.apis import init_model

    # Load the sibling file explicitly: works for both direct and imported usage.
    spec = importlib.util.spec_from_file_location('_conf090_gray', inference_path)
    gray = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gray)
    images = sorted((Path(p) for p in gray.collect_images(args.img)), key=lambda item: item.stem)
    if len(images) != args.expected_count:
        raise ValueError('Expected {} images, found {}'.format(args.expected_count, len(images)))
    source_fingerprint = hashlib.sha256()
    reference_fingerprint = hashlib.sha256()
    for path in images:
        gray.check_readable_image(path, args.expected_size)
        source_fingerprint.update('{}\t{}\n'.format(path.name, sha256_file(path)).encode('utf-8'))
        reference = read_reference(Path(args.reference_dir) / (path.stem + '.png'), args.expected_size)
        digest = hashlib.sha256(reference.tobytes()).hexdigest()
        reference_fingerprint.update('{}\t{}\t{}\n'.format(path.stem, reference.shape, digest).encode('utf-8'))
    ref_hash = reference_fingerprint.hexdigest()
    if args.expected_fingerprint and ref_hash.lower() != args.expected_fingerprint.lower():
        raise ValueError('Installed reference labels no longer match the audited fingerprint')

    cfg = Config.fromfile(str(config_path))
    cfg_text = cfg.pretty_text
    settings = dict(
        schema_version=1, teacher_config=str(config_path), teacher_checkpoint=str(checkpoint),
        teacher_sha256=sha256_file(checkpoint), expanded_config_sha256=hashlib.sha256(cfg_text.encode('utf-8')).hexdigest(),
        inference_script_sha256=sha256_file(inference_path), generator_sha256=sha256_file(__file__),
        image_files_sha256=source_fingerprint.hexdigest(), reference_pixel_fingerprint=ref_hash,
        threshold=args.conf_thresh, label_policy='confidence_and_reference_agreement',
        scales=args.scales, rotations=args.rotations,
        hflip=args.hflip, amp=args.amp, device=args.device,
        expected_count=args.expected_count, expected_size=args.expected_size)
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        if not args.resume or not (out / 'settings.json').is_file():
            raise FileExistsError('Output is not empty; use a new directory or --resume with identical settings')
        if json.loads((out / 'settings.json').read_text(encoding='utf-8')) != settings:
            raise ValueError('Resume settings, teacher, inputs, or implementation changed')
    else:
        out.mkdir(parents=True, exist_ok=True)
        write_json_new(out / 'settings.json', settings)
        (out / 'teacher_config_expanded.py').write_text(cfg_text, encoding='utf-8')
    records_dir = out / 'records'
    records_dir.mkdir(exist_ok=True)
    model = init_model(cfg, str(checkpoint), device=args.device)
    if args.device.startswith('cpu'):
        model = revert_sync_batchnorm(model)
    if int(model.decode_head.num_classes) != 8:
        raise ValueError('Teacher must have 8 classes')
    variants = len(args.scales) * len(args.rotations) * (2 if args.hflip else 1)
    print('教师SHA256: {}'.format(settings['teacher_sha256']), flush=True)
    print('样本={}，TTA={}，置信度阈值={}'.format(len(images), variants, args.conf_thresh), flush=True)
    print('参考硬标签像素指纹: {}'.format(ref_hash), flush=True)
    rows = []
    for index, path in enumerate(images, 1):
        label_path = out / (path.stem + '.png')
        record_path = records_dir / (path.stem + '.json')
        if record_path.exists():
            row = json.loads(record_path.read_text(encoding='utf-8'))
            if not label_path.is_file() or sha256_file(label_path) != row['label_file_sha256']:
                raise ValueError('Previously generated label changed: {}'.format(label_path))
        else:
            if label_path.exists():
                raise FileExistsError('Unverified partial output: {}. Use a new output directory.'.format(label_path))
            image = mmcv.imread(str(path), channel_order='bgr')
            if image is None:
                raise ValueError('Cannot read image: {}'.format(path))
            with torch.inference_mode():
                probabilities, _ = gray.predict_probabilities(
                    model, image, args.scales, args.rotations, args.hflip, args.amp)
            array = probabilities.detach().float().cpu().numpy()
            reference = read_reference(Path(args.reference_dir) / (path.stem + '.png'), args.expected_size)
            label, hard, confidence, before, after = filter_reference(array, reference, args.conf_thresh)
            if label.shape != reference.shape:
                raise ValueError('Prediction/reference shape mismatch')
            # Use uint8 HxW so Pillow writes a real L-mode PNG, including Ignore=0.
            Image.fromarray(label).save(label_path, format='PNG')
            with Image.open(label_path) as check:
                if check.mode != 'L' or not np.array_equal(np.asarray(check), label):
                    raise ValueError('Saved label verification failed: {}'.format(label_path))
            row = dict(name=path.stem, pixels=int(label.size), retained=int(after.sum()),
                       hard_disagreements=int(np.count_nonzero(hard != reference)),
                       mean_confidence=float(confidence.mean()), before=before.tolist(), after=after.tolist(),
                       label_file_sha256=sha256_file(label_path))
            write_json_new(record_path, row)
        rows.append(row)
        print('[{}/{}] {}: 保留={:.2%}, 与原硬标签分歧={:.4%}'.format(
            index, len(images), path.name, row['retained'] / row['pixels'],
            row['hard_disagreements'] / row['pixels']), flush=True)

    total = sum(r['pixels'] for r in rows)
    before = np.sum([r['before'] for r in rows], axis=0)
    after = np.sum([r['after'] for r in rows], axis=0)
    report = dict(complete=True, settings=settings, image_count=len(rows), pixels=total,
                  retained_ratio=sum(r['retained'] for r in rows) / total,
                  hard_disagreement_ratio=sum(r['hard_disagreements'] for r in rows) / total,
                  all_ignore_images=[r['name'] for r in rows if r['retained'] == 0],
                  classes=[dict(name=name, before=int(a), retained=int(b), retention_ratio=float(b / a) if a else None)
                           for name, a, b in zip(CLASSES, before, after)], records=rows)
    report_path = out / 'generation.json'
    if report_path.exists():
        if json.loads(report_path.read_text(encoding='utf-8')) != report:
            raise ValueError('Existing generation report changed unexpectedly')
    else:
        write_json_new(report_path, report)
    print('\n完成: {}张训练伪标签；有效像素={:.2%}；Ignore={:.2%}'.format(
        len(rows), report['retained_ratio'], 1 - report['retained_ratio']), flush=True)
    print('重新预测与原硬标签的分歧比例: {:.6%}'.format(report['hard_disagreement_ratio']))
    print('Class          before_pixels  retained_pixels  class_retention')
    for item in report['classes']:
        ratio = '{:.2%}'.format(item['retention_ratio']) if item['retention_ratio'] is not None else 'N/A'
        print('{:<14} {:>13d} {:>16d} {:>16}'.format(item['name'], item['before'], item['retained'], ratio))
    print('全Ignore图像数: {}'.format(len(report['all_ignore_images'])))
    print('生成报告: {}'.format(report_path))
    print('这些PNG含0，仅用于训练；不要当作比赛提交标签。')


if __name__ == '__main__':
    main()