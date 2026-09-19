"""Build an isolated 0.90 pseudo-label experiment from a saved V1 run config.

Copies labels only (not images).  Keeps original SegmentationClass and split
files intact.  Generates a fully expanded config with unchanged model, losses,
pipelines, schedule and sampling order. Requires a completed generation report.
"""

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from audit_v1_pseudo import read_ids


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_under(data_root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (data_root / path).resolve()


def prepare(cfg, args):
    dataset = cfg.train_dataloader.dataset
    if dataset.get('type') != 'PascalVOCDataset' or not dataset.get('reduce_zero_label', False):
        raise ValueError('Expected plain PascalVOCDataset with reduce_zero_label=True')
    if cfg.model.decode_head.num_classes != 8 or cfg.model.decode_head.type != 'SegformerHead':
        raise ValueError('Expected the original 8-class SegformerHead')
    if cfg.model.get('auxiliary_head') or any(h.get('type') == 'EMAHook' for h in cfg.get('custom_hooks', [])):
        raise ValueError('This is not the V1 baseline config (auxiliary head or EMA found)')
    if dataset.get('img_suffix', '.png') != '.png' or dataset.get('seg_map_suffix', '.png') != '.png':
        raise ValueError('This preparation script expects PNG images and labels')
    root = Path(dataset.data_root).resolve()
    real_path = resolve_under(root, 'train.txt')
    combined_path = resolve_under(root, dataset.ann_file)
    real = read_ids(real_path)
    combined = read_ids(combined_path)
    if len(real) != len(set(real)) or len(combined) != len(set(combined)):
        raise ValueError('V1 expects unique samples, with no repeated pseudo IDs')
    if not set(real).issubset(combined):
        raise ValueError('Combined split is missing real samples')
    validation_cfg = cfg.val_dataloader.dataset
    validation_root = Path(validation_cfg.get('data_root', '')).resolve()
    validation = read_ids(resolve_under(validation_root, validation_cfg.ann_file))
    if set(combined) & set(validation):
        raise ValueError('Training/validation overlap detected')
    pseudo_ids = sorted(set(combined) - set(real))
    pseudo_dir = Path(args.pseudo_dir).resolve()
    report = json.loads((pseudo_dir / 'generation.json').read_text(encoding='utf-8'))
    if report.get('complete') is not True or report['image_count'] != len(pseudo_ids) or len(pseudo_ids) != 500:
        raise ValueError('Expected a completed generation report for all 500 pseudo samples')
    if abs(report['settings']['threshold'] - 0.90) > 1e-8:
        raise ValueError('This experiment expects confidence threshold 0.90')
    records = {r['name']: r for r in report['records']}
    if len(records) != len(report['records']) or set(records) != set(pseudo_ids):
        raise ValueError('Generated pseudo IDs do not match the V1 split')
    if report['hard_disagreement_ratio'] > args.max_hard_disagreement:
        raise ValueError('Regenerated hard labels differ by {:.4%}. Check the teacher/config/TTA before training.'.format(
            report['hard_disagreement_ratio']))
    if report['all_ignore_images']:
        raise ValueError('Some pseudo images are all Ignore; inspect the report before training')
    teacher = Path(args.teacher or cfg.get('load_from') or '').resolve()
    if not teacher.is_file():
        raise FileNotFoundError('Saved teacher is missing; use --teacher with the same teacher checkpoint')
    teacher_hash = sha256_file(teacher)
    if teacher_hash != report['settings']['teacher_sha256']:
        raise ValueError('Student initialization checkpoint differs from the pseudo-label teacher')
    saved_teacher = Path(cfg.get('load_from') or '')
    if saved_teacher.is_file() and sha256_file(saved_teacher) != teacher_hash:
        raise ValueError('Provided teacher differs from the existing baseline initialization checkpoint')

    source_seg = resolve_under(root, dataset.data_prefix.seg_map_path)
    image_dir = resolve_under(root, dataset.data_prefix.img_path)
    out_seg = resolve_under(root, args.output_seg_dir)
    out_list = resolve_under(root, args.output_list)
    out_config = Path(args.output_config).resolve()
    work_dir = Path(args.work_dir).resolve()
    # Restrict newly installed label files to a new sibling directory.
    if out_seg.parent != root or out_seg == source_seg or out_list.parent != root:
        raise ValueError('Output labels/list must be new entries directly under data_root')
    for path in (out_seg, out_list, out_config):
        if path.exists():
            raise FileExistsError('Will not overwrite existing experiment: {}'.format(path))
    if work_dir.exists() and any(work_dir.iterdir()):
        raise FileExistsError('Use a new work directory: {}'.format(work_dir))
    hook = cfg.get('default_hooks', {}).get('checkpoint', {})
    if hook.get('out_dir'):
        raise ValueError('Remove an explicitly configured checkpoint out_dir from the experiment copy first')

    sources = []
    original_fingerprint = hashlib.sha256()
    filtered_count = np.zeros(9, dtype=np.int64)
    for name in combined:
        image_path = image_dir / (name + '.png')
        if not image_path.is_file():
            raise FileNotFoundError('Image missing: {}'.format(image_path))
        original = source_seg / (name + '.png')
        if not original.is_file():
            raise FileNotFoundError('Original label missing: {}'.format(original))
        sources.append((name, pseudo_dir / (name + '.png') if name in records else original))
    # Use the same canonical fingerprint convention as audit_v1_pseudo.py.
    for name in pseudo_ids:
        original = source_seg / (name + '.png')
        with Image.open(original) as im:
            array = np.asarray(im).astype(np.uint8)
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        original_fingerprint.update('{}\t{}\t{}\n'.format(name, array.shape, digest).encode('utf-8'))
        label_path = pseudo_dir / (name + '.png')
        if sha256_file(label_path) != records[name]['label_file_sha256']:
            raise ValueError('Filtered label differs from generation report: {}'.format(label_path))
        with Image.open(label_path) as im:
            filtered = np.asarray(im)
            if im.mode != 'L' or filtered.shape != array.shape or filtered.max() > 8:
                raise ValueError('Filtered label format mismatch: {}'.format(label_path))
            if np.any((filtered != 0) & (filtered != array)):
                raise ValueError('Filtered PNG replaced a class instead of only masking: {}'.format(label_path))
            filtered_count += np.bincount(filtered.ravel(), minlength=9)
    if original_fingerprint.hexdigest() != report['settings']['reference_pixel_fingerprint']:
        raise ValueError('Original installed pseudo labels changed after generation')
    if filtered_count[0] == 0:
        raise ValueError('No pixels were filtered; this would not test confidence filtering')

    required = sum(path.stat().st_size for _, path in sources)
    if shutil.disk_usage(root).free < required + 256 * 1024**2:
        raise OSError('Not enough free data-disk space for independent label copies')
    cfg = copy.deepcopy(cfg)
    cfg.train_dataloader.dataset.data_prefix.seg_map_path = out_seg.name
    cfg.train_dataloader.dataset.ann_file = out_list.name
    cfg.load_from = str(teacher)
    cfg.resume = False
    cfg.work_dir = str(work_dir)
    # Serialize before creating output files, to catch config errors first.
    config_text = cfg.pretty_text
    out_seg.mkdir()
    for index, (name, source) in enumerate(sources, 1):
        dest = out_seg / (name + '.png')
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        if index % 1000 == 0 or index == len(sources):
            print('[{}/{}] 标签已复制'.format(index, len(sources)), flush=True)
    # Preserve exactly the original sampling order and line contents.
    shutil.copy2(combined_path, out_list)
    out_config.parent.mkdir(parents=True, exist_ok=True)
    with out_config.open('x', encoding='utf-8') as stream:
        stream.write(config_text)
    build_report = dict(
        baseline_config=str(Path(args.baseline_config).resolve()),
        baseline_config_file_sha256=sha256_file(args.baseline_config),
        teacher_sha256=teacher_hash, generation_report=str(pseudo_dir / 'generation.json'),
        original_pseudo_fingerprint=original_fingerprint.hexdigest(),
        real_samples=len(real), pseudo_samples=len(pseudo_ids), combined_samples=len(combined),
        output_config=str(out_config), output_seg_dir=str(out_seg), output_list=str(out_list),
        retained_ratio=float(filtered_count[1:].sum() / filtered_count.sum()))
    with (out_seg / 'build_report.json').open('x', encoding='utf-8') as stream:
        json.dump(build_report, stream, ensure_ascii=False, indent=2)
    print('实验配置: {}'.format(out_config))
    print('训练标签: {}'.format(out_seg))
    print('训练列表: {}，共{}张'.format(out_list, len(combined)))
    print('只改变训练标签；模型、损失、增强、优化器和迭代配置来自原训练快照。')


def main():
    parser = argparse.ArgumentParser(description='从V1训练快照创建独立的置信度过滤实验')
    parser.add_argument('baseline_config', help='69.42分那次训练目录中保存的完整配置')
    parser.add_argument('--pseudo-dir', default='outputs/pseudo_labels_v1_conf090')
    parser.add_argument('--teacher', help='仅在原教师路径迁移时传入；SHA256必须与生成教师一致')
    parser.add_argument('--output-seg-dir', default='SegmentationClassV1Conf090')
    parser.add_argument('--output-list', default='train_pseudo_conf090.txt')
    parser.add_argument('--output-config', default='configs/segformer/segformer_b3_mydata_pseudo_conf090.py')
    parser.add_argument('--work-dir', default='work_dirs/segformer_b3_mydata_pseudo_conf090')
    parser.add_argument('--max-hard-disagreement', type=float, default=0.001,
                        help='允许至多0.1%%的重预测硬标签差异；超过则停止准备')
    args = parser.parse_args()
    if not 0 <= args.max_hard_disagreement <= 1:
        parser.error('max-hard-disagreement must be in [0, 1]')
    from mmengine.config import Config
    prepare(Config.fromfile(args.baseline_config), args)


if __name__ == '__main__':
    main()
