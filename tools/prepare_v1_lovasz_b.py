"""Create a full B config from the original unfiltered V1 run snapshot.

Read-only for images, labels, split files and checkpoints. Only writes a new
config and its audit JSON. Run from the MMSegmentation repository root.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_v1_pseudo import label_histogram, read_ids


EXPECTED_FINGERPRINT = '00359c4b21f2be12c76e82001452a6a634161560eb521385dc8006ed8feeec60'
LOSS_MODULE = 'mmseg.models.losses.aerial_lovasz_loss'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_under(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def make_b_config(cfg, teacher, work_dir):
    """Pure config transformation; leave all existing training settings intact."""
    if cfg.model.get('type') != 'EncoderDecoder':
        raise ValueError('Expected the original EncoderDecoder model')
    head = cfg.model.decode_head
    if head.get('type') != 'SegformerHead' or head.get('num_classes') != 8:
        raise ValueError('Expected an 8-class SegformerHead')
    if head.get('sampler') is not None:
        raise ValueError('Pixel sampling is not supported by the B loss adapter')
    if head.get('ignore_index', 255) != 255:
        raise ValueError('This experiment expects training Ignore ID 255')
    if cfg.model.get('auxiliary_head') or any(
            h.get('type') == 'EMAHook' for h in (cfg.get('custom_hooks') or [])):
        raise ValueError('Use the original V1 config, not the auxiliary-head or EMA experiment')
    losses = head.get('loss_decode')
    if not isinstance(losses, (list, tuple)) or len(losses) != 2:
        raise ValueError('Expected exactly CE + Dice in the original snapshot')
    types = [item.get('type') for item in losses]
    if sorted(types) != ['CrossEntropyLoss', 'DiceLoss']:
        raise ValueError('Expected exactly one CrossEntropyLoss and one DiceLoss')
    ce = losses[types.index('CrossEntropyLoss')]
    dice = losses[types.index('DiceLoss')]
    if ce.get('use_sigmoid', False) or not ce.get('avg_non_ignore', False):
        raise ValueError('Expected softmax CE with avg_non_ignore=True')
    if ce.get('loss_weight', 1.0) != 1.0 or dice.get('loss_weight', 1.0) != 0.5:
        raise ValueError('Expected the confirmed baseline: CE weight 1.0 + Dice weight 0.5')
    dataset = cfg.train_dataloader.dataset
    if (dataset.get('type') != 'PascalVOCDataset'
            or not dataset.get('reduce_zero_label', False)
            or dataset.get('ann_file') != 'train_pseudo.txt'
            or dataset.data_prefix.get('seg_map_path') != 'SegmentationClass'):
        raise ValueError('B must use original train_pseudo.txt + SegmentationClass, not A-filtered labels')
    if dataset.get('img_suffix', '.png') != '.png' or dataset.get('seg_map_suffix', '.png') != '.png':
        raise ValueError('Expected PNG image/label suffixes')
    if cfg.get('default_hooks', {}).get('checkpoint', {}).get('out_dir'):
        raise ValueError('Explicit checkpoint out_dir would mix experiment outputs')

    result = copy.deepcopy(cfg)
    changed_losses = list(copy.deepcopy(losses))
    changed_losses[types.index('DiceLoss')] = dict(
        type='AerialLovaszLoss', loss_type='multi_class', classes='present',
        per_image=False, reduction='none', loss_weight=0.5, loss_name='loss_lovasz')
    result.model.decode_head.loss_decode = changed_losses
    imports = copy.deepcopy(result.get('custom_imports') or {})
    names = imports.get('imports', [])
    names = [names] if isinstance(names, str) else list(names)
    if LOSS_MODULE not in names:
        names.append(LOSS_MODULE)
    imports.update(imports=names, allow_failed_imports=False)
    result.custom_imports = imports
    result.load_from = str(Path(teacher).resolve())
    result.resume = False
    result.work_dir = str(Path(work_dir).resolve())
    return result


def audit_data(cfg, expected_fingerprint=EXPECTED_FINGERPRINT,
               expected_counts=(6296, 500, 700), expected_size=1024):
    """Check the original hard-label experiment without installing anything."""
    dataset = cfg.train_dataloader.dataset
    root = Path(dataset.data_root).resolve()
    real_path = root / 'train.txt'
    combined_path = resolve_under(root, dataset.ann_file)
    val_cfg = cfg.val_dataloader.dataset
    val_root = Path(val_cfg.get('data_root', '')).resolve()
    val_path = resolve_under(val_root, val_cfg.ann_file)
    real, combined, validation = (read_ids(path) for path in (real_path, combined_path, val_path))
    if any(len(items) != len(set(items)) for items in (real, combined, validation)):
        raise ValueError('Real, combined and validation splits must have unique IDs')
    if not set(real).issubset(combined) or set(combined) & set(validation):
        raise ValueError('Missing real training samples or training/validation overlap')
    pseudo = sorted(set(combined) - set(real))
    actual = (len(real), len(pseudo), len(validation))
    if actual != tuple(expected_counts):
        raise ValueError('Sample counts differ: expected {}, got {}'.format(expected_counts, actual))
    image_dir = resolve_under(root, dataset.data_prefix.img_path)
    label_dir = resolve_under(root, dataset.data_prefix.seg_map_path)
    for name in combined:
        for path in (image_dir / (name + '.png'), label_dir / (name + '.png')):
            if not path.is_file():
                raise FileNotFoundError('Missing training file: {}'.format(path))
    hist = np.zeros(256, dtype=np.int64)
    digest = hashlib.sha256()
    for index, name in enumerate(pseudo, 1):
        counts, shape, pixel_hash = label_histogram(label_dir / (name + '.png'), expected_size)
        if counts[0] or counts[255]:
            raise ValueError('B expects unfiltered pseudo labels; Ignore found in {}'.format(name))
        hist += counts
        digest.update('{}\t{}\t{}\n'.format(name, shape, pixel_hash).encode('utf-8'))
        if index % 100 == 0 or index == len(pseudo):
            print('[{}/{}] 原硬伪标签已检查'.format(index, len(pseudo)), flush=True)
    fingerprint = digest.hexdigest()
    if fingerprint != expected_fingerprint:
        raise ValueError('Pseudo pixel fingerprint changed: {}'.format(fingerprint))
    return dict(real_samples=len(real), pseudo_samples=len(pseudo), validation_samples=len(validation),
                combined_samples=len(combined), pseudo_pixel_fingerprint=fingerprint,
                pseudo_class_pixels=hist[1:9].tolist(),
                real_list_sha256=sha256_file(real_path), combined_list_sha256=sha256_file(combined_path),
                validation_list_sha256=sha256_file(val_path))


def prepare(args, expected_counts=(6296, 500, 700), expected_size=1024):
    from mmengine.config import Config
    baseline = Path(args.baseline_config).resolve(strict=True)
    out = Path(args.output_config).resolve()
    report_path = out.with_suffix('.audit.json')
    work_dir = Path(args.work_dir).resolve()
    for path in (out, report_path):
        if path.exists():
            raise FileExistsError('Will not overwrite: {}'.format(path))
    if work_dir.exists() and (not work_dir.is_dir() or any(work_dir.iterdir())):
        raise FileExistsError('Use a new work directory: {}'.format(work_dir))
    cfg = Config.fromfile(str(baseline))
    original_teacher = Path(cfg.get('load_from') or '')
    teacher = Path(args.teacher) if args.teacher else original_teacher
    if not teacher.is_file():
        raise FileNotFoundError('Original teacher is missing; provide the relocated identical file with --teacher')
    teacher_hash = sha256_file(teacher)
    if original_teacher.is_file() and sha256_file(original_teacher) != teacher_hash:
        raise ValueError('Teacher override is not identical to the original initialization')
    a_report = json.loads(Path(args.a_report).read_text(encoding='utf-8'))
    if (a_report.get('complete') is not True or a_report.get('image_count') != expected_counts[1]
            or a_report['settings']['teacher_sha256'] != teacher_hash
            or a_report['settings']['reference_pixel_fingerprint'] != args.expected_fingerprint):
        raise ValueError('Teacher/pseudo provenance differs from the completed A generation report')
    result = make_b_config(cfg, teacher, work_dir)
    audit = audit_data(cfg, args.expected_fingerprint, expected_counts, expected_size)
    text = result.pretty_text
    compile(text, str(out), 'exec')
    adapter_path = Path(__file__).resolve().parents[1] / 'mmseg/models/losses/aerial_lovasz_loss.py'
    if not adapter_path.is_file():
        raise FileNotFoundError('Upload the loss adapter first: {}'.format(adapter_path))
    audit.update(baseline_config=str(baseline), baseline_config_sha256=sha256_file(baseline),
                 teacher_checkpoint=str(teacher.resolve()), teacher_sha256=teacher_hash,
                 adapter_sha256=sha256_file(adapter_path), output_config=str(out), work_dir=str(work_dir),
                 old_loss=cfg.model.decode_head.loss_decode,
                 new_loss=result.model.decode_head.loss_decode,
                 config_text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        stream.write(text)
    with report_path.open('x', encoding='utf-8') as stream:
        json.dump(audit, stream, ensure_ascii=False, indent=2)
    print('完整B配置: {}'.format(out))
    print('审计报告: {}'.format(report_path))
    print('损失: CE 1.0 + Lovasz 0.5；原硬伪标签不变；从同一基础教师初始化。')
    print('训练列表: {}；标签目录: {}'.format(cfg.train_dataloader.dataset.ann_file,
          cfg.train_dataloader.dataset.data_prefix.seg_map_path))
    print('迭代数: {}；学习率: {}'.format(cfg.train_cfg.max_iters, cfg.optim_wrapper.optimizer.lr))
    print('没有修改图片、标签、列表、原配置或权重。训练前请运行 tools/check_lovasz_b.py。')


def main():
    parser = argparse.ArgumentParser(description='从原V1训练快照生成独立CE+Lovasz的B配置')
    parser.add_argument('baseline_config', help='原69.42分学生的训练快照，不是A的conf090配置')
    parser.add_argument('--a-report', default='outputs/pseudo_labels_v1_conf090/generation.json',
                        help='仅用于核验基础教师与原标签指纹，不读取A的过滤PNG')
    parser.add_argument('--teacher', help='仅用于迁移后的同一基础教师checkpoint')
    parser.add_argument('--expected-fingerprint', default=EXPECTED_FINGERPRINT)
    parser.add_argument('--output-config', default='configs/segformer/segformer_b3_mydata_pseudo_lovasz_b.py')
    parser.add_argument('--work-dir', default='work_dirs/segformer_b3_mydata_pseudo_lovasz_b')
    prepare(parser.parse_args())


if __name__ == '__main__':
    main()