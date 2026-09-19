"""Read-only audit of the pseudo labels actually installed for V1 training.

Requires NumPy and Pillow, but no CUDA or MMSegmentation.  Raw label IDs 1--8
are classes; 0 and 255 are counted as Ignore.  Ignore pixels do not by
themselves prove that confidence filtering was used: generation provenance
must also be checked.  This script never changes images, labels or split files.
"""

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image


CLASSES = ('Background', 'Building', 'Road', 'Water', 'Barren',
           'Vegetation', 'Agricultural', 'Vehicle')


def read_ids(path):
    ids = []
    for line_number, line in enumerate(path.read_text(
            encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 1:
            raise ValueError('Expected one sample ID on line {}: {}'.format(
                line_number, path))
        name = parts[0]
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Sample ID must be relative: {}'.format(name))
        if relative.suffix:
            raise ValueError('Use suffix-free sample IDs: {}'.format(name))
        ids.append(name)
    if not ids:
        raise ValueError('Empty split: {}'.format(path))
    return ids


def label_histogram(path, expected_size):
    with Image.open(path) as image:
        array = np.asarray(image)
        if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
            raise ValueError('Expected integer single-channel label: {}'.format(path))
        if expected_size and array.shape != (expected_size, expected_size):
            raise ValueError('Unexpected label size {}: {}'.format(array.shape, path))
        ids = np.unique(array)
        allowed = ((ids >= 0) & (ids <= 8)) | (ids == 255)
        if not allowed.all():
            raise ValueError('Invalid raw IDs {}: {}'.format(ids.tolist(), path))
        array = np.ascontiguousarray(array, dtype=np.uint8)
        histogram = np.bincount(array.reshape(-1), minlength=256)
        # Hash decoded pixels, not PNG compression metadata.
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        return histogram, array.shape, digest


def main():
    parser = argparse.ArgumentParser(description='只读检查V1实际安装的伪标签')
    parser.add_argument('--data-root', default='data/mydata')
    parser.add_argument('--real-list', default='train.txt')
    parser.add_argument('--combined-list', default='train_pseudo.txt')
    parser.add_argument('--val-list', default='val.txt')
    parser.add_argument('--seg-dir', default='SegmentationClass')
    parser.add_argument('--expected-count', type=int, default=500)
    parser.add_argument('--expected-size', type=int, default=1024)
    args = parser.parse_args()
    if args.expected_count <= 0 or args.expected_size < 0:
        parser.error('expected-count must be positive; expected-size must be nonnegative')

    root = Path(args.data_root)
    real = read_ids(root / args.real_list)
    combined = read_ids(root / args.combined_list)
    validation = read_ids(root / args.val_list)
    if len(real) != len(set(real)):
        raise ValueError('Real training split contains duplicate IDs')
    if set(real) - set(combined):
        raise ValueError('Combined split is missing some real training samples')
    overlap = set(combined) & set(validation)
    if overlap:
        raise ValueError('Training/validation overlap: {}'.format(sorted(overlap)[:5]))

    pseudo = sorted(set(combined) - set(real))
    if len(pseudo) != args.expected_count:
        raise ValueError('Expected {} pseudo IDs, found {}'.format(
            args.expected_count, len(pseudo)))
    frequency = Counter(combined)
    pseudo_rows = sum(frequency[name] for name in pseudo)
    print('真实样本={}，合并列表行数={}，伪标签唯一ID={}，伪样本行数={}'.format(
        len(real), len(combined), len(pseudo), pseudo_rows), flush=True)
    print('检查标签目录: {}'.format((root / args.seg_dir).resolve()), flush=True)
    if len(combined) != len(set(combined)):
        print('注意：合并列表含重复ID，请确认是否有意重复采样。', flush=True)

    hist = np.zeros(256, dtype=np.int64)
    with_ignore = 0
    all_ignore = 0
    fingerprint = hashlib.sha256()
    for index, name in enumerate(pseudo, 1):
        path = root / args.seg_dir / (name + '.png')
        sample_hist, shape, digest = label_histogram(path, args.expected_size)
        hist += sample_hist
        ignored = int(sample_hist[0] + sample_hist[255])
        with_ignore += int(ignored > 0)
        all_ignore += int(ignored == int(sample_hist.sum()))
        record = '{}\t{}\t{}\n'.format(name, shape, digest)
        fingerprint.update(record.encode('utf-8'))
        if index % 100 == 0 or index == len(pseudo):
            print('[{}/{}] 已检查'.format(index, len(pseudo)), flush=True)

    total = int(hist.sum())
    valid = int(hist[1:9].sum())
    ignored = int(hist[0] + hist[255])
    print('\n伪标签有效像素比例: {:.4%}'.format(valid / total))
    print('伪标签Ignore比例: {:.4%}'.format(ignored / total))
    print('原始ID=0比例: {:.4%}；原始ID=255比例: {:.4%}'.format(
        hist[0] / total, hist[255] / total))
    print('含Ignore的图像: {}/{}；全Ignore图像: {}'.format(
        with_ignore, len(pseudo), all_ignore))
    print('\nClass          retained_pixels  share_of_valid')
    for raw_id, class_name in enumerate(CLASSES, 1):
        print('{:<14} {:>15d} {:>14.3%}'.format(
            class_name, int(hist[raw_id]), hist[raw_id] / max(valid, 1)))
    print('\n伪标签像素指纹SHA256: {}'.format(fingerprint.hexdigest()))
    if ignored == 0:
        print('结论：这些标签保留了全部像素，没有任何像素以Ignore排除。')
    else:
        print('结论：这些标签包含Ignore；仅凭PNG无法还原阈值或生成教师。')
    print('检查完成：没有修改任何文件。')


if __name__ == '__main__':
    main()
