# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.transforms import BaseTransform

from mmseg.registry import DATASETS, TRANSFORMS
from .basesegdataset import BaseSegDataset


@TRANSFORMS.register_module()
class RandomRotate90(BaseTransform):
    """随机将航拍图像和标签旋转90、180或270度。"""

    def __init__(self, prob=0.75):
        if not 0.0 <= prob <= 1.0:
            raise ValueError('prob必须在[0, 1]范围内')
        self.prob = prob

    def transform(self, results):
        if np.random.random() >= self.prob:
            return results

        k = int(np.random.randint(1, 4))
        results['img'] = np.ascontiguousarray(
            np.rot90(results['img'], k=k))
        results['img_shape'] = results['img'].shape[:2]

        for key in results.get('seg_fields', []):
            results[key] = np.ascontiguousarray(
                np.rot90(results[key], k=k))

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(prob={self.prob})'


@TRANSFORMS.register_module()
class RandomCropByClass(BaseTransform):
    """优先裁出包含指定类别的区域，同时保留普通随机裁剪。

    这个变换用于把参考遥感项目的“按类别像素占比挑选训练块”策略迁移到
    MMSegmentation。``target_class`` 必须使用 LoadAnnotations 处理后的训练
    ID，而不是标签 PNG 中的原始 ID。

    Args:
        crop_size (tuple[int, int]): 裁剪高度和宽度。
        target_class (int): 需要加强的训练类别 ID。
        target_prob (float): 当前图像含目标类别时，执行定向裁剪的概率。
        min_target_ratio (float): 目标类别在有效裁剪像素中的最低占比。
        cat_max_ratio (float): 任一类别在裁剪中的最大占比。
        max_attempts (int): 寻找合适裁剪框的最大次数。
        ignore_index (int): 忽略标签值。
    """

    def __init__(self,
                 crop_size,
                 target_class,
                 target_prob=0.75,
                 min_target_ratio=0.005,
                 cat_max_ratio=0.80,
                 max_attempts=20,
                 ignore_index=255):
        if len(crop_size) != 2 or min(crop_size) <= 0:
            raise ValueError('crop_size必须是两个正整数')
        if target_class < 0:
            raise ValueError('target_class不能为负数')
        if not 0.0 <= target_prob <= 1.0:
            raise ValueError('target_prob必须在[0, 1]范围内')
        if not 0.0 <= min_target_ratio <= 1.0:
            raise ValueError('min_target_ratio必须在[0, 1]范围内')
        if not 0.0 < cat_max_ratio <= 1.0:
            raise ValueError('cat_max_ratio必须在(0, 1]范围内')
        if max_attempts <= 0:
            raise ValueError('max_attempts必须为正整数')

        self.crop_size = tuple(int(value) for value in crop_size)
        self.target_class = int(target_class)
        self.target_prob = float(target_prob)
        self.min_target_ratio = float(min_target_ratio)
        self.cat_max_ratio = float(cat_max_ratio)
        self.max_attempts = int(max_attempts)
        self.ignore_index = int(ignore_index)

    def _random_bbox(self, height, width):
        crop_height, crop_width = self.crop_size
        margin_height = max(height - crop_height, 0)
        margin_width = max(width - crop_width, 0)
        y1 = int(np.random.randint(0, margin_height + 1))
        x1 = int(np.random.randint(0, margin_width + 1))
        return (y1, min(y1 + crop_height, height),
                x1, min(x1 + crop_width, width))

    def _target_bbox(self, seg_map, target_y, target_x):
        height, width = seg_map.shape[:2]
        crop_height, crop_width = self.crop_size
        max_y1 = max(height - crop_height, 0)
        max_x1 = max(width - crop_width, 0)

        low_y1 = max(0, int(target_y) - crop_height + 1)
        high_y1 = min(int(target_y), max_y1)
        low_x1 = max(0, int(target_x) - crop_width + 1)
        high_x1 = min(int(target_x), max_x1)
        y1 = int(np.random.randint(low_y1, high_y1 + 1))
        x1 = int(np.random.randint(low_x1, high_x1 + 1))
        return (y1, min(y1 + crop_height, height),
                x1, min(x1 + crop_width, width))

    @staticmethod
    def _crop_array(array, bbox):
        y1, y2, x1, x2 = bbox
        return np.ascontiguousarray(array[y1:y2, x1:x2, ...])

    def _crop_stats(self, seg_map, bbox):
        crop = self._crop_array(seg_map, bbox)
        valid = crop[crop != self.ignore_index]
        if valid.size == 0:
            return 0.0, False
        target_ratio = float(np.count_nonzero(
            valid == self.target_class) / valid.size)
        _, counts = np.unique(valid, return_counts=True)
        category_ok = float(counts.max() / valid.size) < self.cat_max_ratio
        return target_ratio, category_ok

    def _choose_bbox(self, seg_map):
        height, width = seg_map.shape[:2]
        target_points = np.argwhere(seg_map == self.target_class)

        if target_points.size and np.random.random() < self.target_prob:
            best_bbox = None
            best_ratio = -1.0
            for _ in range(self.max_attempts):
                target_y, target_x = target_points[
                    np.random.randint(len(target_points))]
                bbox = self._target_bbox(seg_map, target_y, target_x)
                target_ratio, category_ok = self._crop_stats(seg_map, bbox)
                if target_ratio > best_ratio:
                    best_bbox = bbox
                    best_ratio = target_ratio
                if target_ratio >= self.min_target_ratio and category_ok:
                    return bbox

            # 目标类别优先：即使没有同时满足cat_max_ratio，也返回目标像素
            # 占比最高的候选框，避免再次把Barren裁掉。
            if best_bbox is not None and best_ratio >= self.min_target_ratio:
                return best_bbox

        bbox = self._random_bbox(height, width)
        for _ in range(self.max_attempts - 1):
            _, category_ok = self._crop_stats(seg_map, bbox)
            if category_ok:
                break
            bbox = self._random_bbox(height, width)
        return bbox

    def transform(self, results):
        if 'gt_seg_map' not in results:
            raise KeyError('RandomCropByClass必须放在LoadAnnotations之后')

        bbox = self._choose_bbox(results['gt_seg_map'])
        results['img'] = self._crop_array(results['img'], bbox)
        results['img_shape'] = results['img'].shape[:2]
        for key in results.get('seg_fields', []):
            results[key] = self._crop_array(results[key], bbox)
        return results

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'crop_size={self.crop_size}, '
            f'target_class={self.target_class}, '
            f'target_prob={self.target_prob}, '
            f'min_target_ratio={self.min_target_ratio}, '
            f'cat_max_ratio={self.cat_max_ratio}, '
            f'max_attempts={self.max_attempts})')


@DATASETS.register_module()
class PascalVOCDataset(BaseSegDataset):
    """无人机低空航拍语义分割数据集。

    原始ID 0为Ignore；原始ID 1-8为八个评分类别。使用
    reduce_zero_label=True后，训练标签为0-7，Ignore标签为255。
    """

    METAINFO = dict(
        classes=(
            'Background',
            'Building',
            'Road',
            'Water',
            'Barren',
            'Vegetation',
            'Agricultural',
            'Vehicle',
        ),
        palette=[
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
        ],
    )

    def __init__(self,
                 ann_file='',
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=True,
                 **kwargs):
        super().__init__(
            ann_file=ann_file,
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)

