# Dataset settings
dataset_type = 'PascalVOCDataset'
data_root = 'data/mydata'
crop_size = (512, 512)
backend_args = None

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='LoadAnnotations',
        reduce_zero_label=True,
        backend_args=backend_args),
    # 扩大尺度范围，使模型同时学习小目标和大面积地物。
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    # 航拍图没有固定朝向，90度旋转配合水平翻转可覆盖八种正交变换。
    dict(type='RandomRotate90', prob=0.75),
    # 原始标签Barren=5；reduce_zero_label后训练ID为4。
    # 从0.75降到0.35，保留少数类采样，同时避免模型过预测Barren。
    dict(
        type='RandomCropByClass',
        crop_size=crop_size,
        target_class=4,
        target_prob=0.35,
        min_target_ratio=0.003,
        cat_max_ratio=0.80,
        max_attempts=15),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    # 加强不同数据源之间亮度、对比度、饱和度和色调的域泛化。
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.6, 1.4),
        saturation_range=(0.6, 1.4),
        hue_delta=18),
    # 小概率遮挡增强，遮挡区域标签设为Ignore，避免产生错误监督。
    dict(
        type='RandomCutOut',
        prob=0.30,
        n_holes=(1, 3),
        cutout_shape=[(32, 32), (64, 64), (96, 96)],
        fill_in=(128, 128, 128),
        seg_fill_in=255),
    dict(type='PackSegInputs'),
]

# 验证时保留1024x1024原始分辨率，由模型执行重叠滑窗推理。
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='LoadAnnotations',
        reduce_zero_label=True,
        backend_args=backend_args),
    dict(type='PackSegInputs'),
]

# tools/test.py --tta 使用的多尺度水平翻转测试增强。
img_ratios = [0.75, 1.0, 1.25]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=ratio, keep_ratio=True)
                for ratio in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0.0, direction='horizontal'),
                dict(type='RandomFlip', prob=1.0, direction='horizontal'),
            ],
            [dict(type='LoadAnnotations', reduce_zero_label=True)],
            [dict(type='PackSegInputs')],
        ])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        img_suffix='.png',
        seg_map_suffix='.png',
        reduce_zero_label=True,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        img_suffix='.png',
        seg_map_suffix='.png',
        reduce_zero_label=True,
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

