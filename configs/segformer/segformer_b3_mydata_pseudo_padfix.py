# Copyright (c) OpenMMLab. All rights reserved.
"""SegFormer-B3 + V1伪标签 + 训练尺寸填充修复。

主要设置：
1. 使用SegFormer-B3；
2. 使用train_pseudo.txt进行真标签和V1伪标签联合训练；
3. 主损失为CE + Dice；
4. 训练裁剪为512x512；
5. 数据预处理固定填充尺寸同步为512x512；
6. 验证和提交使用640x640重叠滑窗；
7. 支持0.75、1.0、1.25尺度和水平翻转TTA；
8. 不使用辅助头、EMA、注意力头和LoveDA预训练。
"""

_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

# -----------------------------------------------------------------------------
# 数据集
# -----------------------------------------------------------------------------
dataset_type = 'PascalVOCDataset'
data_root = 'data/mydata'
train_crop_size = (512, 512)
backend_args = None

train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=backend_args),
    dict(
        type='LoadAnnotations',
        reduce_zero_label=True,
        backend_args=backend_args),

    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),

    # 航拍图像没有固定方向。
    dict(
        type='RandomRotate90',
        prob=0.75),

    # 原始Barren标签ID为5；
    # reduce_zero_label=True后，训练ID为4。
    dict(
        type='RandomCropByClass',
        crop_size=train_crop_size,
        target_class=4,
        target_prob=0.35,
        min_target_ratio=0.003,
        cat_max_ratio=0.80,
        max_attempts=15),

    dict(
        type='RandomFlip',
        prob=0.5,
        direction='horizontal'),

    # 用于缓解不同数据来源之间的颜色域偏移。
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.6, 1.4),
        saturation_range=(0.6, 1.4),
        hue_delta=18),

    # 保留V1方案原始设置，方便进行单变量对照实验。
    dict(
        type='RandomCutOut',
        prob=0.30,
        n_holes=(1, 3),
        cutout_shape=[
            (32, 32),
            (64, 64),
            (96, 96),
        ],
        fill_in=(128, 128, 128),
        seg_fill_in=255),

    dict(type='PackSegInputs'),
]

# 验证时保留1024x1024原始分辨率，
# 由模型执行640x640重叠滑窗推理。
test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=backend_args),
    dict(
        type='LoadAnnotations',
        reduce_zero_label=True,
        backend_args=backend_args),
    dict(type='PackSegInputs'),
]

# 3个尺度 × 水平翻转/不翻转 = 6种TTA。
img_ratios = [0.75, 1.0, 1.25]

tta_model = dict(type='SegTTAModel')

tta_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=backend_args),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(
                    type='Resize',
                    scale_factor=ratio,
                    keep_ratio=True)
                for ratio in img_ratios
            ],
            [
                dict(
                    type='RandomFlip',
                    prob=0.0,
                    direction='horizontal'),
                dict(
                    type='RandomFlip',
                    prob=1.0,
                    direction='horizontal'),
            ],
            [
                dict(
                    type='LoadAnnotations',
                    reduce_zero_label=True,
                    backend_args=backend_args),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ])
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(
        type='InfiniteSampler',
        shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,

        # 真标签与V1伪标签联合训练列表。
        ann_file='train_pseudo.txt',

        data_prefix=dict(
            img_path='JPEGImages',
            seg_map_path='SegmentationClass'),

        img_suffix='.png',
        seg_map_suffix='.png',
        reduce_zero_label=True,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.txt',

        data_prefix=dict(
            img_path='JPEGImages',
            seg_map_path='SegmentationClass'),

        img_suffix='.png',
        seg_map_suffix='.png',
        reduce_zero_label=True,
        test_mode=True,
        pipeline=test_pipeline))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='IoUMetric',
    iou_metrics=['mIoU'])

test_evaluator = val_evaluator

# -----------------------------------------------------------------------------
# SegFormer-B3
# -----------------------------------------------------------------------------
checkpoint = (
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/'
    'segformer/mit_b3_20220624-13b1141c.pth'
)

# 关键修复：
# 训练裁剪是512x512，因此训练期固定填充尺寸也必须是512x512。
# 不再把512训练样本额外填充成640。
data_preprocessor = dict(
    size=train_crop_size)

model = dict(
    data_preprocessor=data_preprocessor,

    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint),
        embed_dims=64,
        num_heads=[1, 2, 5, 8],
        num_layers=[3, 4, 18, 3],
        drop_path_rate=0.1),

    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        channels=256,
        num_classes=8,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                avg_non_ignore=True,
                loss_name='loss_ce',
                loss_weight=1.0),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                activate=True,
                naive_dice=True,
                eps=1.0,
                loss_name='loss_dice',
                loss_weight=0.5),
        ]),

    # 对1024x1024原图进行重叠滑窗预测。
    test_cfg=dict(
        mode='slide',
        crop_size=(640, 640),
        stride=(480, 480)))

# -----------------------------------------------------------------------------
# 优化器
# -----------------------------------------------------------------------------
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale='dynamic',

    optimizer=dict(
        type='AdamW',
        lr=2e-5,
        betas=(0.9, 0.999),
        weight_decay=0.01),

    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(
                decay_mult=0.0),
            'norm': dict(
                decay_mult=0.0),
            'head': dict(
                lr_mult=10.0),
        }),

    clip_grad=dict(
        max_norm=1.0,
        norm_type=2))

# -----------------------------------------------------------------------------
# 学习率
# -----------------------------------------------------------------------------
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500),

    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=40000,
        by_epoch=False),
]

# -----------------------------------------------------------------------------
# 训练、验证和测试
# -----------------------------------------------------------------------------
train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=40000,
    val_interval=2000)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    logger=dict(
        type='LoggerHook',
        interval=50,
        log_metric_by_epoch=False),

    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=1))

# 通过训练命令传入教师权重，避免写死路径。
load_from = None
resume = False

randomness = dict(
    seed=3407,
    deterministic=False)