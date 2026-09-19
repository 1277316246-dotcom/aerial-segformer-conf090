backend_args = None
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth'
crop_size = (
    640,
    640,
)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    pad_val=0,
    seg_pad_val=255,
    size=(
        640,
        640,
    ),
    std=[
        58.395,
        57.12,
        57.375,
    ],
    type='SegDataPreProcessor')
data_root = 'data/mydata'
dataset_type = 'PascalVOCDataset'
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False,
        interval=2000,
        max_keep_ckpts=3,
        rule='greater',
        save_best='mIoU',
        type='CheckpointHook'),
    logger=dict(interval=50, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook'))
default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_ratios = [
    0.75,
    1.0,
    1.25,
]
launcher = 'none'
load_from = 'work_dirs/segformer_b3_mydata_68_retrain/best_mIoU_iter_40000.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=False)
model = dict(
    backbone=dict(
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        drop_rate=0.0,
        embed_dims=64,
        in_channels=3,
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth',
            type='Pretrained'),
        mlp_ratio=4,
        num_heads=[
            1,
            2,
            5,
            8,
        ],
        num_layers=[
            3,
            4,
            18,
            3,
        ],
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        patch_sizes=[
            7,
            3,
            3,
            3,
        ],
        qkv_bias=True,
        sr_ratios=[
            8,
            4,
            2,
            1,
        ],
        type='MixVisionTransformer'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_val=0,
        seg_pad_val=255,
        size=(
            640,
            640,
        ),
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='SegDataPreProcessor'),
    decode_head=dict(
        align_corners=False,
        channels=256,
        dropout_ratio=0.1,
        in_channels=[
            64,
            128,
            320,
            512,
        ],
        in_index=[
            0,
            1,
            2,
            3,
        ],
        loss_decode=[
            dict(
                avg_non_ignore=True,
                loss_name='loss_ce',
                loss_weight=1.0,
                type='CrossEntropyLoss',
                use_sigmoid=False),
            dict(
                activate=True,
                eps=1.0,
                loss_name='loss_dice',
                loss_weight=0.5,
                naive_dice=True,
                type='DiceLoss',
                use_sigmoid=False),
        ],
        norm_cfg=dict(requires_grad=True, type='SyncBN'),
        num_classes=8,
        type='SegformerHead'),
    pretrained=None,
    test_cfg=dict(crop_size=(
        640,
        640,
    ), mode='slide', stride=(
        480,
        480,
    )),
    train_cfg=dict(),
    type='EncoderDecoder')
norm_cfg = dict(requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
    loss_scale='dynamic',
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=2e-5, type='AdamW', weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys=dict(
            head=dict(lr_mult=10.0),
            norm=dict(decay_mult=0.0),
            pos_block=dict(decay_mult=0.0))),
    type='AmpOptimWrapper')
optimizer = dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0005)
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1500, start_factor=1e-06,
        type='LinearLR'),
    dict(
        begin=1500,
        by_epoch=False,
        end=40000,
        eta_min=0.0,
        power=1.0,
        type='PolyLR'),
]
randomness = dict(deterministic=False, seed=3407)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        data_root='data/mydata',
        img_suffix='.png',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                backend_args=None,
                reduce_zero_label=True,
                type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        reduce_zero_label=True,
        seg_map_suffix='.png',
        test_mode=True,
        type='PascalVOCDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(backend_args=None, reduce_zero_label=True, type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]
train_cfg = dict(max_iters=40000, type='IterBasedTrainLoop', val_interval=2000)
train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file='train_pseudo.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        data_root='data/mydata',
        img_suffix='.png',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                backend_args=None,
                reduce_zero_label=True,
                type='LoadAnnotations'),
            dict(
                keep_ratio=True,
                ratio_range=(
                    0.5,
                    2.0,
                ),
                scale=(
                    1024,
                    1024,
                ),
                type='RandomResize'),
            dict(prob=0.75, type='RandomRotate90'),
            dict(
                cat_max_ratio=0.8,
                crop_size=(
                    512,
                    512,
                ),
                max_attempts=15,
                min_target_ratio=0.003,
                target_class=4,
                target_prob=0.35,
                type='RandomCropByClass'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(
                brightness_delta=32,
                contrast_range=(
                    0.6,
                    1.4,
                ),
                hue_delta=18,
                saturation_range=(
                    0.6,
                    1.4,
                ),
                type='PhotoMetricDistortion'),
            dict(
                cutout_shape=[
                    (
                        32,
                        32,
                    ),
                    (
                        64,
                        64,
                    ),
                    (
                        96,
                        96,
                    ),
                ],
                fill_in=(
                    128,
                    128,
                    128,
                ),
                n_holes=(
                    1,
                    3,
                ),
                prob=0.3,
                seg_fill_in=255,
                type='RandomCutOut'),
            dict(type='PackSegInputs'),
        ],
        reduce_zero_label=True,
        seg_map_suffix='.png',
        type='PascalVOCDataset'),
    drop_last=True,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='InfiniteSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(backend_args=None, reduce_zero_label=True, type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
        ),
        scale=(
            1024,
            1024,
        ),
        type='RandomResize'),
    dict(prob=0.75, type='RandomRotate90'),
    dict(
        cat_max_ratio=0.8,
        crop_size=(
            512,
            512,
        ),
        max_attempts=15,
        min_target_ratio=0.003,
        target_class=4,
        target_prob=0.35,
        type='RandomCropByClass'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(
        brightness_delta=32,
        contrast_range=(
            0.6,
            1.4,
        ),
        hue_delta=18,
        saturation_range=(
            0.6,
            1.4,
        ),
        type='PhotoMetricDistortion'),
    dict(
        cutout_shape=[
            (
                32,
                32,
            ),
            (
                64,
                64,
            ),
            (
                96,
                96,
            ),
        ],
        fill_in=(
            128,
            128,
            128,
        ),
        n_holes=(
            1,
            3,
        ),
        prob=0.3,
        seg_fill_in=255,
        type='RandomCutOut'),
    dict(type='PackSegInputs'),
]
tta_model = dict(type='SegTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale_factor=0.75, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.0, type='Resize'),
                dict(keep_ratio=True, scale_factor=1.25, type='Resize'),
            ],
            [
                dict(direction='horizontal', prob=0.0, type='RandomFlip'),
                dict(direction='horizontal', prob=1.0, type='RandomFlip'),
            ],
            [
                dict(reduce_zero_label=True, type='LoadAnnotations'),
            ],
            [
                dict(type='PackSegInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val.txt',
        data_prefix=dict(
            img_path='JPEGImages', seg_map_path='SegmentationClass'),
        data_root='data/mydata',
        img_suffix='.png',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                backend_args=None,
                reduce_zero_label=True,
                type='LoadAnnotations'),
            dict(type='PackSegInputs'),
        ],
        reduce_zero_label=True,
        seg_map_suffix='.png',
        test_mode=True,
        type='PascalVOCDataset'),
    num_workers=4,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    iou_metrics=[
        'mIoU',
    ], type='IoUMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='SegLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/segformer_b3_mydata_68_retrain'
