# Experimental branch A: a low-cost high-resolution training recipe without
# changing the SegFormer-B3 architecture. It jointly tests a 768 crop, a less
# extreme resize range, and weaker CutOut; later ablation is needed if it wins.
_base_ = ['./segformer_b3_mydata_pseudo.py']

load_from = (
    'work_dirs/segformer_b3_mydata_pseudo/'
    'best_mIoU_iter_40000.pth')

crop_size = (768, 768)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    test_cfg=dict(
        mode='slide', crop_size=(768, 768), stride=(512, 512)))

# Keep the existing transforms but reduce extreme down-scaling and lower the
# chance of erasing small vehicles with CutOut.
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.75, 1.5),
        keep_ratio=True),
    dict(type='RandomRotate90', prob=0.75),
    dict(
        type='RandomCropByClass',
        crop_size=crop_size,
        target_class=4,
        target_prob=0.35,
        min_target_ratio=0.003,
        cat_max_ratio=0.80,
        max_attempts=15),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.6, 1.4),
        saturation_range=(0.6, 1.4),
        hue_delta=18),
    dict(
        type='RandomCutOut',
        prob=0.10,
        n_holes=(1, 2),
        cutout_shape=[(32, 32), (64, 64)],
        fill_in=(128, 128, 128),
        seg_fill_in=255),
    dict(type='PackSegInputs'),
]

# A 768 crop uses roughly 1.44x the image pixels of a 640 crop. Batch size one
# plus two-step gradient accumulation keeps the effective batch size at two.
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        ann_file='train_pseudo.txt',
        pipeline=train_pipeline))

optim_wrapper = dict(
    accumulative_counts=2,
    optimizer=dict(
        type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.01))

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=300),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=300,
        end=12000,
        by_epoch=False),
]

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=12000, val_interval=1000)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=1000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=3))
