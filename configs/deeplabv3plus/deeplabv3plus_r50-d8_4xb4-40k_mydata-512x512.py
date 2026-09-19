_base_ = [
    '../_base_/models/segformer_mit-b0.py',
    '../_base_/datasets/pascal_voc12.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py',
]

# 按要求保留原配置文件名，但实际网络为SegFormer-B3、输入为640x640。
# 这里只加载 MiT-B3 的 ImageNet 预训练主干，不加载 LoveDA 权重。
checkpoint = (
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/'
    'segformer/mit_b3_20220624-13b1141c.pth')

crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
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
    # 训练管线裁剪512x512，数据预处理补齐到640x640；验证和提交对1024原图使用重叠滑窗，
    # 可减轻直接缩放造成的小车辆、窄道路细节损失。
    test_cfg=dict(
        mode='slide', crop_size=(640, 640), stride=(480, 480)))

# SegFormer官方方案采用AdamW；AMP降低MiT-B3的显存占用。
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW', lr=6e-5, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'head': dict(lr_mult=10.0),
        }),
    clip_grad=dict(max_norm=1.0, norm_type=2))

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

# MiT-B3显存需求高于ResNet-50，单卡先使用batch_size=2。
# 若显存不少于24 GB且实测有余量，可改为4。
train_dataloader = dict(batch_size=2, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=40000, val_interval=2000)

default_hooks = dict(
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=2000,
        save_best='mIoU',
        rule='greater',
        max_keep_ckpts=3))

randomness = dict(seed=3407, deterministic=False)


