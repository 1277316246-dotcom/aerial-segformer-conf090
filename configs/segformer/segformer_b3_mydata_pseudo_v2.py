# 第二轮伪标签自训练。
# 该配置直接继承已验证有效的第一轮伪标签配置，避免改变任何网络或数据增强。
_base_ = ['./segformer_b3_mydata_pseudo.py']

# 从第一轮最佳模型继续微调，而非从ImageNet或68.4分初始模型重新训练。
load_from = (
    'work_dirs/segformer_b3_mydata_pseudo/'
    'best_mIoU_iter_40000.pth')

# train_pseudo_v2_x3.txt = 真标签6296张 + V2伪标签500张重复3次。
# 伪样本图片采样比例约为19.2%，兼顾测试域适应与伪标签噪声控制。
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(ann_file='train_pseudo_v2_x3.txt'))

# 第二轮是短程低学习率微调，避免破坏第一轮已收敛的判别边界。
optim_wrapper = dict(
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
