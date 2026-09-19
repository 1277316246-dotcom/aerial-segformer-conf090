# Experimental branch B: keep the final 640 pipeline and only replace the
# decode head with source-aware scale attention plus CBAM attention.
_base_ = ['./segformer_b3_mydata_pseudo.py']

custom_imports = dict(
    imports=['mmseg.models.decode_heads.attention_segformer_head'],
    allow_failed_imports=False)

# Initialize all compatible backbone/head parameters from the current final
# model. New attention gates start as identity mappings.
load_from = (
    'work_dirs/segformer_b3_mydata_pseudo/'
    'best_mIoU_iter_40000.pth')

model = dict(
    decode_head=dict(
        type='AttentionSegformerHead',
        attention_reduction=16,
        spatial_kernel_size=7))

# Use the original high-confidence pseudo-label list, not later experimental
# pseudo-label lists.
train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(ann_file='train_pseudo.txt'))

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