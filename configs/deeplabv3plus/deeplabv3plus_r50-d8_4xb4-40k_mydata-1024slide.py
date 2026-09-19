_base_ = './deeplabv3plus_r50-d8_4xb4-40k_mydata-512x512.py'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs'),
]

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
    ),
)

test_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
    ),
)

model = dict(
    test_cfg=dict(
        mode='slide',
        crop_size=(512, 512),
        stride=(384, 384),
    ),
)