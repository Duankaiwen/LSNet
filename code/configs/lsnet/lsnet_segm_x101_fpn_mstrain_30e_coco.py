_base_ = './lsnet_segm_r50_fpn_mstrain_2x_coco.py'

lr_config = dict(step=[28, 30])
total_epochs = 30

model = dict(
    pretrained='../checkpoints/pretrained/resnext101_64x4d-ee2c6f71.pth',
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=64,
        base_width=4,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        with_cp=True,
        style='pytorch'))
