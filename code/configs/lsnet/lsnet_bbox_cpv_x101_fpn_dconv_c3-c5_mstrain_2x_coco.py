_base_ = './lsnet_bbox_r50_fpn_mstrain_2x_coco.py'
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='LSCPVDetector',
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
        dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        norm_eval=True,
        with_cp=True,
        style='pytorch'),
    bbox_head=dict(
        type='LSCPVHead',
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        shared_stacked_convs=1,
        first_kernel_size=3,
        kernel_size=1,
        corner_dim=64,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        norm_cfg=norm_cfg,
        conv_module_type='dcn', #norm or dcn, norm is faster
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox_init=dict(type='CrossIOULoss', loss_weight=1.0),
        loss_bbox_refine=dict(type='CrossIOULoss', loss_weight=2.0),
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            alpha=2.0,
            gamma=4.0,
            loss_weight=0.25),
        loss_offset=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
        loss_sem=dict(
            type='SEPFocalLoss',
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.1),
        _delete_=True))
# training and testing settings
train_cfg = dict(
    init=dict(
        assigner=dict(type='CentroidAssigner', scale=4, pos_num=1, iou_type='center'),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    heatmap=dict(
        assigner=dict(type='PointHMAssigner', gaussian_bump=True, gaussian_iou=0.7),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    refine=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_extreme=True),
    dict(
        type='Resize',
        img_scale=[(1333, 480), (1333, 960)],
        multiscale_mode='range',
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='LoadRPDV2Annotations'),
    dict(type='RPDV2FormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_sem_map', 'gt_sem_weights',
                               'gt_extremes']),
]
data = dict(train=dict(pipeline=train_pipeline))


########### multi-scale testing, we follow ATSS, https://github.com/sfzhang15/ATSS #############

# test_cfg = dict(method = 'vote',
#                 scale_ranges = [[96, 10000], [96, 10000], [64, 10000], [64, 10000],
#                                 [64, 10000], [0, 10000], [0, 10000], [0, 256], [0, 256],
#                                 [0, 192], [0, 192], [0, 96]])

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=[(3000, 400), (3000, 500), (3000, 600), (3000, 640), (3000, 700), (3000, 900),
#                    (3000, 1000), (3000, 1100), (3000, 1200), (3000, 1300), (3000, 1400), (3000, 1800)],
#         flip=True,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='Pad', size_divisor=32),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
# data = dict(test=dict(pipeline=test_pipeline))