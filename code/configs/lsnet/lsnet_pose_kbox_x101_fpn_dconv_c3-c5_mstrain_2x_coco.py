_base_ = './lsnet_pose_bbox_x101_fpn_dconv_c3-c5_mstrain_2x_coco.py'

lr_config = dict(step=[12, 20])
total_epochs = 24

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    bbox_head=dict(
        type='LSHead',
        task='pose_kbox',
        num_vectors=17,
        num_classes=1,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_kernel_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        norm_cfg=norm_cfg,
        conv_module_type='dcn', #norm or dcn, norm is faster
        loss_cls=dict(type='FocalLoss',  use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_pose_init=dict(type='CrossIOULoss',   loss_weight=1.0, loss_type='keypoint'),
        loss_pose_refine=dict(type='CrossIOULoss', loss_weight=2.0, loss_type='keypoint'),
        _delete_=True))

evaluation = dict(interval=1, metric=['keypoints'])

########### flip testing  #############

# test_cfg = dict(method = 'vote', scale_ranges = [[0, 10000]])

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=[(1333, 800)],
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