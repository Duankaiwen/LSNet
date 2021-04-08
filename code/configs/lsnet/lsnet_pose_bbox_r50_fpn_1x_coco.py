_base_ = [
    '../_base_/datasets/coco_pose.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(
    type='LSDetector',
    pretrained='../checkpoints/pretrained/resnet50-19c8e357.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5,
        norm_cfg=norm_cfg),
    bbox_head=dict(
        type='LSHead',
        task='pose_bbox',
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
        loss_bbox_init=dict(type='CrossIOULoss',   loss_weight=0.1, loss_type='bbox'),
        loss_bbox_refine=dict(type='CrossIOULoss', loss_weight=0.2, loss_type='bbox'),
        loss_pose_init=dict(type='CrossIOULoss',   loss_weight=1.0, loss_type='keypoint'),
        loss_pose_refine=dict(type='CrossIOULoss', loss_weight=2.0, loss_type='keypoint')))
# training and testing settings
train_cfg = dict(
    init=dict(
        assigner=dict(type='CentroidAssigner', scale=4, pos_num=1, iou_type='center'), 
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    refine=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False))
test_cfg = dict(
    nms_pre=100,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.6),
    max_per_img=20)
optimizer = dict(lr=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2), _delete_=True)
evaluation = dict(interval=1, metric=['bbox', 'keypoints'])