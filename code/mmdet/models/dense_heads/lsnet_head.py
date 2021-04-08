import pdb
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init, kaiming_init

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, 
                        multiclass_nms_lsvr, unmap)
from mmdet.ops import DeformConv, PyramidDeformConv, DeformConvPack, ModulatedDeformConvPack
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead

@HEADS.register_module()
class LSHead(AnchorFreeHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 point_feat_channels=256,
                 num_kernel_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 task = 'bbox',
                 num_vectors = 4,
                 conv_module_type= 'norm', #norm of dcn, norm is faster
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init=dict(type='CrossIOULoss', loss_weight=1.0),
                 loss_bbox_refine=dict(type='CrossIOULoss', loss_weight=2.0),
                 loss_segm_init = None,
                 loss_segm_refine = None,
                 loss_pose_init = None,
                 loss_pose_refine = None,
                 **kwargs):
        self.task = task
        self.num_vectors = num_vectors
        self.num_kernel_points = num_kernel_points
        self.point_feat_channels = point_feat_channels
        self.conv_module_type = conv_module_type

        # we use deformable conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_kernel_points))
        self.dcn_pad    = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_kernel_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.gradient_mul     = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides    = point_strides
        self.fpn_levels       = [i for i in range(len(self.point_strides))]
        self.point_generators = [PointGenerator() for _ in self.point_strides]

        if self.train_cfg:
            self.init_assigner = build_assigner(self.train_cfg.init.assigner)
            self.refine_assigner = build_assigner(self.train_cfg.refine.assigner)
            # use PseudoSampler when sampling is False
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.cls_out_channels = self.num_classes

        if self.task == 'bbox':
            self.loss_bbox_init = build_loss(loss_bbox_init)
            self.loss_bbox_refine = build_loss(loss_bbox_refine)
        elif self.task == 'segm':
            self.loss_segm_init = build_loss(loss_segm_init)
            self.loss_segm_refine = build_loss(loss_segm_refine)
        elif self.task == 'pose_bbox':
            self.loss_bbox_init = build_loss(loss_bbox_init)
            self.loss_bbox_refine = build_loss(loss_bbox_refine)
            self.loss_pose_init = build_loss(loss_pose_init)
            self.loss_pose_refine = build_loss(loss_pose_refine)
        elif self.task == 'pose_kbox':
            self.loss_pose_init = build_loss(loss_pose_init)
            self.loss_pose_refine = build_loss(loss_pose_refine)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()
        self.cls_GN = nn.GroupNorm(self.norm_cfg.num_groups, self.feat_channels)
        self.cls_convs = nn.ModuleList()

        if self.task == 'bbox':
            self.bbox_GN = nn.GroupNorm(self.norm_cfg.num_groups, self.feat_channels)
            self.bbox_convs = nn.ModuleList()
        elif self.task == 'segm':
            self.segm_GN = nn.GroupNorm(self.norm_cfg.num_groups, self.feat_channels)
            self.segm_convs = nn.ModuleList()
        elif self.task == 'pose_bbox':
            self.bbox_GN = nn.GroupNorm(self.norm_cfg.num_groups, self.feat_channels)
            self.bbox_convs = nn.ModuleList()
            self.pose_GN = nn.GroupNorm(self.norm_cfg.num_groups, self.feat_channels)
            self.pose_convs = nn.ModuleList()
        elif self.task == 'pose_kbox':
            self.pose_GN = nn.GroupNorm(self.norm_cfg.num_groups, self.feat_channels)
            self.pose_convs = nn.ModuleList()
        
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.conv_module_type == 'norm':
                self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                                  conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))                                
            else: #dcn
                self.cls_convs.append(DCNConvModule(chn, self.feat_channels, self.dcn_kernel, 1,
                                                   self.norm_cfg.num_groups, self.dcn_pad))
                
            if self.task == 'bbox':
                if self.conv_module_type == 'norm':
                    self.bbox_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                                    conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))                                
                else: #dcn
                    self.bbox_convs.append(DCNConvModule(chn, self.feat_channels, self.dcn_kernel, 1,
                                                   self.norm_cfg.num_groups, self.dcn_pad))
            elif self.task == 'segm':
                if self.conv_module_type == 'norm':
                    self.segm_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                                    conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))                                
                else: #dcn
                    self.segm_convs.append(DCNConvModule(chn, self.feat_channels, self.dcn_kernel, 1,
                                                   self.norm_cfg.num_groups, self.dcn_pad))
            elif self.task == 'pose_bbox':
                if self.conv_module_type == 'norm':
                    self.bbox_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                                    conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))     
                    self.pose_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                                    conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))                                
                else: #dcn
                    self.bbox_convs.append(DCNConvModule(chn, self.feat_channels, self.dcn_kernel, 1,
                                                   self.norm_cfg.num_groups, self.dcn_pad)) 
                    self.pose_convs.append(DCNConvModule(chn, self.feat_channels, self.dcn_kernel, 1,
                                                   self.norm_cfg.num_groups, self.dcn_pad)) 
            elif self.task == 'pose_kbox':
                if self.conv_module_type == 'norm':    
                    self.pose_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                                    conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))                                
                else: #dcn
                    self.pose_convs.append(DCNConvModule(chn, self.feat_channels, self.dcn_kernel, 1,
                                                   self.norm_cfg.num_groups, self.dcn_pad)) 

        
        self.pts_cls_conv = PyramidDeformConv(self.feat_channels, self.point_feat_channels,
                                              self.dcn_kernel, 1, self.dcn_pad)
        self.pts_cls_out = nn.Conv2d(self.point_feat_channels, self.cls_out_channels, 1, 1, 0)

        self.cls_af_dcn_conv = nn.Sequential(
                                nn.Conv2d(3 * self.point_feat_channels,
                                            self.point_feat_channels,
                                            1, 1, 0),
                                nn.ReLU())
        self.cls_feat_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)

        if self.task == 'bbox':
            bbox_out_dim = 4*(self.num_vectors+1) + (self.num_kernel_points-self.num_vectors-1)*2
            self.pts_bbox_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
            self.pts_bbox_init_out = nn.Conv2d(self.point_feat_channels, bbox_out_dim, 1, 1, 0)

            self.pts_bbox_refine_conv = PyramidDeformConv(self.feat_channels, self.point_feat_channels,
                                                          self.dcn_kernel, 1, self.dcn_pad)
            self.pts_bbox_refine_out = nn.Conv2d(self.point_feat_channels, 4*(self.num_vectors+1), 1, 1, 0)

            self.bbox_af_dcn_conv = nn.Sequential(
                                nn.Conv2d(3 * self.point_feat_channels,
                                            self.point_feat_channels,
                                            1, 1, 0),
                                nn.ReLU())

            self.bbox_feat_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)

        elif self.task == 'segm':
            segm_out_dim = (self.num_vectors+1)*4

            self.pts_segm_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
            self.pts_segm_init_out = nn.Conv2d(self.point_feat_channels, segm_out_dim, 1, 1, 0)

            self.pts_segm_refine_conv = PyramidDeformConv(self.feat_channels, 
                                                          self.point_feat_channels,
                                                          self.dcn_kernel, 1, self.dcn_pad)
            self.pts_segm_refine_out = nn.Conv2d(self.point_feat_channels, segm_out_dim, 1, 1, 0)


            self.segm_af_dcn_conv = nn.Sequential(
                                nn.Conv2d(3 * self.point_feat_channels,
                                            self.point_feat_channels,
                                            1, 1, 0),
                                nn.ReLU())

            self.segm_feat_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)    
        elif self.task == 'pose_bbox':
            self.pts_bbox_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
            self.pts_bbox_init_out = nn.Conv2d(self.point_feat_channels, 28, 1, 1, 0)

            self.pts_bbox_refine_conv = PyramidDeformConv(self.feat_channels, self.point_feat_channels,
                                                          self.dcn_kernel, 1, self.dcn_pad)
            self.pts_bbox_refine_out = nn.Conv2d(self.point_feat_channels, 20, 1, 1, 0)

            self.bbox_af_dcn_conv = nn.Sequential(
                                nn.Conv2d(3 * self.point_feat_channels,
                                            self.point_feat_channels,
                                            1, 1, 0),
                                nn.ReLU())

            self.bbox_feat_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)

            pose_out_dim = (self.num_vectors+1)*4

            self.pts_pose_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
            self.pts_pose_init_out = nn.Conv2d(self.point_feat_channels, pose_out_dim, 1, 1, 0)

            self.pts_pose_refine_conv = PyramidDeformConv(self.feat_channels, 
                                                          self.point_feat_channels,
                                                          self.dcn_kernel, 1, self.dcn_pad)
            self.pts_pose_refine_out = nn.Conv2d(self.point_feat_channels, pose_out_dim, 1, 1, 0)


            self.pose_af_dcn_conv = nn.Sequential(
                                    nn.Conv2d(3 * self.point_feat_channels,
                                                self.point_feat_channels,
                                                1, 1, 0),
                                    nn.ReLU())

            self.pose_feat_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
        elif self.task == 'pose_kbox':
            pose_out_dim = (self.num_vectors+1)*4

            self.pts_pose_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
            self.pts_pose_init_out = nn.Conv2d(self.point_feat_channels, pose_out_dim, 1, 1, 0)

            self.pts_pose_refine_conv = PyramidDeformConv(self.feat_channels, 
                                                          self.point_feat_channels,
                                                          self.dcn_kernel, 1, self.dcn_pad)
            self.pts_pose_refine_out = nn.Conv2d(self.point_feat_channels, pose_out_dim, 1, 1, 0)


            self.pose_af_dcn_conv = nn.Sequential(
                                    nn.Conv2d(3 * self.point_feat_channels,
                                                self.point_feat_channels,
                                                1, 1, 0),
                                    nn.ReLU())

            self.pose_feat_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        if self.task == 'bbox':
            for m in self.bbox_convs:
                normal_init(m.conv, std=0.01)
        elif self.task == 'segm':
            for m in self.segm_convs:
                normal_init(m.conv, std=0.01)
        elif self.task == 'pose_bbox':
            for m in self.bbox_convs:
                normal_init(m.conv, std=0.01)

            for m in self.pose_convs:
                normal_init(m.conv, std=0.01)
        elif self.task == 'pose_kbox':
            for m in self.pose_convs:
                normal_init(m.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        kaiming_init(self.pts_cls_conv)
        normal_init(self.pts_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.cls_feat_conv, std=0.01)
        normal_init(self.cls_af_dcn_conv[0], std=0.01)

        if self.task == 'bbox':
            normal_init(self.pts_bbox_init_conv, std=0.01)
            normal_init(self.pts_bbox_init_out, std=0.01)
            kaiming_init(self.pts_bbox_refine_conv)
            normal_init(self.pts_bbox_refine_out, std=0.01)
            normal_init(self.bbox_feat_conv, std=0.01)
            normal_init(self.bbox_af_dcn_conv[0], std=0.01)
        elif self.task == 'segm':
            normal_init(self.pts_segm_init_conv, std=0.01)
            normal_init(self.pts_segm_init_out, std=0.01)
            kaiming_init(self.pts_segm_refine_conv)
            normal_init(self.pts_segm_refine_out, std=0.01)
            normal_init(self.segm_feat_conv, std=0.01)
            normal_init(self.segm_af_dcn_conv[0], std=0.01)
        elif self.task == 'pose_bbox':
            normal_init(self.pts_bbox_init_conv, std=0.01)
            normal_init(self.pts_bbox_init_out, std=0.01)
            kaiming_init(self.pts_bbox_refine_conv)
            normal_init(self.pts_bbox_refine_out, std=0.01)
            normal_init(self.bbox_feat_conv, std=0.01)
            normal_init(self.bbox_af_dcn_conv[0], std=0.01)

            normal_init(self.pts_pose_init_conv, std=0.01)
            normal_init(self.pts_pose_init_out, std=0.01)
            kaiming_init(self.pts_pose_refine_conv)
            normal_init(self.pts_pose_refine_out, std=0.01)
            normal_init(self.pose_feat_conv, std=0.01)
            normal_init(self.pose_af_dcn_conv[0], std=0.01)
        elif self.task == 'pose_kbox':
            normal_init(self.pts_pose_init_conv, std=0.01)
            normal_init(self.pts_pose_init_out, std=0.01)
            kaiming_init(self.pts_pose_refine_conv)
            normal_init(self.pts_pose_refine_out, std=0.01)
            normal_init(self.pose_feat_conv, std=0.01)
            normal_init(self.pose_af_dcn_conv[0], std=0.01)

    def extreme_points2bbox(self, pts, y_first=True, extreme=False):
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        valid_pts, inds = torch.max(pts_reshape, dim=2)
        neg_inds = inds == 0
        valid_pts[neg_inds] *= -1
        valid_pts_xy = valid_pts.view(valid_pts.shape[0], -1, 2, *valid_pts.shape[2:])

        pts_y = valid_pts_xy[:, :, 0, ...] if y_first else valid_pts_xy[:, :, 1, ...]
        pts_x = valid_pts_xy[:, :, 1, ...] if y_first else valid_pts_xy[:, :, 0, ...]

        bbox_left   = pts_x[:, 1, :, :].unsqueeze(1)
        bbox_right  = pts_x[:, 3, :, :].unsqueeze(1)
        bbox_up     = pts_y[:, 0, :, :].unsqueeze(1)
        bbox_bottom = pts_y[:, 2, :, :].unsqueeze(1)

        bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)

        if extreme:
            extreme_up     = torch.cat((pts_x[:, 0:1, ...], pts_y[:, 0:1, ...]), dim = 1)
            extreme_left   = torch.cat((pts_x[:, 1:2, ...], pts_y[:, 1:2, ...]), dim = 1)
            extreme_bottom = torch.cat((pts_x[:, 2:3, ...], pts_y[:, 2:3, ...]), dim = 1)
            extreme_right  = torch.cat((pts_x[:, 3:4, ...], pts_y[:, 3:4, ...]), dim = 1)
            extremes       = torch.cat([extreme_up, extreme_left, extreme_bottom, extreme_right], 
                                       dim=1)
            return extremes, bbox
        else:
            return bbox
    
    def vectors2bbox(self, pts, y_first=True, vector=False):
        pts_reshape = pts[:,:-4,...].view(pts.shape[0], -1, 2, *pts.shape[2:])
        valid_pts, inds = torch.max(pts_reshape, dim=2)
        neg_inds = inds == 0
        valid_pts[neg_inds] *= -1
        valid_pts_xy = valid_pts.view(valid_pts.shape[0], -1, 2, *valid_pts.shape[2:])

        pts_y = valid_pts_xy[:, :, 0, ...] if y_first else valid_pts_xy[:, :, 1, ...]
        pts_x = valid_pts_xy[:, :, 1, ...] if y_first else valid_pts_xy[:, :, 0, ...]

        vectors_xmin = pts_x.min(1)[0]
        vectors_ymin = pts_y.min(1)[0]
        vectors_xmax = pts_x.max(1)[0]
        vectors_ymax = pts_y.max(1)[0]

        bbox = torch.stack([vectors_xmin, vectors_ymin, vectors_xmax, vectors_ymax], 1)

        if vector:
            vectors = torch.stack([pts_x, pts_y], 2).reshape(pts_y.shape[0], -1, *pts_y.shape[2:])
            return vectors, bbox
        else:
            return bbox

    def get_pred_reg(self, raw_reg1, raw_reg2):
        if raw_reg2 is not None:
            raw_reg_reshape = raw_reg1.view(raw_reg1.shape[0], -1, 2, *raw_reg1.shape[2:])
            pos_reg, inds = torch.max(raw_reg_reshape, dim=2)
            neg_inds = inds == 0
            pos_reg[neg_inds] *= -1

            reg_for_dcn = torch.cat((pos_reg, raw_reg2), dim =1)
            return reg_for_dcn
        else:
            raw_reg_reshape = raw_reg1.view(raw_reg1.shape[0], -1, 4, *raw_reg1.shape[2:])
            raw_reg_cts = raw_reg_reshape[:, -1:, ...]
            raw_reg_polys = raw_reg_reshape[:, :-1, ...]

            if self.task == 'segm':
                kernel_stride = math.ceil(self.num_vectors/(self.num_kernel_points-1))
                raw_reg_poly_offs = raw_reg_polys[:, ::kernel_stride, ...]
            elif 'pose' in self.task:
                kernel_stride = 2
                raw_reg_poly_offs = raw_reg_polys[:, 1::kernel_stride, ...]

            raw_reg_offsets = torch.cat([raw_reg_poly_offs, raw_reg_cts], dim=1)
            raw_reg_offsets_reshape = raw_reg_offsets.reshape(raw_reg_offsets.shape[0], -1, 2,
                                                              *raw_reg_offsets.shape[3:])

            reg_for_dcn, inds = torch.max(raw_reg_offsets_reshape, dim = 2)
            neg_inds = inds==0
            reg_for_dcn[neg_inds] *= -1
            return reg_for_dcn

    def get_bbox_gt_reg(self, gt_pts, anchor_pts, bbox_weights):
        gt_reg = gt_pts.new_zeros([gt_pts.size(0), 20])
        anchor_pts_repeat = anchor_pts[:, :2].repeat(1, 5)
        offset_reg = gt_pts - anchor_pts_repeat
        br_reg = offset_reg >= 0 
        tl_reg = offset_reg < 0
        tlbr_inds = torch.stack([tl_reg, br_reg], -1).reshape(-1, 20)
        gt_reg[tlbr_inds] = torch.abs(offset_reg.reshape(-1))

        pos_inds = bbox_weights[:,0]>0
        neg_inds = bbox_weights[:,0]==0
        gt_reg[neg_inds] = 0

        xl_reg = gt_reg[..., 0::4]
        xr_reg = gt_reg[..., 1::4]
        yt_reg = gt_reg[..., 2::4]
        yb_reg = gt_reg[..., 3::4]
        yx_gt_reg = torch.stack([yt_reg, yb_reg, xl_reg, xr_reg], -1).reshape(-1, 20)

        xl_inds = tlbr_inds[..., 0::4]
        xr_inds = tlbr_inds[..., 1::4]
        yt_inds = tlbr_inds[..., 2::4]
        yb_inds = tlbr_inds[..., 3::4]
        yx_inds = torch.stack([yt_inds, yb_inds, xl_inds, xr_inds], -1).reshape(-1, 20)

        return yx_gt_reg, yx_inds

    def get_poly_gt_reg(self, gt_pts, anchor_pts, bbox_weights):
        gt_reg = gt_pts.new_zeros([gt_pts.size(0), gt_pts.size(1)*2])
        anchor_pts_repeat = anchor_pts[:, :2].repeat(1, self.num_vectors+1)
        offset_reg = gt_pts - anchor_pts_repeat
        br_reg = offset_reg >= 0 
        tl_reg = offset_reg < 0
        tlbr_inds = torch.stack([tl_reg, br_reg], -1).reshape(-1, gt_pts.size(1)*2)
        gt_reg[tlbr_inds] = torch.abs(offset_reg.reshape(-1))

        pos_inds = bbox_weights[:,0]>0
        neg_inds = bbox_weights[:,0]==0
        gt_reg[neg_inds] = 0

        xl_reg = gt_reg[..., 0::4]
        xr_reg = gt_reg[..., 1::4]
        yt_reg = gt_reg[..., 2::4]
        yb_reg = gt_reg[..., 3::4]
        yx_gt_reg = torch.stack([yt_reg, yb_reg, xl_reg, xr_reg], -1).reshape(-1, gt_pts.size(1)*2)

        xl_inds = tlbr_inds[..., 0::4]
        xr_inds = tlbr_inds[..., 1::4]
        yt_inds = tlbr_inds[..., 2::4]
        yb_inds = tlbr_inds[..., 3::4]
        yx_inds = torch.stack([yt_inds, yb_inds, xl_inds, xr_inds], -1).reshape(-1, gt_pts.size(1)*2)

        return yx_gt_reg, yx_inds

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_extremes = None,
                      gt_keypoints = None,
                      gt_masks = None,
                      gt_labels = None,
                      gt_bboxes_ignore=None,
                      proposal_cfg = None,
                      **kwargs):
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, gt_extremes, gt_keypoints, gt_masks, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_extremes, gt_keypoints, gt_masks, gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs, img_metas, cfg=proposal_cfg)
            return losses, proposal_list

    def forward(self, feats):
        (cls_feats, bbox_feats, bbox_reg_init_sps, bbox_dcn_offsets,
         segm_feats, segm_reg_init_sps, segm_dcn_offsets,
         pose_feats, pose_reg_init_sps, pose_dcn_offsets) = multi_apply(self.forward_single1, feats)

        (pts_cls_outs, bbox_reg_refine_sps, segm_reg_refine_sps, pose_reg_refine_sps
        ) = multi_apply(self.forward_single2,
                        bbox_dcn_offsets,
                        bbox_reg_init_sps,
                        segm_dcn_offsets,
                        segm_reg_init_sps,
                        pose_dcn_offsets,
                        pose_reg_init_sps,
                        self.fpn_levels,
                        cls_feats = cls_feats,
                        bbox_feats = bbox_feats,
                        segm_feats = segm_feats,
                        pose_feats = pose_feats,
                        num_levels = len(self.fpn_levels))

        return (pts_cls_outs, bbox_reg_init_sps, bbox_reg_refine_sps, 
                segm_reg_init_sps, segm_reg_refine_sps, pose_reg_init_sps, pose_reg_refine_sps)

    def forward_single1(self, x):
        """ Forward feature map of a single FPN level."""
        dcn_base_offset = self.dcn_base_offset.type_as(x)

        cls_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)

        if self.task == 'bbox':
            bbox_feat = x
            for bbox_conv in self.bbox_convs:
                 bbox_feat = bbox_conv(bbox_feat)

            bbox_reg_init_out = self.pts_bbox_init_out(
                        self.relu(self.pts_bbox_init_conv(bbox_feat)))

            bbox_reg_init_sp = self.softplus(bbox_reg_init_out[:, :4*(self.num_vectors+1), ...])
            bbox_pred_reg = self.get_pred_reg(bbox_reg_init_sp, 
                                              bbox_reg_init_out[:, 4*(self.num_vectors+1):, ...])

            bbox_pred_reg_grad_mul = (1 - self.gradient_mul)*bbox_pred_reg.detach(
                                    ) + self.gradient_mul * bbox_pred_reg

            bbox_dcn_offset = bbox_pred_reg_grad_mul - dcn_base_offset

            return (cls_feat, bbox_feat, bbox_reg_init_sp, bbox_dcn_offset, None, None, None,
                    None, None, None)

        elif self.task == 'segm':
            segm_feat = x
            for segm_conv in self.segm_convs:
                 segm_feat = segm_conv(segm_feat)

            segm_reg_init_out = self.pts_segm_init_out(
                        self.relu(self.pts_segm_init_conv(segm_feat)))

            segm_reg_init_sp = self.softplus(segm_reg_init_out)
            segm_pred_reg = self.get_pred_reg(segm_reg_init_sp, None)

            segm_pred_reg_grad_mul = (1 - self.gradient_mul)*segm_pred_reg.detach(
                                    ) + self.gradient_mul * segm_pred_reg

            segm_dcn_offset = segm_pred_reg_grad_mul - dcn_base_offset

            return (cls_feat, None, None, None, segm_feat, segm_reg_init_sp, segm_dcn_offset,
                    None, None, None)
        
        elif self.task == 'pose_bbox':
            bbox_feat = x
            pose_feat = x
            for bbox_conv in self.bbox_convs:
                 bbox_feat = bbox_conv(bbox_feat)
            for pose_conv in self.pose_convs:
                 pose_feat = pose_conv(pose_feat)

            bbox_reg_init_out = self.pts_bbox_init_out(
                        self.relu(self.pts_bbox_init_conv(bbox_feat)))

            bbox_reg_init_sp = self.softplus(bbox_reg_init_out[:, :20, ...])
            bbox_pred_reg = self.get_pred_reg(bbox_reg_init_sp, bbox_reg_init_out[:, 20:, ...])

            bbox_pred_reg_grad_mul = (1 - self.gradient_mul)*bbox_pred_reg.detach(
                                    ) + self.gradient_mul * bbox_pred_reg

            bbox_dcn_offset = bbox_pred_reg_grad_mul - dcn_base_offset

            pose_reg_init_out = self.pts_pose_init_out(
                        self.relu(self.pts_pose_init_conv(pose_feat)))

            pose_reg_init_sp = self.softplus(pose_reg_init_out)
            pose_pred_reg = self.get_pred_reg(pose_reg_init_sp, None)

            pose_pred_reg_grad_mul = (1 - self.gradient_mul)*pose_pred_reg.detach(
                                    ) + self.gradient_mul * pose_pred_reg

            pose_dcn_offset = pose_pred_reg_grad_mul - dcn_base_offset

            return (cls_feat, bbox_feat, bbox_reg_init_sp, bbox_dcn_offset, None, None, None,
                    pose_feat, pose_reg_init_sp, pose_dcn_offset)
        elif self.task == 'pose_kbox':
            pose_feat = x
            for pose_conv in self.pose_convs:
                 pose_feat = pose_conv(pose_feat)

            pose_reg_init_out = self.pts_pose_init_out(
                        self.relu(self.pts_pose_init_conv(pose_feat)))

            pose_reg_init_sp = self.softplus(pose_reg_init_out)
            pose_pred_reg = self.get_pred_reg(pose_reg_init_sp, None)

            pose_pred_reg_grad_mul = (1 - self.gradient_mul)*pose_pred_reg.detach(
                                    ) + self.gradient_mul * pose_pred_reg

            pose_dcn_offset = pose_pred_reg_grad_mul - dcn_base_offset

            return (cls_feat, None, None, None, None, None, None,
                    pose_feat, pose_reg_init_sp, pose_dcn_offset)

    def forward_single2(self, bbox_dcn_offset, bbox_reg_init_sp, segm_dcn_offset, segm_reg_init_sp, 
                        pose_dcn_offset, pose_reg_init_sp, fpn_level, cls_feats, bbox_feats, segm_feats, 
                        pose_feats, num_levels):
        level_list = []
        level_list.append(fpn_level)
        if fpn_level == 0:
            level_list.append(fpn_level+1)
            level_list.append(fpn_level+2)
        elif fpn_level == num_levels-1:
            level_list.append(fpn_level-1)
            level_list.append(fpn_level-2)
        else:
            level_list.append(fpn_level-1)
            level_list.append(fpn_level+1)

        base_h = cls_feats[fpn_level].size(2)
        base_w = cls_feats[fpn_level].size(3)
        bbox_refine_raws = []
        pts_cls_raws     = []
        segm_refine_raws = []
        pose_refine_raws = []

        for level in level_list:
            current_h = cls_feats[level].size(2)
            current_w = cls_feats[level].size(3)
            scale_h = current_h/base_h
            scale_w = current_w/base_w
            if self.task == 'bbox':
                offset_y = bbox_dcn_offset[:, 0::2, ...]
                offset_x = bbox_dcn_offset[:, 1::2, ...]
                offset_y *= scale_h
                offset_x *= scale_w
                bbox_dcn_offset_ = torch.stack([offset_y, offset_x], 2).view(bbox_dcn_offset.size(0),
                                                -1, bbox_dcn_offset.size(2), bbox_dcn_offset.size(3))
                
                bbox_refine_raws.append(self.pts_bbox_refine_conv(bbox_feats[level], bbox_dcn_offset_, 
                                                                  scale_h,
                                                                  scale_w))
                pts_cls_raws.append(self.pts_cls_conv(cls_feats[level], bbox_dcn_offset_, scale_h, scale_w))

            elif self.task == 'segm':
                segm_offset_y = segm_dcn_offset[:, 0::2, ...]
                segm_offset_x = segm_dcn_offset[:, 1::2, ...]
                segm_offset_y *= scale_h
                segm_offset_x *= scale_w
                segm_dcn_offset_ = torch.stack([segm_offset_y, segm_offset_x], 
                                                2).view(segm_dcn_offset.size(0), -1, 
                                                segm_dcn_offset.size(2), segm_dcn_offset.size(3))
                
                segm_refine_raws.append(self.pts_segm_refine_conv(segm_feats[level], 
                                                                  segm_dcn_offset_, 
                                                                  scale_h, scale_w))

                pts_cls_raws.append(self.pts_cls_conv(cls_feats[level], segm_dcn_offset_, 
                                                      scale_h, scale_w))

            elif self.task == 'pose_bbox':
                offset_y = bbox_dcn_offset[:, 0::2, ...]
                offset_x = bbox_dcn_offset[:, 1::2, ...]
                offset_y *= scale_h
                offset_x *= scale_w
                bbox_dcn_offset_ = torch.stack([offset_y, offset_x], 2).view(bbox_dcn_offset.size(0),
                                                -1, bbox_dcn_offset.size(2), bbox_dcn_offset.size(3))
                
                bbox_refine_raws.append(self.pts_bbox_refine_conv(bbox_feats[level], bbox_dcn_offset_, 
                                                                  scale_h,
                                                                  scale_w))

                pose_offset_y = pose_dcn_offset[:, 0::2, ...]
                pose_offset_x = pose_dcn_offset[:, 1::2, ...]
                pose_offset_y *= scale_h
                pose_offset_x *= scale_w
                pose_dcn_offset_ = torch.stack([pose_offset_y, pose_offset_x], 
                                                2).view(pose_dcn_offset.size(0), -1, 
                                                pose_dcn_offset.size(2), pose_dcn_offset.size(3))
                
                pose_refine_raws.append(self.pts_pose_refine_conv(pose_feats[level], 
                                                                  pose_dcn_offset_, 
                                                                  scale_h, scale_w))

                pts_cls_raws.append(self.pts_cls_conv(cls_feats[level], pose_dcn_offset_, 
                                                      scale_h, scale_w))
            elif self.task == 'pose_kbox':
                pose_offset_y = pose_dcn_offset[:, 0::2, ...]
                pose_offset_x = pose_dcn_offset[:, 1::2, ...]
                pose_offset_y *= scale_h
                pose_offset_x *= scale_w
                pose_dcn_offset_ = torch.stack([pose_offset_y, pose_offset_x], 
                                                2).view(pose_dcn_offset.size(0), -1, 
                                                pose_dcn_offset.size(2), pose_dcn_offset.size(3))
                
                pose_refine_raws.append(self.pts_pose_refine_conv(pose_feats[level], 
                                                                  pose_dcn_offset_, 
                                                                  scale_h, scale_w))

                pts_cls_raws.append(self.pts_cls_conv(cls_feats[level], pose_dcn_offset_, 
                                                      scale_h, scale_w))


        if self.task == 'bbox':
            bbox_reg_refine_out = self.pts_bbox_refine_out(
                self.relu(self.bbox_GN(self.bbox_af_dcn_conv(torch.cat(bbox_refine_raws, dim=1))+
                                    self.bbox_feat_conv(bbox_feats[fpn_level]))))

            bbox_reg_refine_sp = self.softplus(bbox_reg_refine_out + bbox_reg_init_sp.detach())

            pts_cls_out = self.pts_cls_out(
                        self.relu(self.cls_GN(self.cls_af_dcn_conv(torch.cat(pts_cls_raws, dim=1))+
                                    self.cls_feat_conv(cls_feats[fpn_level]))))

            return (pts_cls_out, bbox_reg_refine_sp, None, None)

        elif self.task == 'segm':
            segm_reg_refine_out = self.pts_segm_refine_out(
                self.relu(self.segm_GN(self.segm_af_dcn_conv(torch.cat(segm_refine_raws, dim=1))+
                                    self.segm_feat_conv(segm_feats[fpn_level]))))

            segm_reg_refine_sp = self.softplus(segm_reg_refine_out + segm_reg_init_sp.detach())

            pts_cls_out = self.pts_cls_out(
                        self.relu(self.cls_GN(self.cls_af_dcn_conv(torch.cat(pts_cls_raws, dim=1))+
                                    self.cls_feat_conv(cls_feats[fpn_level]))))

            return (pts_cls_out, None, segm_reg_refine_sp, None)

        elif self.task == 'pose_bbox':
            bbox_reg_refine_out = self.pts_bbox_refine_out(
                self.relu(self.bbox_GN(self.bbox_af_dcn_conv(torch.cat(bbox_refine_raws, dim=1))+
                                    self.bbox_feat_conv(bbox_feats[fpn_level]))))

            bbox_reg_refine_sp = self.softplus(bbox_reg_refine_out + bbox_reg_init_sp.detach())

            pts_cls_out = self.pts_cls_out(
                        self.relu(self.cls_GN(self.cls_af_dcn_conv(torch.cat(pts_cls_raws, dim=1))+
                                    self.cls_feat_conv(cls_feats[fpn_level]))))
            
            pose_reg_refine_out = self.pts_pose_refine_out(
                self.relu(self.pose_GN(self.pose_af_dcn_conv(torch.cat(pose_refine_raws, dim=1))+
                                    self.pose_feat_conv(pose_feats[fpn_level]))))

            pose_reg_refine_sp = self.softplus(pose_reg_refine_out + pose_reg_init_sp.detach())

            return (pts_cls_out, bbox_reg_refine_sp, None, pose_reg_refine_sp)

        elif self.task == 'pose_kbox':
            pts_cls_out = self.pts_cls_out(
                        self.relu(self.cls_GN(self.cls_af_dcn_conv(torch.cat(pts_cls_raws, dim=1))+
                                    self.cls_feat_conv(cls_feats[fpn_level]))))

            pose_reg_refine_out = self.pts_pose_refine_out(
                self.relu(self.pose_GN(self.pose_af_dcn_conv(torch.cat(pose_refine_raws, dim=1))+
                                    self.pose_feat_conv(pose_feats[fpn_level]))))

            pose_reg_refine_sp = self.softplus(pose_reg_refine_out + pose_reg_init_sp.detach())

            return (pts_cls_out, None, None, pose_reg_refine_sp)

    def get_points(self, featmap_sizes, img_metas):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.point_strides[i])
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points] for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def _target_single(self,
                       flat_proposals,
                       valid_flags,
                       num_level_proposals,
                       gt_bboxes,
                       gt_extremes,
                       gt_polygons,
                       gt_keypoints,
                       vs_keypoints,
                       gt_bboxes_ignore,
                       gt_labels,
                       label_channels=1,
                       stage='init',
                       unmap_outputs=True):
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None, ) * 8
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]
        num_level_proposals_inside = self.get_num_level_proposals_inside(num_level_proposals, 
                                                                         inside_flags)
        if stage == 'init':
            assigner = self.init_assigner
            assigner_type = self.train_cfg.init.assigner.type
            pos_weight = self.train_cfg.init.pos_weight
        else:
            assigner = self.refine_assigner
            assigner_type = self.train_cfg.refine.assigner.type
            pos_weight = self.train_cfg.refine.pos_weight

        if assigner_type != "ATSSAssigner":
            assign_result = assigner.assign(proposals, gt_bboxes, gt_extremes, gt_bboxes_ignore,
                                            gt_labels)
        else:
            assign_result = assigner.assign(proposals, num_level_proposals_inside, gt_bboxes, 
                                            gt_bboxes_ignore, gt_labels)
        sampling_result = self.sampler.sample(assign_result, proposals, gt_bboxes)

        num_valid_proposals = proposals.shape[0]
        bboxes_gt      = proposals.new_zeros([num_valid_proposals, 4])
        bbox_weights = proposals.new_zeros([num_valid_proposals, 4])
        labels       = proposals.new_full((num_valid_proposals, ), self.background_label, 
                                           dtype=torch.long)
        label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)
        if self.task == 'bbox':
            extremes_gt = proposals.new_zeros([num_valid_proposals, (self.num_vectors+1)*2])
        elif self.task == 'segm':
            polygons_gt = proposals.new_zeros([num_valid_proposals, (self.num_vectors+1)*2])
        elif self.task == 'pose_bbox':
            extremes_gt = proposals.new_zeros([num_valid_proposals, 10])
            keypoints_vs = proposals.new_zeros([num_valid_proposals, self.num_vectors])
            keypoints_gt = proposals.new_zeros([num_valid_proposals, (self.num_vectors+1)*2])
        elif self.task  == 'pose_kbox':
            keypoints_vs = proposals.new_zeros([num_valid_proposals, self.num_vectors])
            keypoints_gt = proposals.new_zeros([num_valid_proposals, (self.num_vectors+1)*2])

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            bboxes_gt[pos_inds, :] = pos_gt_bboxes

            if self.task == 'bbox':
                pos_gt_extremes = gt_extremes[sampling_result.pos_assigned_gt_inds]
                extremes_gt[pos_inds, :] = pos_gt_extremes
            elif self.task == 'segm':
                pos_gt_polygons = gt_polygons[sampling_result.pos_assigned_gt_inds]
                polygons_gt[pos_inds, :] = pos_gt_polygons
            elif self.task == 'pose_bbox':
                pos_gt_extremes = gt_extremes[sampling_result.pos_assigned_gt_inds]
                extremes_gt[pos_inds, :] = pos_gt_extremes

                pos_vs_keypoints = vs_keypoints[sampling_result.pos_assigned_gt_inds]
                keypoints_vs[pos_inds, :] = pos_vs_keypoints

                pos_gt_keypoints = gt_keypoints[sampling_result.pos_assigned_gt_inds]
                keypoints_gt[pos_inds, :] = pos_gt_keypoints
            elif self.task == 'pose_kbox':
                pos_vs_keypoints = vs_keypoints[sampling_result.pos_assigned_gt_inds]
                keypoints_vs[pos_inds, :] = pos_vs_keypoints

                pos_gt_keypoints = gt_keypoints[sampling_result.pos_assigned_gt_inds]
                keypoints_gt[pos_inds, :] = pos_gt_keypoints

            bbox_weights[pos_inds, :] = 1.0
            if gt_labels is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of proposals
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            labels              = unmap(labels, num_total_proposals, inside_flags)
            label_weights       = unmap(label_weights, num_total_proposals, inside_flags)
            bboxes_gt           = unmap(bboxes_gt, num_total_proposals, inside_flags)
            bbox_weights        = unmap(bbox_weights, num_total_proposals, inside_flags)
            if self.task == 'bbox':
                extremes_gt = unmap(extremes_gt, num_total_proposals, inside_flags)
                return (labels, label_weights, bboxes_gt, extremes_gt, None, None, None, 
                        bbox_weights, pos_inds, neg_inds)
            elif self.task == 'segm':
                polygons_gt = unmap(polygons_gt, num_total_proposals, inside_flags)
                return (labels, label_weights, bboxes_gt, None, polygons_gt, None, None, bbox_weights, 
                        pos_inds, neg_inds)
            elif self.task == 'pose_bbox':
                extremes_gt = unmap(extremes_gt, num_total_proposals, inside_flags)
                keypoints_gt = unmap(keypoints_gt, num_total_proposals, inside_flags)
                keypoints_vs = unmap(keypoints_vs, num_total_proposals, inside_flags)
                return (labels, label_weights, bboxes_gt, extremes_gt, None, keypoints_gt, keypoints_vs, 
                        bbox_weights, pos_inds, neg_inds)
            elif self.task == 'pose_kbox':
                keypoints_gt = unmap(keypoints_gt, num_total_proposals, inside_flags)
                keypoints_vs = unmap(keypoints_vs, num_total_proposals, inside_flags)
                return (labels, label_weights, bboxes_gt, None, None, keypoints_gt, keypoints_vs, 
                        bbox_weights, pos_inds, neg_inds)

    def get_targets(self,
                    proposals_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    gt_extremes_list,
                    gt_polygons_list,
                    gt_keypoints_list,
                    vs_keypoints_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    stage='init',
                    label_channels=1,
                    unmap_outputs=True):
        
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]
        num_level_proposals_list = [num_level_proposals] * num_imgs

        # concat all level points and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_extremes_list is None:
            gt_extremes_list = [None for _ in range(num_imgs)]
        if gt_polygons_list is None:
            gt_polygons_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        if gt_keypoints_list is None:
            gt_keypoints_list = [None for _ in range(num_imgs)]
        if vs_keypoints_list is None:
            vs_keypoints_list = [None for _ in range(num_imgs)]


        (all_labels, all_label_weights, all_bboxes_gt, all_extremes_gt, all_polygons_gt,
         all_keypoints_gt, all_keypoints_vs, all_bbox_weights, pos_inds_list, 
         neg_inds_list) = multi_apply(self._target_single,
                                      proposals_list,
                                      valid_flag_list,
                                      num_level_proposals_list,
                                      gt_bboxes_list,
                                      gt_extremes_list,
                                      gt_polygons_list,
                                      gt_keypoints_list,
                                      vs_keypoints_list,
                                      gt_bboxes_ignore_list,
                                      gt_labels_list,
                                      stage=stage,
                                      label_channels=label_channels,
                                      unmap_outputs=unmap_outputs)
       
        # no valid points
        if any([labels is None for labels in all_labels]):
            return None
        # sampled points of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])

        labels_list        = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights, num_level_proposals)
        bboxes_gt_list     = images_to_levels(all_bboxes_gt, num_level_proposals)
        bbox_weights_list  = images_to_levels(all_bbox_weights, num_level_proposals)
        if self.task == 'bbox':
            extremes_gt_list  = images_to_levels(all_extremes_gt, num_level_proposals)
            polygons_gt_list  = [None for _ in self.point_strides]
            keypoints_gt_list = [None for _ in self.point_strides]
            keypoints_vs_list = [None for _ in self.point_strides]
        elif self.task == 'segm':
            polygons_gt_list  = images_to_levels(all_polygons_gt, num_level_proposals)
            extremes_gt_list  = [None for _ in self.point_strides]
            keypoints_gt_list = [None for _ in self.point_strides]
            keypoints_vs_list = [None for _ in self.point_strides]
        elif self.task == 'pose_bbox':
            extremes_gt_list  = images_to_levels(all_extremes_gt, num_level_proposals)
            keypoints_gt_list = images_to_levels(all_keypoints_gt, num_level_proposals)
            keypoints_vs_list = images_to_levels(all_keypoints_vs, num_level_proposals)
            polygons_gt_list  = [None for _ in self.point_strides]
        elif self.task == 'pose_kbox':
            keypoints_gt_list = images_to_levels(all_keypoints_gt, num_level_proposals)
            keypoints_vs_list = images_to_levels(all_keypoints_vs, num_level_proposals)
            polygons_gt_list  = [None for _ in self.point_strides]
            extremes_gt_list  = [None for _ in self.point_strides]

        if stage == 'init':
            anchor_pts_list = images_to_levels(proposals_list, num_level_proposals)
            return (labels_list, label_weights_list, bboxes_gt_list, extremes_gt_list, polygons_gt_list,
                    keypoints_gt_list, keypoints_vs_list, bbox_weights_list, num_total_pos, num_total_neg,
                    anchor_pts_list)
        else:
            return (labels_list, label_weights_list, bboxes_gt_list, extremes_gt_list, polygons_gt_list,
                    keypoints_gt_list, keypoints_vs_list, bbox_weights_list, num_total_pos, num_total_neg)
    
    def loss_single(self, 
                    cls_score, 
                    bbox_pts_pred_init, 
                    bbox_pts_pred_refine,
                    segm_pts_pred_init, 
                    segm_pts_pred_refine,
                    pose_pts_pred_init, 
                    pose_pts_pred_refine,
                    labels, 
                    label_weights,
                    bboxes_gt_init, 
                    bboxes_gt_refine,
                    polygons_gt_init, 
                    polygons_gt_refine, 
                    extremes_gt_init, 
                    extremes_gt_refine,
                    keypoints_gt_init, 
                    keypoints_gt_refine,
                    keypoints_vs_init, 
                    keypoints_vs_refine,
                    bbox_weights_init,
                    bbox_weights_refine,
                    anchor_pts,
                    stride, 
                    num_total_samples_init, 
                    num_total_samples_refine):

        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)

        loss_cls = self.loss_cls(cls_score, labels, label_weights,
                                 avg_factor=num_total_samples_refine)
        # points loss
        loss_bbox_init   = 0
        loss_bbox_refine = 0
        loss_segm_init   = 0
        loss_segm_refine = 0
        loss_pose_init   = 0
        loss_pose_refine = 0

        if self.task == 'bbox':
            bboxes_gt_init     = bboxes_gt_init.reshape(-1, 4)
            extremes_gt_init   = extremes_gt_init.reshape(-1, (self.num_vectors+1)*2)
            bbox_weights_init  = bbox_weights_init.reshape(-1, 4).repeat(1, self.num_vectors+1)
            bbox_pts_pred_init = bbox_pts_pred_init.permute(0, 2, 3, 1).reshape(-1,
                                                             (self.num_vectors+1)*4)*stride
            anchor_pts = anchor_pts.reshape(-1, 3)

            bbox_gt_reg_init, bbox_yx_inds_init = self.get_bbox_gt_reg(extremes_gt_init,
                                                                       anchor_pts,
                                                                       bbox_weights_init)

            normalize_term = self.point_base_scale * stride

            loss_bbox_init += self.loss_bbox_init(bbox_pts_pred_init / normalize_term,
                                                  bbox_gt_reg_init / normalize_term,
                                                  bbox_weights_init,
                                                  avg_factor = num_total_samples_init,
                                                  anchor_pts = anchor_pts[:,:-1] / normalize_term,
                                                  bbox_gt = bboxes_gt_init / normalize_term,
                                                  pos_inds = bbox_yx_inds_init)

            bboxes_gt_refine     = bboxes_gt_refine.reshape(-1, 4)
            extremes_gt_refine   = extremes_gt_refine.reshape(-1, (self.num_vectors+1)*2)
            bbox_weights_refine  = bbox_weights_refine.reshape(-1, 4).repeat(1, self.num_vectors+1)
            bbox_pts_pred_refine = bbox_pts_pred_refine.permute(0, 2, 3, 1).reshape(-1, 
                                                                 (self.num_vectors+1)*4)*stride

            bbox_gt_reg_refine, bbox_yx_inds_refine = self.get_bbox_gt_reg(extremes_gt_refine,
                                                                           anchor_pts,
                                                                           bbox_weights_refine) 

            loss_bbox_refine += self.loss_bbox_refine(bbox_pts_pred_refine / normalize_term,
                                                      bbox_gt_reg_refine / normalize_term,
                                                      bbox_weights_refine,
                                                      avg_factor = num_total_samples_refine,
                                                      anchor_pts = anchor_pts[:,:-1] / normalize_term,
                                                      bbox_gt = bboxes_gt_refine / normalize_term,
                                                      pos_inds = bbox_yx_inds_refine)
        elif self.task == 'segm':
            #points loss
            num_poly_pts = (self.num_vectors+1)*2
            bboxes_gt_init     = bboxes_gt_init.reshape(-1, 4)
            polygons_gt_init   = polygons_gt_init.reshape(-1, num_poly_pts)
            bbox_weights_init  = bbox_weights_init.reshape(-1, 4)[:,:1].repeat(1, num_poly_pts*2)
            segm_pts_pred_init = segm_pts_pred_init.permute(0, 2, 3, 1).reshape(-1,
                                                               num_poly_pts*2)*stride
            anchor_pts = anchor_pts.reshape(-1, 3)

            poly_gt_reg_init, poly_yx_inds_init = self.get_poly_gt_reg(polygons_gt_init,
                                                                       anchor_pts,
                                                                       bbox_weights_init)

            normalize_term = self.point_base_scale * stride

            loss_segm_init += self.loss_segm_init(segm_pts_pred_init / normalize_term,
                                                  poly_gt_reg_init / normalize_term,
                                                  bbox_weights_init,
                                                  avg_factor = num_total_samples_init,
                                                  anchor_pts = anchor_pts[:,:-1] / normalize_term,
                                                  bbox_gt = bboxes_gt_init / normalize_term,
                                                  pos_inds = poly_yx_inds_init)
                                                
            bboxes_gt_refine     = bboxes_gt_refine.reshape(-1, 4)
            polygons_gt_refine   = polygons_gt_refine.reshape(-1, num_poly_pts)
            bbox_weights_refine  = bbox_weights_refine.reshape(-1, 4)[:,:1].repeat(1, num_poly_pts*2)
            segm_pts_pred_refine = segm_pts_pred_refine.permute(0, 2, 3, 1).reshape(-1,
                                                               num_poly_pts*2)*stride

            poly_gt_reg_refine, poly_yx_inds_refine = self.get_poly_gt_reg(polygons_gt_refine,
                                                                           anchor_pts,
                                                                           bbox_weights_refine)

            loss_segm_refine += self.loss_segm_refine(segm_pts_pred_refine / normalize_term,
                                                      poly_gt_reg_refine / normalize_term,
                                                      bbox_weights_refine,
                                                      avg_factor = num_total_samples_refine,
                                                      anchor_pts = anchor_pts[:,:-1] / normalize_term,
                                                      bbox_gt = bboxes_gt_refine / normalize_term,
                                                      pos_inds = poly_yx_inds_refine)
        elif self.task == 'pose_bbox':
            bboxes_gt_init     = bboxes_gt_init.reshape(-1, 4)
            extremes_gt_init   = extremes_gt_init.reshape(-1, 10)
            bbox_weights_init  = bbox_weights_init.reshape(-1, 4).repeat(1, 5)
            bbox_pts_pred_init = bbox_pts_pred_init.permute(0, 2, 3, 1).reshape(-1, 20)*stride
            anchor_pts = anchor_pts.reshape(-1, 3)

            bbox_gt_reg_init, bbox_yx_inds_init = self.get_bbox_gt_reg(extremes_gt_init,
                                                                       anchor_pts,
                                                                       bbox_weights_init)

            normalize_term = self.point_base_scale * stride

            loss_bbox_init += self.loss_bbox_init(bbox_pts_pred_init / normalize_term,
                                                  bbox_gt_reg_init / normalize_term,
                                                  bbox_weights_init,
                                                  avg_factor = num_total_samples_init,
                                                  anchor_pts = anchor_pts[:,:-1] / normalize_term,
                                                  bbox_gt = bboxes_gt_init / normalize_term,
                                                  pos_inds = bbox_yx_inds_init)

            bboxes_gt_refine     = bboxes_gt_refine.reshape(-1, 4)
            extremes_gt_refine   = extremes_gt_refine.reshape(-1, 10)
            bbox_weights_refine  = bbox_weights_refine.reshape(-1, 4).repeat(1, 5)
            bbox_pts_pred_refine = bbox_pts_pred_refine.permute(0, 2, 3, 1).reshape(-1, 20)*stride

            bbox_gt_reg_refine, bbox_yx_inds_refine = self.get_bbox_gt_reg(extremes_gt_refine,
                                                                           anchor_pts,
                                                                           bbox_weights_refine) 

            loss_bbox_refine += self.loss_bbox_refine(bbox_pts_pred_refine / normalize_term,
                                                      bbox_gt_reg_refine / normalize_term,
                                                      bbox_weights_refine,
                                                      avg_factor = num_total_samples_refine,
                                                      anchor_pts = anchor_pts[:,:-1] / normalize_term,
                                                      bbox_gt = bboxes_gt_refine / normalize_term,
                                                      pos_inds = bbox_yx_inds_refine)

            
            num_pose_pts = (self.num_vectors+1)*2
            keypoints_gt_init  = keypoints_gt_init.reshape(-1, num_pose_pts)
            keypoints_vs_init  = keypoints_vs_init.reshape(-1, self.num_vectors)
            bbox_weights_init  = bbox_weights_init[:,:1].repeat(1, num_pose_pts*2)
            pose_pts_pred_init = pose_pts_pred_init.permute(0, 2, 3, 1).reshape(-1,
                                                               num_pose_pts*2)*stride

            pose_gt_reg_init, pose_yx_inds_init = self.get_poly_gt_reg(keypoints_gt_init,
                                                                       anchor_pts,
                                                                       bbox_weights_init)


            loss_pose_init += self.loss_pose_init(pose_pts_pred_init / normalize_term,
                                                  pose_gt_reg_init / normalize_term,
                                                  bbox_weights_init,
                                                  avg_factor = num_total_samples_init,
                                                  anchor_pts = anchor_pts[:,:-1] / normalize_term,
                                                  bbox_gt = None,
                                                  pos_inds = pose_yx_inds_init,
                                                  vs = keypoints_vs_init)
                                                
            keypoints_gt_refine = keypoints_gt_refine.reshape(-1, num_pose_pts)
            keypoints_vs_refine = keypoints_vs_refine.reshape(-1, self.num_vectors)
            bbox_weights_refine = bbox_weights_refine[:,:1].repeat(1, num_pose_pts*2)
            pose_pts_pred_refine = pose_pts_pred_refine.permute(0, 2, 3, 1).reshape(-1,
                                                                 num_pose_pts*2)*stride

            pose_gt_reg_refine, pose_yx_inds_refine = self.get_poly_gt_reg(keypoints_gt_refine,
                                                                           anchor_pts,
                                                                           bbox_weights_refine)

            loss_pose_refine += self.loss_pose_refine(pose_pts_pred_refine / normalize_term,
                                                      pose_gt_reg_refine / normalize_term,
                                                      bbox_weights_refine,
                                                      avg_factor = num_total_samples_refine,
                                                      anchor_pts = anchor_pts[:,:-1] / normalize_term,
                                                      bbox_gt = None,
                                                      pos_inds = pose_yx_inds_refine,
                                                      vs = keypoints_vs_refine)

        elif self.task == 'pose_kbox':
            bbox_weights_init   = bbox_weights_init.reshape(-1, 4)
            bbox_weights_refine = bbox_weights_refine.reshape(-1, 4)
            anchor_pts = anchor_pts.reshape(-1, 3)
            normalize_term = self.point_base_scale * stride

            num_pose_pts = (self.num_vectors+1)*2
            keypoints_gt_init  = keypoints_gt_init.reshape(-1, num_pose_pts)
            keypoints_vs_init  = keypoints_vs_init.reshape(-1, self.num_vectors)
            bbox_weights_init  = bbox_weights_init[:,:1].repeat(1, num_pose_pts*2)
            pose_pts_pred_init = pose_pts_pred_init.permute(0, 2, 3, 1).reshape(-1,
                                                               num_pose_pts*2)*stride

            pose_gt_reg_init, pose_yx_inds_init = self.get_poly_gt_reg(keypoints_gt_init,
                                                                       anchor_pts,
                                                                       bbox_weights_init)


            loss_pose_init += self.loss_pose_init(pose_pts_pred_init / normalize_term,
                                                  pose_gt_reg_init / normalize_term,
                                                  bbox_weights_init,
                                                  avg_factor = num_total_samples_init,
                                                  anchor_pts = anchor_pts[:,:-1] / normalize_term,
                                                  bbox_gt = None,
                                                  pos_inds = pose_yx_inds_init,
                                                  vs = keypoints_vs_init)
                                                
            keypoints_gt_refine = keypoints_gt_refine.reshape(-1, num_pose_pts)
            keypoints_vs_refine = keypoints_vs_refine.reshape(-1, self.num_vectors)
            bbox_weights_refine = bbox_weights_refine[:,:1].repeat(1, num_pose_pts*2)
            pose_pts_pred_refine = pose_pts_pred_refine.permute(0, 2, 3, 1).reshape(-1,
                                                                 num_pose_pts*2)*stride

            pose_gt_reg_refine, pose_yx_inds_refine = self.get_poly_gt_reg(keypoints_gt_refine,
                                                                           anchor_pts,
                                                                           bbox_weights_refine)

            loss_pose_refine += self.loss_pose_refine(pose_pts_pred_refine / normalize_term,
                                                      pose_gt_reg_refine / normalize_term,
                                                      bbox_weights_refine,
                                                      avg_factor = num_total_samples_refine,
                                                      anchor_pts = anchor_pts[:,:-1] / normalize_term,
                                                      bbox_gt = None,
                                                      pos_inds = pose_yx_inds_refine,
                                                      vs = keypoints_vs_refine)

        return (loss_cls, loss_bbox_init, loss_bbox_refine, loss_segm_init, loss_segm_refine,
                loss_pose_init, loss_pose_refine)

    def loss(self,
             cls_scores,
             bbox_pts_preds_init,
             bbox_pts_preds_refine,
             segm_pts_preds_init,
             segm_pts_preds_refine,
             pose_pts_preds_init,
             pose_pts_preds_refine,
             gt_bboxes,
             gt_extremes,
             gt_keypoints_vs,
             gt_masks,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        if self.task == 'bbox':
            gt_polygons = None
            gt_keypoints, vs_keypoints = None, None
            if gt_extremes is None:
                gt_extremes = self.get_border_center(gt_bboxes) # border centers and bbox center
        elif self.task == 'segm':
            gt_extremes = None
            gt_keypoints, vs_keypoints = None, None
            gt_polygons, gt_bboxes = self.process_polygons(gt_masks, cls_scores)
        elif self.task == 'pose_bbox':
            gt_polygons = None
            if gt_extremes is None:
                gt_extremes = self.get_border_center(gt_bboxes) # border centers and bbox center
            gt_keypoints, vs_keypoints = self.process_keypoints_with_bbox(gt_bboxes, gt_keypoints_vs)
        elif self.task == 'pose_kbox':
            gt_polygons = None
            gt_extremes = None
            gt_keypoints, gt_bboxes, vs_keypoints = self.process_keypoints_with_kbox(gt_keypoints_vs)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels

        # target for initial stage
        base_pts_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)

        candidate_list = base_pts_list

        cls_reg_targets_init = self.get_targets(candidate_list.copy(),
                                                valid_flag_list.copy(),
                                                gt_bboxes,
                                                gt_extremes, 
                                                gt_polygons,
                                                gt_keypoints,
                                                vs_keypoints,
                                                img_metas,
                                                gt_bboxes_ignore_list=gt_bboxes_ignore,
                                                gt_labels_list=gt_labels,
                                                stage='init',
                                                label_channels=label_channels)

        (*_, bboxes_gt_list_init, extremes_gt_list_init, polygons_gt_list_init, keypoints_gt_list_init,
         keypoints_vs_list_init, bbox_weights_list_init, num_total_pos_init, num_total_neg_init, 
         anchor_pts_list) = cls_reg_targets_init

        # target for refinement stage
        bbox_list = []
        if self.task == 'bbox' or self.task == 'pose_bbox':
            for i_img, base_pts in enumerate(base_pts_list):
                bbox = []
                for i_lvl in range(len(bbox_pts_preds_init)):
                    bbox_preds_init = self.extreme_points2bbox(bbox_pts_preds_init[i_lvl].detach())
                    bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                    bbox_center = torch.cat([base_pts[i_lvl][:, :2], base_pts[i_lvl][:, :2]], dim=1)
                    bbox.append(bbox_center + bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
                bbox_list.append(bbox)

        elif self.task == 'segm':
            for i_img, base_pts in enumerate(base_pts_list):
                bbox = []
                for i_lvl in range(len(segm_pts_preds_init)):
                    bbox_preds_init = self.vectors2bbox(segm_pts_preds_init[i_lvl].detach())
                    bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                    bbox_center = torch.cat([base_pts[i_lvl][:, :2], base_pts[i_lvl][:, :2]], dim=1)
                    bbox.append(bbox_center + bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
                bbox_list.append(bbox)
        elif self.task == 'pose_kbox':
            for i_img, base_pts in enumerate(base_pts_list):
                bbox = []
                for i_lvl in range(len(pose_pts_preds_init)):
                    bbox_preds_init = self.vectors2bbox(pose_pts_preds_init[i_lvl].detach())
                    bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                    bbox_center = torch.cat([base_pts[i_lvl][:, :2], base_pts[i_lvl][:, :2]], dim=1)
                    bbox.append(bbox_center + bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
                bbox_list.append(bbox)

        cls_reg_targets_refine = self.get_targets(bbox_list,
                                                  valid_flag_list,
                                                  gt_bboxes,
                                                  gt_extremes,
                                                  gt_polygons,
                                                  gt_keypoints,
                                                  vs_keypoints,
                                                  img_metas,
                                                  gt_bboxes_ignore_list=gt_bboxes_ignore,
                                                  gt_labels_list=gt_labels,
                                                  stage='refine',
                                                  label_channels=label_channels)

        (labels_list, label_weights_list, bboxes_gt_list_refine, extremes_gt_list_refine, 
         polygons_gt_list_refine, keypoints_gt_list_refine, keypoints_vs_list_refine, 
         bbox_weights_list_refine, num_total_pos_refine, num_total_neg_refine) = cls_reg_targets_refine

        # compute loss
        (loss_cls, loss_bbox_init, loss_bbox_refine, loss_segm_init, loss_segm_refine,
         loss_pose_init, loss_pose_refine) = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_pts_preds_init,
            bbox_pts_preds_refine,
            segm_pts_preds_init,
            segm_pts_preds_refine,
            pose_pts_preds_init,
            pose_pts_preds_refine,
            labels_list,
            label_weights_list,
            bboxes_gt_list_init,
            bboxes_gt_list_refine,
            polygons_gt_list_init,
            polygons_gt_list_refine,
            extremes_gt_list_init,
            extremes_gt_list_refine,
            keypoints_gt_list_init,
            keypoints_gt_list_refine,
            keypoints_vs_list_init,
            keypoints_vs_list_refine,
            bbox_weights_list_init,
            bbox_weights_list_refine,
            anchor_pts_list,
            self.point_strides,
            num_total_samples_init=num_total_pos_init,
            num_total_samples_refine=num_total_pos_refine)

        if self.task == 'bbox':
            loss_dict_all = {
                'loss_cls': loss_cls,
                'loss_bbox_init': loss_bbox_init,
                'loss_bbox_refine': loss_bbox_refine
            }
        elif self.task == 'segm':
            loss_dict_all = {
                'loss_cls': loss_cls,
                'loss_segm_init': loss_segm_init,
                'loss_segm_refine': loss_segm_refine
            }
        elif self.task == 'pose_bbox':
            loss_dict_all = {
                'loss_cls': loss_cls,
                'loss_bbox_init': loss_bbox_init,
                'loss_bbox_refine': loss_bbox_refine,
                'loss_pose_init': loss_pose_init,
                'loss_pose_refine': loss_pose_refine
            }
        elif self.task == 'pose_kbox':
            loss_dict_all = {
                'loss_cls': loss_cls,
                'loss_pose_init': loss_pose_init,
                'loss_pose_refine': loss_pose_refine
            }

        return loss_dict_all

    def get_bboxes(self,
                   cls_scores,
                   bbox_pts_preds_init,
                   bbox_pts_preds_refine,
                   segm_pts_preds_init,
                   segm_pts_preds_refine,
                   pose_pts_preds_init,
                   pose_pts_preds_refine,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   nms=True):
        if self.task == 'bbox':
            extreme_bbox_preds = [self.extreme_points2bbox(pts_pred, extreme=True)
                                  for pts_pred in bbox_pts_preds_refine]
        elif self.task == 'segm':
            poly_bbox_preds = [self.vectors2bbox(pts_pred, vector=True)
                                  for pts_pred in segm_pts_preds_refine]
        elif self.task == 'pose_bbox':
            extreme_bbox_preds = [self.extreme_points2bbox(pts_pred, extreme=True)
                                  for pts_pred in bbox_pts_preds_refine]
            kps_bbox_preds = [self.vectors2bbox(pts_pred, vector=True) for pts_pred in pose_pts_preds_refine]
        elif self.task == 'pose_kbox':
            kps_bbox_preds = [self.vectors2bbox(pts_pred, vector=True) for pts_pred in pose_pts_preds_refine]
        
        num_levels = len(cls_scores)
        mlvl_points = [self.point_generators[i].grid_points(cls_scores[i].size()[-2:],
                                                 self.point_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            if self.task == 'bbox':
                bbox_pred_list = [extreme_bbox_preds[i][1][img_id].detach() 
                                         for i in range(num_levels)]
                extreme_pred_list = [extreme_bbox_preds[i][0][img_id].detach() 
                                         for i in range(num_levels)]
                polygon_pred_list = [None for _ in range(num_levels)]
                kps_pred_list = [None for _ in range(num_levels)]

            elif self.task == 'segm':
                bbox_pred_list = [poly_bbox_preds[i][1][img_id].detach() 
                                         for i in range(num_levels)]
                polygon_pred_list = [poly_bbox_preds[i][0][img_id].detach() 
                                         for i in range(num_levels)]
                extreme_pred_list = [None for _ in range(num_levels)]
                kps_pred_list = [None for _ in range(num_levels)]

            elif self.task == 'pose_bbox':
                bbox_pred_list = [extreme_bbox_preds[i][1][img_id].detach() 
                                         for i in range(num_levels)]
                kps_pred_list = [kps_bbox_preds[i][0][img_id].detach() 
                                         for i in range(num_levels)]
                extreme_pred_list = [None for _ in range(num_levels)]
                polygon_pred_list = [None for _ in range(num_levels)]
            elif self.task == 'pose_kbox':
                bbox_pred_list = [kps_bbox_preds[i][1][img_id].detach() 
                                         for i in range(num_levels)]
                kps_pred_list = [kps_bbox_preds[i][0][img_id].detach() 
                                         for i in range(num_levels)]
                extreme_pred_list = [None for _ in range(num_levels)]
                polygon_pred_list = [None for _ in range(num_levels)]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']

            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list,
                                                extreme_pred_list, polygon_pred_list, kps_pred_list,
                                                mlvl_points, img_shape, scale_factor, cfg, rescale, nms)

            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           extreme_preds,
                           polygon_preds,
                           kps_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           nms=True):
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(mlvl_points)
        mlvl_bboxes   = []
        mlvl_extremes = []
        mlvl_polygons = []
        mlvl_kps      = []
        mlvl_scores   = []
        for i_lvl, (cls_score, bbox_pred, extreme_pred, polygon_pred,
                    kps_pred, points) in enumerate(zip(cls_scores, bbox_preds, extreme_preds, polygon_preds,
                                                       kps_preds, mlvl_points)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if self.task == 'bbox':
                extreme_pred = extreme_pred.permute(1, 2, 0).reshape(-1, self.num_vectors*2)
            elif self.task == 'segm':
                polygon_pred = polygon_pred.permute(1, 2, 0).reshape(-1, self.num_vectors*2)
            elif self.task == 'pose_bbox' or self.task == 'pose_kbox':
                kps_pred = kps_pred.permute(1, 2, 0).reshape(-1, self.num_vectors*2)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                if self.task == 'bbox':
                    extreme_pred = extreme_pred[topk_inds, :]
                elif self.task == 'segm':
                    polygon_pred = polygon_pred[topk_inds, :]
                elif self.task == 'pose_bbox' or self.task == 'pose_kbox':
                    kps_pred = kps_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bbox_pos_center = torch.cat([points[:, :2], points[:, :2]], dim=1)
            bboxes = bbox_pred * self.point_strides[i_lvl] + bbox_pos_center
            if self.task == 'bbox':
                extreme_pos_center = torch.cat([points[:, :2], points[:, :2],
                                                points[:, :2], points[:, :2]],dim=1)

                extremes = extreme_pred*self.point_strides[i_lvl] + extreme_pos_center
            elif self.task == 'segm':
                poly_pos_center = bbox_pos_center[:,:2].repeat(1, self.num_vectors)
                polygons = polygon_pred*self.point_strides[i_lvl] + poly_pos_center
            elif self.task == 'pose_bbox' or self.task == 'pose_kbox':
                kps_pos_center = bbox_pos_center[:,:2].repeat(1, self.num_vectors)
                kps = kps_pred*self.point_strides[i_lvl] + kps_pos_center

            x1 = bboxes[:, 0].clamp(min=0, max=img_shape[1])
            y1 = bboxes[:, 1].clamp(min=0, max=img_shape[0])
            x2 = bboxes[:, 2].clamp(min=0, max=img_shape[1])
            y2 = bboxes[:, 3].clamp(min=0, max=img_shape[0])
            mlvl_bboxes.append(torch.stack([x1, y1, x2, y2], dim=-1))

            if self.task == 'bbox':
                xt = extremes[:, 0].clamp(min=0, max=img_shape[1])
                yl = extremes[:, 3].clamp(min=0, max=img_shape[0])
                xb = extremes[:, 4].clamp(min=0, max=img_shape[1])
                yr = extremes[:, 7].clamp(min=0, max=img_shape[0])
                mlvl_extremes.append(torch.stack([xt, y1, x1, yl, xb, y2, x2, yr], dim=-1))

            elif self.task == 'segm':
                polygons_x = polygons[:, 0::2].reshape(-1)
                polygons_y = polygons[:, 1::2].reshape(-1)

                polygons_x_ = polygons_x.clamp(min=0, max=img_shape[1])
                polygons_y_ = polygons_y.clamp(min=0, max=img_shape[0])

                polygons_x_ = polygons_x_.reshape(polygons.size(0), -1)
                polygons_y_ = polygons_y_.reshape(polygons.size(0), -1)

                polygons_ = torch.stack([polygons_x_, polygons_y_], 2).reshape(polygons.size(0), -1)
                mlvl_polygons.append(polygons_)
            elif self.task == 'pose_bbox' or self.task == 'pose_kbox':
                kps_x = kps[:, 0::2].reshape(-1)
                kps_y = kps[:, 1::2].reshape(-1)

                kps_x_ = kps_x.clamp(min=0, max=img_shape[1])
                kps_y_ = kps_y.clamp(min=0, max=img_shape[0])

                kps_x_ = kps_x_.reshape(kps.size(0), -1)
                kps_y_ = kps_y_.reshape(kps.size(0), -1)

                kps_ = torch.stack([kps_x_, kps_y_], 2).reshape(kps.size(0), -1)

                mlvl_kps.append(kps_)

            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if self.task == 'bbox':
            mlvl_extremes = torch.cat(mlvl_extremes)
        elif self.task == 'segm':
            mlvl_polygons = torch.cat(mlvl_polygons)
        elif self.task == 'pose_bbox' or self.task == 'pose_kbox':
            mlvl_kps = torch.cat(mlvl_kps)

        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            if self.task == 'bbox':
                extreme_scale_factor = np.tile(scale_factor, 2)
                mlvl_extremes /= mlvl_extremes.new_tensor(extreme_scale_factor)
            elif self.task == 'segm':
                poly_scale_factor = np.tile(scale_factor[:2], self.num_vectors)
                mlvl_polygons /= mlvl_polygons.new_tensor(poly_scale_factor)
            elif self.task == 'pose_bbox' or self.task == 'pose_kbox':
                kps_scale_factor = np.tile(scale_factor[:2], self.num_vectors)
                mlvl_kps /= mlvl_kps.new_tensor(kps_scale_factor)         

        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        if nms:
            if self.task == 'bbox':
                det_bboxes, det_extremes, det_labels = multiclass_nms_lsvr(mlvl_bboxes, 
                                                                           mlvl_extremes,
                                                                           mlvl_scores,
                                                                           self.num_vectors,
                                                                           cfg.score_thr, 
                                                                           cfg.nms, cfg.max_per_img)
                return det_bboxes, det_extremes, det_labels
            elif self.task == 'segm':
                det_bboxes, det_polygons, det_labels = multiclass_nms_lsvr(mlvl_bboxes, 
                                                                           mlvl_polygons,
                                                                           mlvl_scores,
                                                                           self.num_vectors,
                                                                           cfg.score_thr, 
                                                                           cfg.nms, cfg.max_per_img)
                return det_bboxes, det_polygons, det_labels
            elif self.task == 'pose_bbox' or self.task == 'pose_kbox':
                det_bboxes, det_kps, det_labels = multiclass_nms_lsvr(mlvl_bboxes, 
                                                                      mlvl_kps,
                                                                      mlvl_scores,
                                                                      self.num_vectors,
                                                                      cfg.score_thr, 
                                                                      cfg.nms, cfg.max_per_img)
                return det_bboxes, det_kps, det_labels

        else:
            if self.task == 'bbox':
                return mlvl_bboxes, mlvl_extremes, mlvl_scores
            elif self.task == 'segm':
                return mlvl_bboxes, mlvl_polygons, mlvl_scores
            elif self.task == 'pose_bbox' or self.task == 'pose_kbox':
                return mlvl_bboxes, mlvl_kps, mlvl_scores

    def get_num_level_proposals_inside(self, num_level_proposals, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_proposals)
        num_level_proposals_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_proposals_inside

    def get_border_center(self, gt_bboxes_list):
        gt_extremes_list = []
        for gt_bboxes in gt_bboxes_list:
            border_t_center_x = (gt_bboxes[:, 2] + gt_bboxes[:, 0])/2.0
            border_t_center_y = gt_bboxes[:, 1]
            border_l_center_x = gt_bboxes[:, 0]
            border_l_center_y = (gt_bboxes[:, 3] + gt_bboxes[:, 1])/2.0
            border_b_center_x = (gt_bboxes[:, 2] + gt_bboxes[:, 0])/2.0
            border_b_center_y = gt_bboxes[:, 3]
            border_r_center_x = gt_bboxes[:, 2] 
            border_r_center_y = (gt_bboxes[:, 3] + gt_bboxes[:, 1])/2.0

            bbox_ct_x = (gt_bboxes[:, 2] + gt_bboxes[:, 0])/2.0
            bbox_ct_y = (gt_bboxes[:, 3] + gt_bboxes[:, 1])/2.0

            gt_extremes_list.append(torch.stack([border_t_center_x, border_t_center_y,
                                                 border_l_center_x, border_l_center_y,
                                                 border_b_center_x, border_b_center_y,
                                                 border_r_center_x, border_r_center_y,
                                                 bbox_ct_x, bbox_ct_y], dim=1))
        return gt_extremes_list

    def component_polygon_area(self, poly):
        """Compute the area of a component of a polygon.

        Using the shoelace formula:
        https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates

        Args:
            x (ndarray): x coordinates of the component
            y (ndarray): y coordinates of the component

        Return:
            float: the are of the component
        """ 
        x = poly[:,0]
        y = poly[:,1]
        return 0.5 * np.abs(
            np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def process_polygons(self, gt_masks_list, cls_scores):
        device = cls_scores[0].device
        dtype  = cls_scores[0].dtype
        gt_polygons_list = []
        gt_bboxes_list = []
        for img_id, gt_masks in enumerate(gt_masks_list):
            polygons = gt_masks.masks
            polygon_areas = gt_masks.areas 
            gt_polygons = []
            gt_bboxes = []
            for instance_id, instance_polys in enumerate(polygons):
                max_area = self.component_polygon_area(instance_polys[0].reshape(-1, 2))
                max_idx = 0
                for component_id, component_poly in enumerate(instance_polys):
                    component_area = self.component_polygon_area(component_poly.reshape(-1, 2))
                    if max_area < component_area:
                        max_area = component_area
                        max_idx = component_id

                poly = torch.tensor(instance_polys[max_idx].reshape(-1, 2), dtype=dtype, device=device)
                gt_polygons.append(poly)

            gt_polygons_stack = torch.stack(gt_polygons)

            polys_xmin = gt_polygons_stack[:,:,0].min(1)[0]
            polys_ymin = gt_polygons_stack[:,:,1].min(1)[0]
            polys_xmax = gt_polygons_stack[:,:,0].max(1)[0]
            polys_ymax = gt_polygons_stack[:,:,1].max(1)[0]

            polys_ct_x = (polys_xmin + polys_xmax)/2
            polys_ct_y = (polys_ymin + polys_ymax)/2

            polys_cts = torch.stack([polys_ct_x, polys_ct_y], 1).unsqueeze(1)

            gt_polygons_list.append(torch.cat([gt_polygons_stack, polys_cts],
                                              dim=1).reshape(polys_cts.size(0), -1))
            gt_bboxes_list.append(torch.stack([polys_xmin, polys_ymin, polys_xmax, polys_ymax], 
                                                1))

        return gt_polygons_list, gt_bboxes_list

    def process_keypoints_with_bbox(self, gt_bboxes_list, gt_keypoints_vs_list):
        gt_keypoints_list = []
        vs_keypoints_list = []
        for img_id, gt_bboxes in enumerate(gt_bboxes_list):
            gt_keypoints_vs = gt_keypoints_vs_list[img_id]

            gt_keypoints_x = gt_keypoints_vs[:, 0::3]
            gt_keypoints_y = gt_keypoints_vs[:, 1::3]
            vs_keypoints   = gt_keypoints_vs[:, 2::3]

            xmin = gt_bboxes[:, 0]
            ymin = gt_bboxes[:, 1]
            xmax = gt_bboxes[:, 2]
            ymax = gt_bboxes[:, 3]

            ct_x = (xmin + xmax)/2
            ct_y = (ymin + ymax)/2

            bbox_cts = torch.stack([ct_x, ct_y], 1)
            gt_keypoints = torch.stack((gt_keypoints_x, gt_keypoints_y),
                                       dim=2).reshape(gt_keypoints_vs.size(0), -1)
            gt_keypoints = torch.cat((gt_keypoints, bbox_cts), 1)

            gt_keypoints_list.append(gt_keypoints)
            vs_keypoints_list.append(vs_keypoints)

        return gt_keypoints_list, vs_keypoints_list

    def process_keypoints_with_kbox(self, gt_keypoints_vs_list):
        gt_keypoints_list = []
        vs_keypoints_list = []
        gt_keybboxes_list = []
        for img_id, gt_keypoints_vs in enumerate(gt_keypoints_vs_list):
            gt_keypoints_x = gt_keypoints_vs[:, 0::3]
            gt_keypoints_y = gt_keypoints_vs[:, 1::3]
            vs_keypoints   = gt_keypoints_vs[:, 2::3]

            vs_zero_x = gt_keypoints_x[vs_keypoints==0]
            vs_zero_y = gt_keypoints_y[vs_keypoints==0]

            gt_keypoints_x[vs_keypoints==0] = 10000000
            gt_keypoints_y[vs_keypoints==0] = 10000000

            xmin = gt_keypoints_x.min(1)[0]
            ymin = gt_keypoints_y.min(1)[0]

            gt_keypoints_x[vs_keypoints==0] = -1
            gt_keypoints_y[vs_keypoints==0] = -1

            xmax = gt_keypoints_x.max(1)[0]
            ymax = gt_keypoints_y.max(1)[0]


            gt_keypoints_x[vs_keypoints==0] = vs_zero_x
            gt_keypoints_y[vs_keypoints==0] = vs_zero_y

            ct_x = (xmin + xmax)/2
            ct_y = (ymin + ymax)/2

            bbox_cts = torch.stack([ct_x, ct_y], 1)
            gt_keypoints = torch.stack((gt_keypoints_x, gt_keypoints_y),
                                       dim=2).reshape(gt_keypoints_vs.size(0), -1)

            gt_keypoints = torch.cat((gt_keypoints, bbox_cts), 1)
            gt_keybboxes = torch.stack([xmin, ymin, xmax, ymax], 1)

            gt_keypoints_list.append(gt_keypoints)
            vs_keypoints_list.append(vs_keypoints)
            gt_keybboxes_list.append(gt_keybboxes)

        return gt_keypoints_list, gt_keybboxes_list, vs_keypoints_list

class DCNConvModule(nn.Module):
    def __init__(
        self, 
        in_channels = 256,
        out_channels = 256,
        kernel_size = 3,
        dilation = 1,
        num_groups = 1,
        dcn_pad = 1
    ):
        super(DCNConvModule, self).__init__()

        self.conv = ModulatedDeformConvPack(in_channels, out_channels, kernel_size, 1, dcn_pad)
        self.bn = nn.GroupNorm(num_groups, out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


                


