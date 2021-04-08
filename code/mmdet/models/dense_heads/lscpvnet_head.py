import numpy as np
import torch
import pdb
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init

from mmdet.core import (PointGenerator, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms, unmap)
from mmdet.ops import DeformConv, PyramidDeformConv, ModulatedDeformConvPack, TLPool, BRPool
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead

@HEADS.register_module()
class LSCPVHead(AnchorFreeHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 point_feat_channels=256,
                 shared_stacked_convs=1,
                 first_kernel_size=3,
                 kernel_size=1,
                 corner_dim=64,
                 num_points=9,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 conv_module_type= 'norm', #norm of dcn, norm is faster
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_heatmap=dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     loss_weight=0.25),
                 loss_offset=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_sem=dict(type='SEPFocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=0.1),
                 use_grid_points=False,
                 center_init=True,
                 moment_mul=0.01,
                 **kwargs):
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels
        self.shared_stacked_convs = shared_stacked_convs
        self.use_grid_points = use_grid_points
        self.center_init = center_init

        self.first_kernel_size = first_kernel_size
        self.kernel_size = kernel_size
        self.corner_dim = corner_dim
        self.conv_module_type = conv_module_type

        # we use deformable conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.fpn_levels = [i for i in range(len(self.point_strides))]
        self.point_generators = [PointGenerator() for _ in self.point_strides]

        if self.train_cfg:
            self.init_assigner = build_assigner(self.train_cfg.init.assigner)
            self.refine_assigner = build_assigner(self.train_cfg.refine.assigner)
            self.hm_assigner = build_assigner(self.train_cfg.heatmap.assigner)
            # use PseudoSampler when sampling is False
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.cls_out_channels = self.num_classes
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.loss_offset = build_loss(loss_offset)
        self.loss_sem = build_loss(loss_sem)

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()
        self.cls_GN = nn.GroupNorm(self.norm_cfg.num_groups, self.feat_channels)
        self.bbox_GN = nn.GroupNorm(self.norm_cfg.num_groups, self.feat_channels)
        self.cls_convs = nn.ModuleList()
        self.bbox_convs = nn.ModuleList()
        self.shared_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.conv_module_type == 'norm':
                self.cls_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                                  conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
                self.bbox_convs.append(ConvModule(chn, self.feat_channels, 3, stride=1, padding=1,
                                                  conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))                                    
            else:
                self.cls_convs.append(DCNConvModule(chn, self.feat_channels, self.dcn_kernel, 1,
                                                   self.norm_cfg.num_groups, self.dcn_pad))
                self.bbox_convs.append(DCNConvModule(chn, self.feat_channels, self.dcn_kernel, 1,
                                                   self.norm_cfg.num_groups, self.dcn_pad))

        for i in range(self.shared_stacked_convs):
            if self.conv_module_type == 'norm':
                self.shared_convs.append(
                    ConvModule(self.feat_channels, self.feat_channels, 3, stride=1, padding=1,
                               conv_cfg=self.conv_cfg, norm_cfg=self.norm_cfg))
            else: #dcn
                self.shared_convs.append(
                    DCNConvModule(self.feat_channels, self.feat_channels, 3, 1,
                                  self.norm_cfg.num_groups, self.dcn_pad))

        self.hem_tl = TLPool(self.feat_channels, self.conv_cfg, self.norm_cfg, 
                             first_kernel_size=self.first_kernel_size, kernel_size=self.kernel_size,
                             corner_dim=self.corner_dim)
        self.hem_br = BRPool(self.feat_channels, self.conv_cfg, self.norm_cfg, 
                             first_kernel_size=self.first_kernel_size, kernel_size=self.kernel_size, 
                             corner_dim=self.corner_dim)

       
        pts_out_dim = 4*5 + (self.num_points-5)*2
        cls_in_channels = self.feat_channels + 6

        self.pts_cls_conv = PyramidDeformConv(cls_in_channels, self.point_feat_channels,
                                              self.dcn_kernel, 1, self.dcn_pad)

        self.pts_cls_out = nn.Conv2d(self.point_feat_channels, self.cls_out_channels, 1, 1, 0)

        self.pts_bbox_init_conv = nn.Conv2d(self.feat_channels, self.point_feat_channels, 3, 1, 1)
        self.pts_bbox_init_out = nn.Conv2d(self.point_feat_channels, pts_out_dim, 1, 1, 0)
        pts_in_channels = self.feat_channels + 6
        self.pts_bbox_refine_conv = PyramidDeformConv(pts_in_channels, self.point_feat_channels,
                                                    self.dcn_kernel, 1, self.dcn_pad)
        self.pts_bbox_refine_out = nn.Conv2d(self.point_feat_channels, 20, 1, 1, 0)

        self.reppoints_hem_tl_score_out = nn.Conv2d(self.feat_channels, 1, 3, 1, 1)
        self.reppoints_hem_br_score_out = nn.Conv2d(self.feat_channels, 1, 3, 1, 1)
        self.reppoints_hem_tl_offset_out = nn.Conv2d(self.feat_channels, 2, 3, 1, 1)
        self.reppoints_hem_br_offset_out = nn.Conv2d(self.feat_channels, 2, 3, 1, 1)

        self.reppoints_sem_out = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1, 1, 0)
        self.reppoints_sem_embedding = ConvModule(
            self.feat_channels,
            self.feat_channels,
            1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

        self.cls_af_dcn_conv = nn.Sequential(
                                nn.Conv2d(3 * self.point_feat_channels,
                                            self.point_feat_channels,
                                            1, 1, 0),
                                            nn.ReLU())

        self.bbox_af_dcn_conv = nn.Sequential(
                                nn.Conv2d(3 * self.point_feat_channels,
                                            self.point_feat_channels,
                                            1, 1, 0),
                                            nn.ReLU())

        self.cls_feat_conv = nn.Conv2d(cls_in_channels, self.point_feat_channels, 3, 1, 1)
        self.bbox_feat_conv = nn.Conv2d(pts_in_channels, self.point_feat_channels, 3, 1, 1)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.bbox_convs:
            normal_init(m.conv, std=0.01)
        for m in self.shared_convs:
            normal_init(m.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.pts_cls_conv, std=0.01)
        normal_init(self.pts_cls_out, std=0.01, bias=bias_cls)
        normal_init(self.pts_bbox_init_conv, std=0.01)
        normal_init(self.pts_bbox_init_out, std=0.01)
        normal_init(self.pts_bbox_refine_conv, std=0.01)
        normal_init(self.pts_bbox_refine_out, std=0.01)
        normal_init(self.reppoints_hem_tl_score_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_hem_tl_offset_out, std=0.01)
        normal_init(self.reppoints_hem_br_score_out, std=0.01, bias=bias_cls)
        normal_init(self.reppoints_hem_br_offset_out, std=0.01)
        normal_init(self.reppoints_sem_out, std=0.01, bias=bias_cls)

        normal_init(self.cls_feat_conv, std=0.01)
        normal_init(self.bbox_feat_conv, std=0.01)
        normal_init(self.cls_af_dcn_conv[0], std=0.01)
        normal_init(self.bbox_af_dcn_conv[0], std=0.01)

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

    def get_pred_reg(self, raw_reg1, raw_reg2):
        raw_reg_reshape = raw_reg1.view(raw_reg1.shape[0], -1, 2, *raw_reg1.shape[2:])
        pos_reg, inds = torch.max(raw_reg_reshape, dim=2)
        neg_inds = inds == 0
        pos_reg[neg_inds] *= -1

        reg_for_dcn = torch.cat((pos_reg, raw_reg2), dim =1)
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

    def forward(self, feats):
        (cls_feats, bbox_feats, bbox_reg_init_sps, hem_score_outs, 
         hem_offset_outs, sem_scores_outs, bbox_dcn_offsets) = multi_apply(self.forward_single1, feats)

        pts_cls_outs, bbox_reg_refine_sps = multi_apply(self.forward_single2,
                                                        bbox_dcn_offsets,
                                                        bbox_reg_init_sps,
                                                        self.fpn_levels,
                                                        cls_feats = cls_feats,
                                                        bbox_feats = bbox_feats,
                                                        num_levels = len(self.fpn_levels))

        return (pts_cls_outs, bbox_reg_init_sps, bbox_reg_refine_sps, hem_score_outs,
               hem_offset_outs, sem_scores_outs)

    def forward_single1(self, x):
        ''' Forward feature map of a single FPN level.'''
        dcn_base_offset = self.dcn_base_offset.type_as(x)

        cls_feat = x
        bbox_feat = x

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for bbox_conv in self.bbox_convs:
            bbox_feat = bbox_conv(bbox_feat)

        shared_feat = bbox_feat
        for shared_conv in self.shared_convs:
            shared_feat = shared_conv(shared_feat)

        sem_feat = shared_feat
        hem_feat = shared_feat

        sem_scores_out = self.reppoints_sem_out(sem_feat)
        sem_feat = self.reppoints_sem_embedding(sem_feat)

        cls_feat = cls_feat + sem_feat
        bbox_feat = bbox_feat + sem_feat
        hem_feat = hem_feat + sem_feat

        # generate heatmap and offset
        hem_tl_feat = self.hem_tl(hem_feat)
        hem_br_feat = self.hem_br(hem_feat)

        hem_tl_score_out = self.reppoints_hem_tl_score_out(hem_tl_feat)
        hem_tl_offset_out = self.reppoints_hem_tl_offset_out(hem_tl_feat)
        hem_br_score_out = self.reppoints_hem_br_score_out(hem_br_feat)
        hem_br_offset_out = self.reppoints_hem_br_offset_out(hem_br_feat)

        hem_score_out = torch.cat([hem_tl_score_out, hem_br_score_out], dim=1)
        hem_offset_out = torch.cat([hem_tl_offset_out, hem_br_offset_out], dim=1)

        bbox_reg_init_out = self.pts_bbox_init_out(
                        self.relu(self.pts_bbox_init_conv(bbox_feat)))

        bbox_reg_init_sp = self.softplus(bbox_reg_init_out[:, :20, ...])
        bbox_pred_reg = self.get_pred_reg(bbox_reg_init_sp, bbox_reg_init_out[:, 20:, ...])

        bbox_pred_reg_grad_mul = (1 - self.gradient_mul)*bbox_pred_reg.detach(
                                  ) + self.gradient_mul * bbox_pred_reg

        bbox_dcn_offset = bbox_pred_reg_grad_mul - dcn_base_offset

        hem_feat = torch.cat([hem_score_out, hem_offset_out], dim=1)
        cls_feat = torch.cat([cls_feat, hem_feat], dim=1)
        bbox_feat = torch.cat([bbox_feat, hem_feat], dim=1)

        return (cls_feat, bbox_feat, bbox_reg_init_sp, hem_score_out, hem_offset_out, sem_scores_out,
               bbox_dcn_offset)

    def forward_single2(self, bbox_dcn_offset, bbox_reg_init_sp, fpn_level, 
                              cls_feats, bbox_feats, num_levels):
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

        base_h = bbox_feats[fpn_level].size(2)
        base_w = bbox_feats[fpn_level].size(3)
        bbox_refine_raws = []
        pts_cls_raws = []
        for level in level_list:
            current_h = bbox_feats[level].size(2)
            current_w = bbox_feats[level].size(3)
            scale_h = current_h/base_h
            scale_w = current_w/base_w
            offset_y = bbox_dcn_offset[:, 0::2, ...]
            offset_x = bbox_dcn_offset[:, 1::2, ...]
            offset_y *= scale_h
            offset_x *= scale_w
            bbox_dcn_offset_ = torch.stack([offset_y, offset_x], 2).view(bbox_dcn_offset.size(0),
                                              -1, bbox_dcn_offset.size(2), bbox_dcn_offset.size(3))
            
            bbox_refine_raws.append(self.pts_bbox_refine_conv(bbox_feats[level], bbox_dcn_offset_, scale_h,
                                                              scale_w))
            pts_cls_raws.append(self.pts_cls_conv(cls_feats[level], bbox_dcn_offset_, scale_h, scale_w))

        bbox_reg_refine_out = self.pts_bbox_refine_out(
                self.relu(self.bbox_GN(self.bbox_af_dcn_conv(torch.cat(bbox_refine_raws, dim=1))+
                                    self.bbox_feat_conv(bbox_feats[fpn_level]))))

        bbox_reg_refine_sp = self.softplus(bbox_reg_refine_out + bbox_reg_init_sp.detach())

        pts_cls_out = self.pts_cls_out(
                      self.relu(self.cls_GN(self.cls_af_dcn_conv(torch.cat(pts_cls_raws, dim=1))+
                                self.cls_feat_conv(cls_feats[fpn_level]))))

        return (pts_cls_out, bbox_reg_refine_sp)

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

    def centers_to_bboxes(self, point_list):
        """Get bboxes according to center points. Only used in MaxIOUAssigner.
        """
        bbox_list = []
        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5
                bbox_shift = torch.Tensor([-scale, -scale, scale, scale]).view(1, 4).type_as(point[0])
                bbox_center = torch.cat([point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list

    def offset_to_pts(self, center_list, pred_list):
        """Change from point offset to point coordinate."""
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, self.num_points)
                pts_shift = pred_list[i_lvl][i_img]
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                    -1, 2 * self.num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def _point_target_single(self,
                             flat_proposals,
                             valid_flags,
                             num_level_proposals,
                             gt_bboxes,
                             gt_extremes,
                             gt_bboxes_ignore,
                             gt_labels,
                             label_channels=1,
                             stage='init',
                             unmap_outputs=True):
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None, ) * 6
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
        bboxes_gt    = proposals.new_zeros([num_valid_proposals, 4])
        extremes_gt  = proposals.new_zeros([num_valid_proposals, 10])
        bbox_weights = proposals.new_zeros([num_valid_proposals, 4])
        labels       = proposals.new_full((num_valid_proposals, ), self.background_label, 
                                           dtype=torch.long)
        label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            bboxes_gt[pos_inds, :] = pos_gt_bboxes

            pos_gt_extremes = gt_extremes[sampling_result.pos_assigned_gt_inds]
            extremes_gt[pos_inds, :] = pos_gt_extremes

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
            extremes_gt         = unmap(extremes_gt, num_total_proposals, inside_flags)
            bbox_weights        = unmap(bbox_weights, num_total_proposals, inside_flags)

        return labels, label_weights, bboxes_gt, extremes_gt, bbox_weights, pos_inds, neg_inds

    def get_targets(self,
                    proposals_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    gt_extremes_list,
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
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
    
        (all_labels, all_label_weights, all_bboxes_gt, all_extremes_gt,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._point_target_single,
             proposals_list,
             valid_flag_list,
             num_level_proposals_list,
             gt_bboxes_list,
             gt_extremes_list,
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
        extremes_gt_list   = images_to_levels(all_extremes_gt, num_level_proposals)
        bbox_weights_list  = images_to_levels(all_bbox_weights, num_level_proposals)

        if stage == 'init':
            anchor_pts_list = images_to_levels(proposals_list, num_level_proposals)
            return (labels_list, label_weights_list, bboxes_gt_list, extremes_gt_list,
                    bbox_weights_list, num_total_pos, num_total_neg, anchor_pts_list)
        else:
            return (labels_list, label_weights_list, bboxes_gt_list, extremes_gt_list,
                    bbox_weights_list, num_total_pos, num_total_neg)

    def _hm_target_single(self,
                          flat_points,
                          inside_flags,
                          gt_bboxes,
                          gt_labels,
                          unmap_outputs=True):
        # assign gt and sample points
        if not inside_flags.any():
            return (None, ) * 12
        points = flat_points[inside_flags, :]

        assigner = self.hm_assigner
        gt_hm_tl, gt_offset_tl, pos_inds_tl, neg_inds_tl, \
        gt_hm_br, gt_offset_br, pos_inds_br, neg_inds_br = \
            assigner.assign(points, gt_bboxes, gt_labels)

        num_valid_points = points.shape[0]
        hm_tl_weights = points.new_zeros(num_valid_points, dtype=torch.float)
        hm_br_weights = points.new_zeros(num_valid_points, dtype=torch.float)
        offset_tl_weights = points.new_zeros([num_valid_points, 2], dtype=torch.float)
        offset_br_weights = points.new_zeros([num_valid_points, 2], dtype=torch.float)

        hm_tl_weights[pos_inds_tl] = 1.0
        hm_tl_weights[neg_inds_tl] = 1.0
        offset_tl_weights[pos_inds_tl, :] = 1.0

        hm_br_weights[pos_inds_br] = 1.0
        hm_br_weights[neg_inds_br] = 1.0
        offset_br_weights[pos_inds_br, :] = 1.0

        # map up to original set of grids
        if unmap_outputs:
            num_total_points = flat_points.shape[0]
            gt_hm_tl = unmap(gt_hm_tl, num_total_points, inside_flags)
            gt_offset_tl = unmap(gt_offset_tl, num_total_points, inside_flags)
            hm_tl_weights = unmap(hm_tl_weights, num_total_points, inside_flags)
            offset_tl_weights = unmap(offset_tl_weights, num_total_points, inside_flags)

            gt_hm_br = unmap(gt_hm_br, num_total_points, inside_flags)
            gt_offset_br = unmap(gt_offset_br, num_total_points, inside_flags)
            hm_br_weights = unmap(hm_br_weights, num_total_points, inside_flags)
            offset_br_weights = unmap(offset_br_weights, num_total_points, inside_flags)

        return (gt_hm_tl, gt_offset_tl, hm_tl_weights, offset_tl_weights, pos_inds_tl, neg_inds_tl,
                gt_hm_br, gt_offset_br, hm_br_weights, offset_br_weights, pos_inds_br, neg_inds_br)

    def get_hm_targets(self,
                       proposals_list,
                       valid_flag_list,
                       gt_bboxes_list,
                       img_metas,
                       gt_labels_list=None,
                       unmap_outputs=True):
        """Compute refinement and classification targets for points.

        Args:
            points_list (list[list]): Multi level points of each image.
            valid_flag_list (list[list]): Multi level valid flags of each image.
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            img_metas (list[dict]): Meta info of each image.
            cfg (dict): train sample configs.

        Returns:
            tuple
        """
        num_imgs = len(img_metas)
        assert len(proposals_list) == len(valid_flag_list) == num_imgs

        # points number of multi levels
        num_level_proposals = [points.size(0) for points in proposals_list[0]]

        # concat all level points and flags to a single tensor
        for i in range(len(proposals_list)):
            assert len(proposals_list[i]) == len(valid_flag_list[i])
            proposals_list[i] = torch.cat(proposals_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_gt_hm_tl, all_gt_offset_tl, all_hm_tl_weights, all_offset_tl_weights, pos_inds_tl_list,
         neg_inds_tl_list, all_gt_hm_br, all_gt_offset_br, all_hm_br_weights, all_offset_br_weights,
         pos_inds_br_list, neg_inds_br_list) = \
            multi_apply(
                self._hm_target_single,
                proposals_list,
                valid_flag_list,
                gt_bboxes_list,
                gt_labels_list,
                unmap_outputs=unmap_outputs)
        # no valid points
        if any([gt_hm_tl is None for gt_hm_tl in all_gt_hm_tl]):
            return None
        # sampled points of all images
        num_total_pos_tl = sum([max(inds.numel(), 1) for inds in pos_inds_tl_list])
        num_total_neg_tl = sum([max(inds.numel(), 1) for inds in neg_inds_tl_list])
        num_total_pos_br = sum([max(inds.numel(), 1) for inds in pos_inds_br_list])
        num_total_neg_br = sum([max(inds.numel(), 1) for inds in neg_inds_br_list])

        gt_hm_tl_list = images_to_levels(all_gt_hm_tl, num_level_proposals)
        gt_offset_tl_list = images_to_levels(all_gt_offset_tl, num_level_proposals)
        hm_tl_weight_list = images_to_levels(all_hm_tl_weights, num_level_proposals)
        offset_tl_weight_list = images_to_levels(all_offset_tl_weights, num_level_proposals)

        gt_hm_br_list = images_to_levels(all_gt_hm_br, num_level_proposals)
        gt_offset_br_list = images_to_levels(all_gt_offset_br, num_level_proposals)
        hm_br_weight_list = images_to_levels(all_hm_br_weights, num_level_proposals)
        offset_br_weight_list = images_to_levels(all_offset_br_weights, num_level_proposals)

        return (gt_hm_tl_list, gt_offset_tl_list, hm_tl_weight_list, offset_tl_weight_list,
                gt_hm_br_list, gt_offset_br_list, hm_br_weight_list, offset_br_weight_list,
                num_total_pos_tl, num_total_neg_tl, num_total_pos_br, num_total_neg_br)

    def loss_single(self, cls_score, bbox_pts_pred_init, bbox_pts_pred_refine, hm_score, hm_offset,
                    labels, label_weights, instance_bboxes_gt_init, instance_bboxes_gt_refine,
                    instance_extremes_gt_init, instance_extremes_gt_refine,
                    bbox_weights_init, bbox_weights_refine, 
                    gt_hm_tl, gt_offset_tl, gt_hm_tl_weight, gt_offset_tl_weight,
                    gt_hm_br, gt_offset_br, gt_hm_br_weight, gt_offset_br_weight,
                    anchor_pts, stride,
                    num_total_samples_init, num_total_samples_refine,
                    num_total_samples_tl, num_total_samples_br):

        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples_refine)

        # points loss
        instance_bboxes_gt_init   = instance_bboxes_gt_init.reshape(-1, 4)
        instance_extremes_gt_init = instance_extremes_gt_init.reshape(-1, 10)
        bbox_weights_init         = bbox_weights_init.reshape(-1, 4).repeat(1, 5)
        bbox_pts_pred_init        = bbox_pts_pred_init.permute(0, 2, 3, 1).reshape(-1, 20)*stride
        anchor_pts = anchor_pts.reshape(-1, 3)

        bbox_gt_reg_init, bbox_yx_inds_init = self.get_bbox_gt_reg(instance_extremes_gt_init,
                                                                   anchor_pts,
                                                                   bbox_weights_init)

        normalize_term = self.point_base_scale * stride
        loss_bbox_init = 0
        
        loss_bbox_init += self.loss_bbox_init(bbox_pts_pred_init / normalize_term,
                                              bbox_gt_reg_init / normalize_term,
                                              bbox_weights_init,
                                              avg_factor = num_total_samples_init,
                                              anchor_pts = anchor_pts[:,:-1] / normalize_term,
                                              bbox_gt = instance_bboxes_gt_init / normalize_term,
                                              pos_inds = bbox_yx_inds_init)

        instance_bboxes_gt_refine   = instance_bboxes_gt_refine.reshape(-1, 4)
        instance_extremes_gt_refine = instance_extremes_gt_refine.reshape(-1, 10)
        bbox_weights_refine         = bbox_weights_refine.reshape(-1, 4).repeat(1, 5)
        bbox_pts_pred_refine        = bbox_pts_pred_refine.permute(0, 2, 3, 1).reshape(-1, 20)*stride 

        bbox_gt_reg_refine, bbox_yx_inds_refine = self.get_bbox_gt_reg(instance_extremes_gt_refine,
                                                                       anchor_pts,
                                                                       bbox_weights_refine)
        loss_bbox_refine = 0
        
        loss_bbox_refine += self.loss_bbox_refine(bbox_pts_pred_refine / normalize_term,
                                                  bbox_gt_reg_refine / normalize_term,
                                                  bbox_weights_refine,
                                                  avg_factor = num_total_samples_refine,
                                                  anchor_pts = anchor_pts[:,:-1] / normalize_term,
                                                  bbox_gt = instance_bboxes_gt_refine / normalize_term,
                                                  pos_inds = bbox_yx_inds_refine)

        # heatmap cls loss
        hm_score = hm_score.permute(0, 2, 3, 1).reshape(-1, 2)
        hm_score_tl, hm_score_br = torch.chunk(hm_score, 2, dim=-1)
        hm_score_tl = hm_score_tl.squeeze(1).sigmoid()
        hm_score_br = hm_score_br.squeeze(1).sigmoid()

        gt_hm_tl = gt_hm_tl.reshape(-1)
        gt_hm_tl_weight = gt_hm_tl_weight.reshape(-1)
        gt_hm_br = gt_hm_br.reshape(-1)
        gt_hm_br_weight = gt_hm_br_weight.reshape(-1)

        loss_heatmap = 0
        loss_heatmap += self.loss_heatmap(
            hm_score_tl, gt_hm_tl, gt_hm_tl_weight, avg_factor=num_total_samples_tl
        )
        loss_heatmap += self.loss_heatmap(
            hm_score_br, gt_hm_br, gt_hm_br_weight, avg_factor=num_total_samples_br
        )
        loss_heatmap /= 2.0

        # heatmap offset loss
        hm_offset = hm_offset.permute(0, 2, 3, 1).reshape(-1, 4)
        hm_offset_tl, hm_offset_br = torch.chunk(hm_offset, 2, dim=-1)

        gt_offset_tl = gt_offset_tl.reshape(-1, 2)
        gt_offset_tl_weight = gt_offset_tl_weight.reshape(-1, 2)
        gt_offset_br = gt_offset_br.reshape(-1, 2)
        gt_offset_br_weight = gt_offset_br_weight.reshape(-1, 2)

        loss_offset = 0
        loss_offset += self.loss_offset(
            hm_offset_tl, gt_offset_tl, gt_offset_tl_weight,
            avg_factor=num_total_samples_tl
        )
        loss_offset += self.loss_offset(
            hm_offset_br, gt_offset_br, gt_offset_br_weight,
            avg_factor=num_total_samples_br
        )
        loss_offset /= 2.0

        return loss_cls, loss_bbox_init, loss_bbox_refine, loss_heatmap, loss_offset

    def loss(self,
             cls_scores,
             bbox_pts_preds_init,
             bbox_pts_preds_refine,
             hm_scores,
             hm_offsets,
             sem_scores,
             gt_bboxes,
             gt_extremes,
             gt_sem_map,
             gt_sem_weights,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels

        # target for initial stage
        base_pts_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)

        if self.train_cfg.init.assigner['type'] != 'MaxIoUAssigner':
            # Assign target for center list
            candidate_list = base_pts_list
        else:
            # transform center list to bbox list and
            #   assign target for bbox list
            bbox_list = self.centers_to_bboxes(center_list)
            candidate_list = bbox_list

        cls_reg_targets_init = self.get_targets(candidate_list.copy(),
                                                valid_flag_list.copy(),
                                                gt_bboxes,
                                                gt_extremes, img_metas,
                                                gt_bboxes_ignore_list=gt_bboxes_ignore,
                                                gt_labels_list=gt_labels,
                                                stage='init',
                                                label_channels=label_channels)

        (*_, instance_bboxes_gt_list_init, instance_extremes_gt_list_init, bbox_weights_list_init,
         num_total_pos_init, num_total_neg_init, anchor_pts_list) = cls_reg_targets_init

        # target for heatmap in initial stage
        proposal_list, valid_flag_list = self.get_points(featmap_sizes, img_metas)
        heatmap_targets = self.get_hm_targets(
            proposal_list,
            valid_flag_list.copy(),
            gt_bboxes,
            img_metas,
            gt_labels)
        (gt_hm_tl_list, gt_offset_tl_list, gt_hm_tl_weight_list, gt_offset_tl_weight_list,
         gt_hm_br_list, gt_offset_br_list, gt_hm_br_weight_list, gt_offset_br_weight_list,
         num_total_pos_tl, num_total_neg_tl, num_total_pos_br, num_total_neg_br) = heatmap_targets

        # target for refinement stage
        bbox_list = []
        for i_img, base_pts in enumerate(base_pts_list):
            bbox = []
            for i_lvl in range(len(bbox_pts_preds_init)):
                bbox_preds_init = self.extreme_points2bbox(
                             bbox_pts_preds_init[i_lvl].detach())
                bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
                bbox_center = torch.cat([base_pts[i_lvl][:, :2], base_pts[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
            bbox_list.append(bbox)

        cls_reg_targets_refine = self.get_targets(bbox_list,
                                                  valid_flag_list,
                                                  gt_bboxes,
                                                  gt_extremes, img_metas,
                                                  gt_bboxes_ignore_list=gt_bboxes_ignore,
                                                  gt_labels_list=gt_labels,
                                                  stage='refine',
                                                  label_channels=label_channels)

        (labels_list, label_weights_list, instance_bboxes_gt_list_refine, instance_extremes_gt_list_refine,
        bbox_weights_list_refine, num_total_pos_refine, num_total_neg_refine) = cls_reg_targets_refine

        # compute loss
        loss_cls, loss_bbox_init, loss_bbox_refine, losses_heatmap, losses_offset = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_pts_preds_init,
            bbox_pts_preds_refine,
            hm_scores,
            hm_offsets,
            labels_list,
            label_weights_list,
            instance_bboxes_gt_list_init,
            instance_bboxes_gt_list_refine,
            instance_extremes_gt_list_init,
            instance_extremes_gt_list_refine,
            bbox_weights_list_init,
            bbox_weights_list_refine,
            gt_hm_tl_list,
            gt_offset_tl_list,
            gt_hm_tl_weight_list,
            gt_offset_tl_weight_list,
            gt_hm_br_list,
            gt_offset_br_list,
            gt_hm_br_weight_list,
            gt_offset_br_weight_list,
            anchor_pts_list,
            self.point_strides,
            num_total_samples_init=num_total_pos_init,
            num_total_samples_refine=num_total_pos_refine,
            num_total_samples_tl=num_total_pos_tl,
            num_total_samples_br=num_total_pos_br)

        # sem loss
        concat_sem_scores = []
        concat_gt_sem_map = []
        concat_gt_sem_weights = []

        for i in range(5):
            sem_score = sem_scores[i]
            gt_lvl_sem_map = F.interpolate(gt_sem_map, sem_score.shape[-2:]).reshape(-1)
            gt_lvl_sem_weight = F.interpolate(gt_sem_weights, sem_score.shape[-2:]).reshape(-1)
            sem_score = sem_score.reshape(-1)

            try:
                concat_sem_scores = torch.cat([concat_sem_scores, sem_score])
                concat_gt_sem_map = torch.cat([concat_gt_sem_map, gt_lvl_sem_map])
                concat_gt_sem_weights = torch.cat([concat_gt_sem_weights, gt_lvl_sem_weight])
            except:
                concat_sem_scores = sem_score
                concat_gt_sem_map = gt_lvl_sem_map
                concat_gt_sem_weights = gt_lvl_sem_weight

        loss_sem = self.loss_sem(concat_sem_scores, concat_gt_sem_map, concat_gt_sem_weights, 
                                 avg_factor=(concat_gt_sem_map > 0).sum())

        loss_dict_all = {'loss_cls': loss_cls,
                         'loss_bbox_init': loss_bbox_init,
                         'loss_bbox_refine': loss_bbox_refine,
                         'loss_heatmap': losses_heatmap,
                         'loss_offset': losses_offset,
                         'loss_sem': loss_sem,
                         }
        return loss_dict_all

    def get_bboxes(self,
                   cls_scores,
                   bbox_pts_preds_init,
                   bbox_pts_preds_refine,
                   hm_scores,
                   hm_offsets,
                   sem_scores,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   nms=True):
        assert len(cls_scores) == len(bbox_pts_preds_refine)
        extreme_bbox_preds = [self.extreme_points2bbox(pts_pred, extreme=True) 
                              for pts_pred in bbox_pts_preds_refine]
        num_levels = len(cls_scores)
        mlvl_points = [
            self.point_generators[i].grid_points(cls_scores[i].size()[-2:],
                                                 self.point_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                extreme_bbox_preds[i][1][img_id].detach() for i in range(num_levels)
            ]
            hm_scores_list = [
                hm_scores[i][img_id].detach() for i in range(num_levels)
            ]
            hm_offsets_list = [
                hm_offsets[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score_list, bbox_pred_list, hm_scores_list, 
                                                hm_offsets_list,
                                                mlvl_points, img_shape,
                                                scale_factor, cfg, rescale,
                                                nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           hm_scores,
                           hm_offsets,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           nms=True):
        def select(score_map, x, y, ks=2, i=0):
            H, W = score_map.shape[-2], score_map.shape[-1]
            score_map = score_map.sigmoid()
            score_map_original = score_map.clone()

            score_map, indices = F.max_pool2d_with_indices(score_map.unsqueeze(0), kernel_size=ks, stride=1, 
                                                           padding=(ks - 1) // 2)

            indices = indices.squeeze(0).squeeze(0)

            if ks % 2 == 0:
                round_func = torch.floor
            else:
                round_func = torch.round

            x_round = round_func((x / self.point_strides[i]).clamp(min=0, max=score_map.shape[-1] - 1))
            y_round = round_func((y / self.point_strides[i]).clamp(min=0, max=score_map.shape[-2] - 1))

            select_indices = indices[y_round.to(torch.long), x_round.to(torch.long)]
            new_x = select_indices % W
            new_y = select_indices // W

            score_map_squeeze = score_map_original.squeeze(0)
            score = score_map_squeeze[new_y, new_x]

            new_x, new_y = new_x.to(torch.float), new_y.to(torch.float)

            return new_x, new_y, score

        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        for i_lvl, (cls_score, bbox_pred, points) in enumerate(zip(cls_scores, bbox_preds, mlvl_points)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bbox_pos_center = torch.cat([points[:, :2], points[:, :2]], dim=1)
            bboxes = bbox_pred * self.point_strides[i_lvl] + bbox_pos_center
            x1 = bboxes[:, 0].clamp(min=0, max=img_shape[1])
            y1 = bboxes[:, 1].clamp(min=0, max=img_shape[0])
            x2 = bboxes[:, 2].clamp(min=0, max=img_shape[1])
            y2 = bboxes[:, 3].clamp(min=0, max=img_shape[0])

            if i_lvl > 0:
                i = 0 if i_lvl in (1, 2) else 1

                x1_new, y1_new, score1_new = select(hm_scores[i][0, ...], x1, y1, 2, i)
                x2_new, y2_new, score2_new = select(hm_scores[i][1, ...], x2, y2, 2, i)

                hm_offset = hm_offsets[i].permute(1, 2, 0)
                point_stride = self.point_strides[i]

                x1 = ((x1_new + hm_offset[y1_new.to(torch.long), x1_new.to(torch.long), 0]) * 
                       point_stride).clamp(min=0, max=img_shape[1])
                y1 = ((y1_new + hm_offset[y1_new.to(torch.long), x1_new.to(torch.long), 1]) * 
                       point_stride).clamp(min=0, max=img_shape[0])
                x2 = ((x2_new + hm_offset[y2_new.to(torch.long), x2_new.to(torch.long), 2]) * 
                       point_stride).clamp(min=0, max=img_shape[1])
                y2 = ((y2_new + hm_offset[y2_new.to(torch.long), x2_new.to(torch.long), 3]) * 
                       point_stride).clamp(min=0, max=img_shape[0])
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        if nms:
            det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores

    def get_num_level_proposals_inside(self, num_level_proposals, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_proposals)
        num_level_proposals_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_proposals_inside


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