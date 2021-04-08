import pdb
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES
from .utils import weighted_loss

def get_bbox_from_extreme(pred, anchor_pts):
    pred_reshape = pred.view(pred.shape[0], -1, 2)
    pred_pts, inds = torch.max(pred_reshape, dim=2)
    neg_inds = inds == 0
    pred_pts[neg_inds] *= -1
    pred_pts = pred_pts.view(pred_pts.shape[0], -1, 2)
    pred_y = pred_pts[:,:,0]
    pred_x = pred_pts[:,:,1]
    pred_pts_ = torch.stack([pred_x, pred_y], -1).reshape(-1, 10)

    anchor_pts_repeat = anchor_pts[:, :2].repeat(1, 5)
    pred_pts_ += anchor_pts_repeat
    pred_pts_reshape = pred_pts_.view(pred_pts_.shape[0], -1, 2)

    pred_pts_x = pred_pts_reshape[:,:,0]
    pred_pts_y = pred_pts_reshape[:,:,1]

    bbox_left   = pred_pts_x[:,1].unsqueeze(1)
    bbox_right  = pred_pts_x[:,3].unsqueeze(1)
    bbox_up     = pred_pts_y[:,0].unsqueeze(1)
    bbox_bottom = pred_pts_y[:,2].unsqueeze(1)

    bbox_pred = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)
    return bbox_pred

def get_bbox_from_polygon(pred, anchor_pts):
    num_poly_pts = int(pred[:,:-4, ...].size(1)/4)
    pred_reshape = pred[:,:-4,...].view(pred.shape[0], -1, 2, *pred.shape[2:])
    pred_pts, inds = torch.max(pred_reshape, dim=2)
    neg_inds = inds == 0
    pred_pts[neg_inds] *= -1
    pred_pts = pred_pts.view(pred_pts.shape[0], -1, 2)
    pred_y = pred_pts[:,:,0]
    pred_x = pred_pts[:,:,1]
    pred_pts_ = torch.stack([pred_x, pred_y], -1).reshape(-1, num_poly_pts*2)

    anchor_pts_repeat = anchor_pts[:, :2].repeat(1, num_poly_pts)
    pred_pts_ += anchor_pts_repeat

    pred_pts_reshape = pred_pts_.view(pred_pts_.shape[0], -1, 2)
    pred_pts_x = pred_pts_reshape[:,:,0]
    pred_pts_y = pred_pts_reshape[:,:,1]

    polys_xmin = pred_pts_x.min(1)[0]
    polys_ymin = pred_pts_y.min(1)[0]
    polys_xmax = pred_pts_x.max(1)[0]
    polys_ymax = pred_pts_y.max(1)[0]

    bbox_pred = torch.stack([polys_xmin, polys_ymin, polys_xmax, polys_ymax], 1)
    return bbox_pred

@weighted_loss
def cross_iou_loss(pred, target, loss_type=None, anchor_pts=None, vs=None,
                   bbox_gt=None, pos_inds=None, eps=1e-6, alpha=0.2, stride=9):
    
    neg_inds = ~pos_inds
    target[neg_inds] = alpha*target[pos_inds]

    if loss_type == 'polygon':
        overlaps = []
        total = torch.stack([pred, target], -1)
        total_reshape = total.reshape(total.size(0), -1, 4, total.size(-1))
        for i in range(stride):
            total_ = total_reshape[:, i::stride, ...].reshape(total.size(0), -1, total.size(-1))
            l_max = total_.max(dim=2)[0]
            l_min = total_.min(dim=2)[0]
            overlaps.append(l_min.sum(dim=1)/l_max.sum(dim=1))
        overlaps = torch.stack(overlaps, -1).sum(-1)/stride

    elif loss_type == 'bbox':
        total = torch.stack([pred, target], -1)
        l_max = total.max(dim=2)[0]
        l_min = total.min(dim=2)[0]

        overlaps = l_min.sum(dim=1)/l_max.sum(dim=1)
    else: #keypoint
        target_reshape = target.reshape(target.size(0), -1, 2)
        pred_reshape   = pred.reshape(pred.size(0), -1, 2)
        total = torch.stack([pred_reshape, target_reshape], -1)
        l_max = total.max(dim=-1)[0].clamp(min=eps)
        l_min = total.min(dim=-1)[0]
        overlaps = l_min.sum(dim=-1)/l_max.sum(dim=-1)
        vs[vs>0]=1
        vs_stack = torch.stack((vs, vs), 2).reshape(vs.size(0), -1)
        overlaps[:,:-2] *= vs_stack

        overlaps = overlaps.sum(-1)/total.size(1)

    if loss_type == 'bbox':
        bbox_pred = get_bbox_from_extreme(pred, anchor_pts)
    elif loss_type == 'polygon':
        bbox_pred = get_bbox_from_polygon(pred, anchor_pts)

    if loss_type != 'keypoint':
        enclose_x1y1 = torch.min(bbox_pred[:, :2], bbox_gt[:, :2])
        enclose_x2y2 = torch.max(bbox_pred[:, 2:], bbox_gt[:, 2:])
        enclose_wh   = (enclose_x2y2 - enclose_x1y1).clamp(min=0)

        cw = enclose_wh[:, 0]
        ch = enclose_wh[:, 1]

        c2 = cw**2 + ch**2 +eps

        b1_x1, b1_y1 = bbox_pred[:, 0], bbox_pred[:, 1]
        b1_x2, b1_y2 = bbox_pred[:, 2], bbox_pred[:, 3]
        b2_x1, b2_y1 = bbox_gt[:, 0], bbox_gt[:, 1]
        b2_x2, b2_y2 = bbox_gt[:, 2], bbox_gt[:, 3]

        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

        left = ((b2_x1 + b2_x2) - (b1_x1 + b1_x2))**2 / 4
        right = ((b2_y1 + b2_y2) - (b1_y1 + b1_y2))**2 / 4
        rho2 = left + right

        factor = 4 / math.pi**2
        v = factor * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)

        loss = 1 - (overlaps - (rho2 / c2 + v**2 / (1 - overlaps +v)))
    else:
        loss = 1 - overlaps

    return loss

@LOSSES.register_module()
class CrossIOULoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0, loss_type='bbox', alpha=0.2, stride=9):
        super(CrossIOULoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.loss_type = loss_type
        self.alpha = alpha
        self.stride =stride

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        if weight is not None and not torch.any(weight>0):
            return (pred * weight).sum() #0
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        loss = self.loss_weight * cross_iou_loss(
            pred,
            target,
            weight,
            loss_type=self.loss_type,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            alpha = self.alpha,
            stride = self.stride,
            **kwargs)
        return loss
