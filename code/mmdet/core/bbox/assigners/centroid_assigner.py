import pdb
import torch

from ..builder import BBOX_ASSIGNERS
from .base_assigner import BaseAssigner
from .assign_result import AssignResult


@BBOX_ASSIGNERS.register_module()
class CentroidAssigner(BaseAssigner):
    """Assign a corresponding gt bbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    """

    def __init__(self, scale=4, pos_num=3, iou_type='center'):
        self.scale    = scale
        self.pos_num  = pos_num
        self.iou_type = iou_type

    def assign(self, points, gt_bboxes, gt_extreme_pts, gt_bboxes_ignore=None, gt_labels=None):
        """Assign gt to points.
        if iou_type == 'center':
             assign gt to the center points, which is the same as point_assigner_v2. 
        elif iou_type == 'centroid':
            assign gt to the centroid points. 
        """
        INF = 1e8
        num_gts, num_points = gt_bboxes.shape[0], points.shape[0]

        if num_gts == 0 or num_points == 0:
            # If no truth assign everything to the background
            assigned_gt_inds = points.new_full((num_points, ),
                                               0,
                                               dtype=torch.long)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = points.new_full((num_points, ),
                                                  -1,
                                                  dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        points_xy = points[:, :2]
        points_stride = points[:, 2]
        points_lvl = torch.log2(points_stride).int()  # [3...,4...,5...,6...,7...]
        lvl_min, lvl_max = points_lvl.min(), points_lvl.max()

        # assign gt box
        if self.iou_type == 'centroid':
            gt_bboxes_xy = self.gen_centroid(gt_extreme_pts, num_gts)
        else: 
            gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
            
        gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)
        scale = self.scale
        gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) +
                          torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
        gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

        distances = ((points_xy[:, None, :] - gt_bboxes_xy[None, :, :]) / gt_bboxes_wh[None, :, :]).norm(dim=2)

        distances[points_lvl[:, None] != gt_bboxes_lvl[None, :]] = INF

        # stores the assigned gt index of each point
        assigned_gt_inds = points.new_zeros((num_points, ), dtype=torch.long)

        min_dist, min_dist_index = torch.topk(distances, self.pos_num, dim=0, largest=False)

        distances_inf = torch.full_like(distances, INF)
        distances_inf[min_dist_index, torch.arange(num_gts)] = min_dist

        min_dist, min_dist_index = distances_inf.min(dim=1)
        assigned_gt_inds[min_dist != INF] = min_dist_index[min_dist != INF] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_points, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)

    def gen_centroid(self, pts, num_gts):
        extreme_pts = pts[:, :-2]
        pts_repeat = extreme_pts.repeat(1, 2)
        pts_reshape = pts_repeat.view(pts_repeat.shape[0], -1, 2, *pts_repeat.shape[2:])
        pts_x = pts_reshape[:, :, 0, ...]
        pts_y = pts_reshape[:, :, 1, ...]

        centroid_x_list = []
        centroid_y_list = []
        for i in range(4):
            triangle_x = pts_x[:, i:i+3]
            triangle_y = pts_y[:, i:i+3]

            centroid_x = (torch.sum(triangle_x, -1)/3.0).unsqueeze(-1)
            centroid_y = (torch.sum(triangle_y, -1)/3.0).unsqueeze(-1)

            centroid_x_list.append(centroid_x)
            centroid_y_list.append(centroid_y)
        
        centroid_xs = torch.cat(centroid_x_list, -1)
        centroid_ys = torch.cat(centroid_y_list, -1)

        line1_start_xs, line1_start_ys = centroid_xs[:, 0], centroid_ys[:, 0]
        line1_end_xs,   line1_end_ys   = centroid_xs[:, 2], centroid_ys[:, 2]

        line2_start_xs, line2_start_ys = centroid_xs[:, 1], centroid_ys[:, 1]
        line2_end_xs,   line2_end_ys   = centroid_xs[:, 3], centroid_ys[:, 3]

        detL1 = line1_start_xs * line1_end_ys - line1_start_ys * line1_end_xs
        detL2 = line2_start_xs * line2_end_ys - line2_start_ys * line2_end_xs

        x1mx2 = line1_start_xs - line1_end_xs
        x3mx4 = line2_start_xs - line2_end_xs

        y1my2 = line1_start_ys - line1_end_ys
        y3my4 = line2_start_ys - line2_end_ys

        xnom = detL1*x3mx4 - detL2*x1mx2
        ynom = detL1*y3my4 - detL2*y1my2

        denom = x1mx2*y3my4 - y1my2*x3mx4

        polygon_centroid_xs = (xnom/denom).unsqueeze(-1)
        polygon_centroid_ys = (ynom/denom).unsqueeze(-1)

        return torch.cat((polygon_centroid_xs, polygon_centroid_ys), -1)