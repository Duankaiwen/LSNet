import numpy as np
import torch


def bbox_flip(bboxes, img_shape, direction='horizontal'):
    """Flip bboxes horizontally or vertically.

    Args:
        bboxes (Tensor): Shape (..., 4*k)
        img_shape (tuple): Image shape.
        direction (str): Flip direction, options are "horizontal" and
            "vertical". Default: "horizontal"


    Returns:
        Tensor: Flipped bboxes.
    """
    assert bboxes.shape[-1] % 4 == 0
    assert direction in ['horizontal', 'vertical']
    flipped = bboxes.clone()
    if direction == 'vertical':
        flipped[..., 1::4] = img_shape[0] - bboxes[..., 3::4]
        flipped[..., 3::4] = img_shape[0] - bboxes[..., 1::4]
    else:
        flipped[:, 0::4] = img_shape[1] - bboxes[:, 2::4]
        flipped[:, 2::4] = img_shape[1] - bboxes[:, 0::4]
    return flipped


def polygon_flip(polygons, img_shape, direction='horizontal'):
    assert direction in ['horizontal', 'vertical']
    flipped = polygons.clone()
    if direction == 'horizontal':
        dim = img_shape[1]
        idx = 0
    else:
        dim = img_shape[0]
        idx = 1
    flipped[:, idx::2] = dim - flipped[:, idx::2]
    if flipped.size(0) > 0:
        x = flipped.reshape(flipped.size(0), -1, 2)
        new_x = torch.zeros_like(x)
        new_x[:, 1:] = torch.flip(x, [1])[:,:-1]
        new_x[:, 0] = torch.flip(x, [1])[:, -1]
        flipped = new_x.reshape(flipped.size(0), -1)
    return flipped

def extreme_flip(extremes, img_shape, direction='horizontal'):
    assert direction in ['horizontal', 'vertical']
    flipped = extremes.clone()
    if direction == 'horizontal':
        w = img_shape[1]
        flipped[..., 0::8] = w - extremes[..., 0::8]
        flipped[..., 2::8] = w - extremes[..., 6::8]
        flipped[..., 3::8] = extremes[..., 7::8]
        flipped[..., 4::8] = w - extremes[..., 4::8]
        flipped[..., 6::8] = w - extremes[..., 2::8]
        flipped[..., 7::8] = extremes[..., 3::8]
    else:
        h = img_shape[0]
        flipped[..., 1::8] = h - extremes[..., 5::8]
        flipped[..., 0::8] = extremes[..., 4::8]
        flipped[..., 3::8] = h - extremes[..., 3::8]
        flipped[..., 5::8] = h - extremes[..., 1::8]
        flipped[..., 4::8] = extremes[..., 0::8]
        flipped[..., 7::8] = h - extremes[..., 7::8]
    return flipped

def kps_flip(kps, img_shape, direction='horizontal'):
    assert direction in ['horizontal', 'vertical']
    keypoint_flip_idx = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                         [11, 12], [13, 14], [15, 16]]

    flipped = kps.clone()
    if direction == 'horizontal':
        dim = img_shape[1]
        idx = 0
    else:
        dim = img_shape[0]
        idx = 1
    if flipped.size(0) > 0:
        flipped[:, idx::2] = dim - flipped[:, idx::2]
        flipped = flipped.reshape(flipped.shape[0], -1, 2)
        for e in keypoint_flip_idx:
            flipped[:, e[0]], flipped[:, e[1]] = flipped[:, e[1]].clone(), flipped[:, e[0]].clone()
        flipped = flipped.reshape(flipped.shape[0], -1)

    return flipped


def bbox_mapping(bboxes,
                 img_shape,
                 scale_factor,
                 flip,
                 flip_direction='horizontal'):
    """Map bboxes from the original image scale to testing scale"""
    new_bboxes = bboxes * bboxes.new_tensor(scale_factor)
    if flip:
        new_bboxes = bbox_flip(new_bboxes, img_shape, flip_direction)
    return new_bboxes


def bbox_mapping_back(bboxes,
                      img_shape,
                      scale_factor,
                      flip,
                      flip_direction='horizontal'):
    """Map bboxes from testing scale to original image scale"""
    new_bboxes = bbox_flip(bboxes, img_shape,
                           flip_direction) if flip else bboxes
    new_bboxes = new_bboxes.view(-1, 4) / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


def instance_mapping_back(bboxes,
                          vectors,
                          img_shape,
                          scale_factor,
                          flip,
                          task,
                          flip_direction='horizontal'):
    
    new_bboxes = bbox_flip(bboxes, img_shape, flip_direction) if flip else bboxes
    new_bboxes = new_bboxes.view(-1, 4) / new_bboxes.new_tensor(scale_factor)
    if task == 'bbox':
        new_vectors = extreme_flip(vectors, img_shape, flip_direction) if flip else vectors
        vect_scale_factor = new_vectors.new_tensor(scale_factor[:2]).repeat(int(new_vectors.size(1)/2))
        new_vectors_view = new_vectors.view(-1, new_vectors.size(1))/vect_scale_factor
    elif task == 'segm':
        new_vectors = polygon_flip(vectors, img_shape, flip_direction) if flip else vectors
        vect_scale_factor = new_vectors.new_tensor(scale_factor[:2]).repeat(int(new_vectors.size(1)/2))
        new_vectors_view = new_vectors.view(-1, new_vectors.size(1))/vect_scale_factor
    elif task == 'pose_bbox' or task == 'pose_kbox':
        new_vectors = kps_flip(vectors, img_shape, flip_direction) if flip else vectors
        vect_scale_factor = new_vectors.new_tensor(scale_factor[:2]).repeat(int(new_vectors.size(1)/2))
        new_vectors_view = new_vectors.view(-1, new_vectors.size(1))/vect_scale_factor
    return new_bboxes.view(bboxes.shape), new_vectors_view.view(new_vectors.shape)

def bbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 5), [batch_ind, x1, y1, x2, y2]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :4]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 5))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def roi2bbox(rois):
    """Convert rois to bounding box format

    Args:
        rois (torch.Tensor): RoIs with the shape (n, 5) where the first
            column indicates batch id of each RoI.

    Returns:
        list[torch.Tensor]: Converted boxes of corresponding rois.
    """
    bbox_list = []
    img_ids = torch.unique(rois[:, 0].cpu(), sorted=True)
    for img_id in img_ids:
        inds = (rois[:, 0] == img_id.item())
        bbox = rois[inds, 1:]
        bbox_list.append(bbox)
    return bbox_list


def bbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (Tensor): shape (n, 5)
        labels (Tensor): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]

def bbox_extreme2result(bboxes, extremes, labels, num_classes):
    if bboxes.shape[0] == 0:
        return [[np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)],
                [np.zeros((0, 8), dtype=np.float32) for i in range(num_classes)]]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        extremes = extremes.cpu().numpy()
        return [[bboxes[labels == i, :] for i in range(num_classes)],
                [extremes[labels == i, :] for i in range(num_classes)]]

def bbox_poly2result(bboxes, polygons, labels, num_classes, num_contour_points):
    if bboxes.shape[0] == 0:
        return [[np.zeros((0, 5), dtype=np.float32) for i in range(num_classes)],
                [np.zeros((0, num_contour_points*2), dtype=np.float32) for i in range(num_classes)]]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        polygons = polygons.cpu().numpy()
        return [[bboxes[labels == i, :] for i in range(num_classes)],
                [polygons[labels == i, :] for i in range(num_classes)]]


def distance2bbox(points, distance, max_shape=None):
    """Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image.

    Returns:
        Tensor: Decoded bboxes.
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return torch.stack([x1, y1, x2, y2], -1)


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """Decode bounding box based on distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format
        max_dis (float): Upper bound of the distance.
        eps (float): a small value to ensure target < max_dis, instead <=

    Returns:
        Tensor: Decoded distances.
    """
    left = points[:, 0] - bbox[:, 0]
    top = points[:, 1] - bbox[:, 1]
    right = bbox[:, 2] - points[:, 0]
    bottom = bbox[:, 3] - points[:, 1]
    if max_dis is not None:
        left = left.clamp(min=0, max=max_dis - eps)
        top = top.clamp(min=0, max=max_dis - eps)
        right = right.clamp(min=0, max=max_dis - eps)
        bottom = bottom.clamp(min=0, max=max_dis - eps)
    return torch.stack([left, top, right, bottom], -1)
