"""
Box utility functions for FlashDet.
"""

import torch
from torchvision.ops import batched_nms


def distance2bbox(points, distance, max_shape=None):
    """
    Decode distance prediction to bounding box.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        distance (Tensor): Distance from the given point to 4
            boundaries (left, top, right, bottom).
        max_shape (tuple): Shape of the image (height, width).

    Returns:
        Tensor: Decoded bboxes [x1, y1, x2, y2].
    """
    x1 = points[..., 0] - distance[..., 0]
    y1 = points[..., 1] - distance[..., 1]
    x2 = points[..., 0] + distance[..., 2]
    y2 = points[..., 1] + distance[..., 3]
    
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    
    bboxes = torch.stack([x1, y1, x2, y2], -1)
    return bboxes


def bbox2distance(points, bbox, max_dis=None, eps=0.1):
    """
    Encode bounding box to distances.

    Args:
        points (Tensor): Shape (n, 2), [x, y].
        bbox (Tensor): Shape (n, 4), "xyxy" format.
        max_dis (float): Upper bound of the distance.
        eps (float): Small value to ensure target < max_dis.

    Returns:
        Tensor: Encoded distances [left, top, right, bottom].
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


def bbox_overlaps(bboxes1, bboxes2, mode="iou", is_aligned=False, eps=1e-6):
    """
    Calculate overlap between two sets of bboxes.

    Args:
        bboxes1 (Tensor): Shape (m, 4) in <x1, y1, x2, y2> format.
        bboxes2 (Tensor): Shape (n, 4) in <x1, y1, x2, y2> format.
        mode (str): "iou", "iof", or "giou".
        is_aligned (bool): If True, m must equal n.
        eps (float): Small value for numerical stability.

    Returns:
        Tensor: Shape (m, n) if not aligned, else (m,).
    """
    assert mode in ["iou", "iof", "giou"]
    
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    
    if rows * cols == 0:
        return bboxes1.new_zeros((rows, cols)) if not is_aligned else bboxes1.new_zeros((rows,))
    
    if is_aligned:
        assert rows == cols
        
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, 0] * wh[:, 1]
        
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        
        if mode == "iof":
            union = area1
        else:
            union = area1 + area2 - overlap
        
        union = torch.clamp(union, min=eps)
        ious = overlap / union
        
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[:, :2], bboxes2[:, :2])
            enclosed_rb = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
            enclosed_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
            enclosed_area = enclosed_wh[:, 0] * enclosed_wh[:, 1]
            enclosed_area = torch.clamp(enclosed_area, min=eps)
            ious = ious - (enclosed_area - union) / enclosed_area
        
        return ious
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[None, :, :2])
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[None, :, 2:])
        wh = (rb - lt).clamp(min=0)
        overlap = wh[:, :, 0] * wh[:, :, 1]
        
        area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])
        area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])
        
        if mode == "iof":
            union = area1[:, None]
        else:
            union = area1[:, None] + area2[None, :] - overlap
        
        union = torch.clamp(union, min=eps)
        ious = overlap / union
        
        if mode == "giou":
            enclosed_lt = torch.min(bboxes1[:, None, :2], bboxes2[None, :, :2])
            enclosed_rb = torch.max(bboxes1[:, None, 2:], bboxes2[None, :, 2:])
            enclosed_wh = (enclosed_rb - enclosed_lt).clamp(min=0)
            enclosed_area = enclosed_wh[:, :, 0] * enclosed_wh[:, :, 1]
            enclosed_area = torch.clamp(enclosed_area, min=eps)
            ious = ious - (enclosed_area - union) / enclosed_area
        
        return ious


def multiclass_nms(boxes, scores, score_thr=0.05, nms_thr=0.6, max_num=100, exclude_last_class=True):
    """
    Multi-class NMS matching official FlashDet implementation.
    
    Key difference from simple argmax approach: each anchor can produce detections
    for MULTIPLE classes if they exceed score_thr. This is important for 
    independent sigmoid classification (not softmax).

    Args:
        boxes (Tensor): Shape (n, 4).
        scores (Tensor): Shape (n, num_classes) or (n, num_classes+1) if background included.
        score_thr (float): Score threshold.
        nms_thr (float): NMS IoU threshold.
        max_num (int): Maximum number of detections.
        exclude_last_class (bool): If True, exclude the last class (assumed background).

    Returns:
        Tuple: (boxes, labels) where boxes is (k, 5) with scores.
    """
    num_classes = scores.size(1)
    num_boxes = boxes.size(0)
    device = boxes.device
    
    # Exclude background class (last class) if specified
    if exclude_last_class and num_classes > 1:
        scores = scores[:, :-1]
        num_classes -= 1
    
    if num_boxes == 0:
        return torch.zeros((0, 5), device=device), torch.zeros((0,), dtype=torch.long, device=device)
    
    # Official FlashDet approach: threshold per class, allowing multiple classes per anchor
    # Create (anchor, class) pairs for all scores above threshold
    valid_mask = scores > score_thr  # [n, num_classes]
    
    if not valid_mask.any():
        return torch.zeros((0, 5), device=device), torch.zeros((0,), dtype=torch.long, device=device)
    
    # Get indices of valid (anchor, class) pairs
    valid_indices = valid_mask.nonzero(as_tuple=False)  # [num_valid, 2]
    anchor_ids = valid_indices[:, 0]
    class_ids = valid_indices[:, 1]
    
    # Get corresponding boxes and scores
    boxes_expanded = boxes[anchor_ids]  # [num_valid, 4]
    scores_expanded = scores[anchor_ids, class_ids]  # [num_valid]
    
    # Use torchvision's batched_nms (NMS per class)
    keep = batched_nms(boxes_expanded, scores_expanded, class_ids, nms_thr)
    
    # Limit to max_num (-1 or 0 means return all)
    if max_num > 0:
        keep = keep[:max_num]
    
    boxes_out = boxes_expanded[keep]
    scores_out = scores_expanded[keep]
    labels_out = class_ids[keep]
    
    # Combine boxes and scores
    dets = torch.cat([boxes_out, scores_out.unsqueeze(1)], dim=1)
    
    return dets, labels_out
