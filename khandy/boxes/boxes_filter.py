import numpy as np


def find_max_area_box(boxes):
    """Find the index of the bounding box with the maximum area.

    Args:
        boxes (numpy array): A 2D numpy array of shape (N, 4) representing a group of bounding boxes. 
            Each row in the array represents a bounding box with the coordinates of the top-left and 
            bottom-right corners.
            
    Returns:
        int: An integer value representing the index of the bounding box with the maximum area in 
            the input 'boxes' array.
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    return np.argmax(areas)


def sort_boxes_by_area(boxes, reverse=False):
    """Sort bounding boxes based on their areas.

    Args:
        boxes (numpy array): A 2D numpy array of shape (N, 4) representing a group of bounding boxes. 
            Each row in the array represents a bounding box with the coordinates of the top-left and 
            bottom-right corners.
            
        reverse (bool): A boolean value indicating whether to sort the bounding boxes in descending order 
            instead of ascending order of their areas. Defaults to False.
    
    Returns:
        int array: A numpy array of integers representing the indices of the bounding boxes sorted 
            by their areas in the input 'boxes' array. The indices can be used to sort the bounding boxes 
            based on their areas. If 'reverse' is set to True, the bounding boxes will be sorted in 
            descending order of their areas.
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    if not reverse:
        return np.argsort(areas)
    else:
        return np.argsort(areas)[::-1]
    

def filter_small_boxes(boxes, min_width, min_height):
    """Filter bounding boxes based on minimum width and height requirements.

    Args:
        boxes (numpy array): A 2D numpy array of shape (N, 4) representing a group of bounding boxes. 
            Each row in the array represents a bounding box with the coordinates of the top-left and 
            bottom-right corners.
        min_width (float or None): The minimum allowed width of the bounding boxes. If set to None, 
            no width requirement will be enforced.
        min_height (float or None): The minimum allowed height of the bounding boxes. If set to None, 
            no height requirement will be enforced.
    
    Returns:
        int array: A numpy array of integers representing the indices of the bounding boxes in the 
            input 'boxes' array that meet the minimum width and height requirements. The indices can be 
            used to filter the bounding boxes based on their dimensions.
        
    Raises:
        ValueError: If both min_width and min_height are set to None at the same time.
        
    References:
        `_filter_boxes` in py-faster-rcnn
        `prune_small_boxes` in TensorFlow object detection API.
        `structures.Boxes.nonempty` in detectron2
        `ops.boxes.remove_small_boxes` in torchvision
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    mask = np.ones_like(widths, dtype=bool)
    if min_width is not None:
        mask &= (widths >= min_width)
    if min_height is not None:
        mask &= (heights >= min_height)
    indices = np.nonzero(mask)[0]
    return indices
    

def filter_boxes_outside(boxes, reference_box):
    """Filters bounding boxes that fall outside reference box.
    
    References:
        `prune_outside_window` in TensorFlow object detection API.
    """
    x_min, y_min, x_max, y_max = reference_box[:4]
    mask = ((boxes[:, 0] >= x_min) & (boxes[:, 1] >= y_min) &
            (boxes[:, 2] <= x_max) & (boxes[:, 3] <= y_max))
    return np.nonzero(mask)[0]


def filter_boxes_completely_outside(boxes, reference_box):
    """Filters bounding boxes that fall completely outside of reference box.
    
    References:
        `prune_completely_outside_window` in TensorFlow object detection API.
    """
    x_min, y_min, x_max, y_max = reference_box[:4]
    mask = ((boxes[:, 0] < x_max) & (boxes[:, 1] < y_max) &
            (boxes[:, 2] > x_min) & (boxes[:, 3] > y_min))
    return np.nonzero(mask)[0]
    

def non_max_suppression(boxes, scores, thresh, classes=None, ratio_type="iou"):
    """Greedily select boxes with high confidence
    Args:
        boxes: [[x_min, y_min, x_max, y_max], ...]
        scores: object confidence
        thresh: retain overlap_ratio <= thresh
        classes: class labels
        
    Returns:
        indices to keep
        
    References:
        `py_cpu_nms` in py-faster-rcnn
        torchvision.ops.nms
        torchvision.ops.batched_nms
    """

    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    if classes is not None:
        # strategy: in order to perform NMS independently per class,
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = np.max(boxes)
        offsets = classes * (max_coordinate + 1)
        if offsets.ndim == 1:
            offsets = offsets[:, None]
        boxes = boxes + offsets

    x_mins = boxes[:, 0]
    y_mins = boxes[:, 1]
    x_maxs = boxes[:, 2]
    y_maxs = boxes[:, 3]
    areas = (x_maxs - x_mins) * (y_maxs - y_mins)
    order = scores.flatten().argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        max_x_mins = np.maximum(x_mins[i], x_mins[order[1:]])
        max_y_mins = np.maximum(y_mins[i], y_mins[order[1:]])
        min_x_maxs = np.minimum(x_maxs[i], x_maxs[order[1:]])
        min_y_maxs = np.minimum(y_maxs[i], y_maxs[order[1:]])
        widths = np.maximum(0, min_x_maxs - max_x_mins)
        heights = np.maximum(0, min_y_maxs - max_y_mins)
        intersect_areas = widths * heights
        
        if ratio_type in ["union", 'iou']:
            ratio = intersect_areas / (areas[i] + areas[order[1:]] - intersect_areas)
        elif ratio_type == "min":
            ratio = intersect_areas / np.minimum(areas[i], areas[order[1:]])
        else:
            raise ValueError('Unsupported ratio_type. Got {}'.format(ratio_type))
            
        inds = np.nonzero(ratio <= thresh)[0]
        order = order[inds + 1]
    return np.asarray(keep)
    