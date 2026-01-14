import warnings
import sys
from typing import Optional, Union

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal
    
if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence
    
import numpy as np

from .boxes_overlap import pairwise_overlap_ratio


def find_max_area_box(boxes: np.ndarray) -> int:
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


def sort_boxes_by_area(
    boxes: np.ndarray, 
    reverse: bool = False
) -> np.ndarray:
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


def filter_boxes_by_size(
    boxes: np.ndarray,
    min_width: Union[int, float, None] = None,
    min_height: Union[int, float, None] = None
) -> np.ndarray:
    """Filter bounding boxes based on minimum width and height requirements.

    Args:
        boxes (numpy array): A 2D numpy array of shape (N, 4) representing a group of bounding boxes.
            Each row in the array represents a bounding box with the coordinates of the top-left and
            bottom-right corners.
        min_width (int, float or None): The minimum allowed width of the bounding boxes. If set to None,
            no width requirement will be enforced.
        min_height (int, float or None): The minimum allowed height of the bounding boxes. If set to None,
            no height requirement will be enforced.

    Returns:
        int array: A numpy array of integers representing the indices of the bounding boxes in the
            input 'boxes' array that meet the minimum width and height requirements. The indices can be
            used to filter the bounding boxes based on their dimensions.

    References:
        `_filter_boxes` in py-faster-rcnn
        `prune_small_boxes` in TensorFlow object detection API.
        `structures.Boxes.nonempty` in detectron2
        `ops.boxes.remove_small_boxes` in torchvision
    """
    mask = np.ones((len(boxes),), dtype=bool)
    if min_width is not None:
        widths = boxes[:, 2] - boxes[:, 0]
        mask &= widths >= min_width
    if min_height is not None:
        heights = boxes[:, 3] - boxes[:, 1]
        mask &= heights >= min_height
    return np.nonzero(mask)[0]


def filter_small_boxes(
    boxes: np.ndarray,
    min_width: Union[int, float, None] = None,
    min_height: Union[int, float, None] = None
) -> np.ndarray:
    warnings.warn(
        "filter_small_boxes will be deprecated, use filter_boxes_by_size instead.",
        DeprecationWarning,
    )
    return filter_boxes_by_size(boxes, min_width, min_height)


def filter_boxes_by_area(
    boxes: np.ndarray, 
    min_area: Union[int, float]
) -> np.ndarray:
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    mask = widths * heights >= min_area
    return np.nonzero(mask)[0]


def filter_boxes_by_ar(
    boxes: np.ndarray,
    min_ar: Union[int, float, None] = None,
    max_ar: Union[int, float, None] = None
) -> np.ndarray:
    if min_ar is not None and max_ar is not None:
        assert min_ar < max_ar
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    mask = heights > 0
    ar = np.zeros_like(widths, dtype=float)
    ar[mask] = widths[mask] / heights[mask]
    if min_ar is not None:
        mask &= ar >= min_ar
    if max_ar is not None:
        mask &= ar <= max_ar
    return np.nonzero(mask)[0]


def filter_boxes_outside(boxes, reference_box):
    """Filters bounding boxes that fall outside reference box.

    References:
        `prune_outside_window` in TensorFlow object detection API.
    """
    x_min, y_min, x_max, y_max = reference_box[:4]
    mask = (
        (boxes[:, 0] >= x_min)
        & (boxes[:, 1] >= y_min)
        & (boxes[:, 2] <= x_max)
        & (boxes[:, 3] <= y_max)
    )
    return np.nonzero(mask)[0]


def filter_boxes_completely_outside(boxes, reference_box):
    """Filters bounding boxes that fall completely outside of reference box.

    References:
        `prune_completely_outside_window` in TensorFlow object detection API.
    """
    x_min, y_min, x_max, y_max = reference_box[:4]
    mask = (
        (boxes[:, 0] < x_max)
        & (boxes[:, 1] < y_max)
        & (boxes[:, 2] > x_min)
        & (boxes[:, 3] > y_min)
    )
    return np.nonzero(mask)[0]


def filter_boxes_by_overlap(
    boxes: np.ndarray, 
    reference_box: Sequence[Union[int, float]],
    ratio_type: str = 'iou',
    thresh: float = 0.5
) -> np.ndarray:
    """Filters bounding boxes based on their overlap ratio with a reference box.

    Args:
        boxes (np.ndarray): Array of bounding boxes with shape (N, 4), 
            where each box is represented as [x_min, y_min, x_max, y_max].
        reference_box (Sequence[Union[int, float]]): A single reference box 
            represented as [x_min, y_min, x_max, y_max].
        ratio_type (str, optional): Type of overlap ratio to compute. 
            Options are 'ioa' (intersection over area) or 'iou' 
            (intersection over union). Default is 'iou'.
        thresh (float, optional): Minimum overlap ratio threshold 
            for filtering boxes. Default is 0.5.

    Returns:
        np.ndarray: Indices of boxes that satisfy the overlap ratio condition.
    """
    reference_boxes = np.array(reference_box[:4]).reshape(1, -1)
    overlap_ratios = pairwise_overlap_ratio(boxes, reference_boxes, ratio_type)
    mask = overlap_ratios >= thresh
    return np.nonzero(mask)[0]


def non_max_suppression(
    boxes: np.ndarray,
    scores: np.ndarray,
    thresh: Union[float, Sequence[float], np.ndarray],
    classes: Optional[np.ndarray] = None,
    ratio_type: Union[Literal['iou', 'ios'], Sequence[Literal['iou', 'ios']]] = "iou"
) -> np.ndarray:
    """Perform Non-Maximum Suppression (NMS) on bounding boxes.

    Args:
        boxes (np.ndarray): Array of shape (N, 4), each row is [x_min, y_min, x_max, y_max].
        scores (np.ndarray): Array of shape (N,) with confidence scores for each box.
        thresh (float or Sequence or np.ndarray): Overlap threshold for suppression.
            - If float, the same threshold is used for all boxes.
            - If array or sequence, should be per-class thresholds (requires `classes`).
            Boxes with overlap ratio greater than `thresh` are suppressed.
        classes (np.ndarray, optional): Array of shape (N,) with class indices for each box.
            If provided, NMS is performed independently per class (using coordinate offsets).
        ratio_type (str or Sequence): Overlap ratio type, one of:
            - "iou" or "union": intersection over union (default)
            - "ios" or "min": intersection over smaller area of two boxes
            - "iom": the same as "ios", but kept for backward compatibility, will be removed in the future.
            - If a sequence, should be per-class ratio types (requires `classes`).

    Returns:
        np.ndarray: Indices of boxes to keep after NMS, sorted by descending score.

    Raises:
        ValueError: If `thresh` or `ratio_type` is not a float/str and `classes` is None.
        ValueError: If `ratio_type` is not one of the supported types.

    References:
        - `py_cpu_nms` in py-faster-rcnn
        - torchvision.ops.nms
        - torchvision.ops.batched_nms
    """
    if not isinstance(thresh, float) and classes is None:
        raise ValueError('When thresh is not a float, classes should not be None')
    if not isinstance(ratio_type, str) and classes is None:
        raise ValueError('When ratio_type is not a str, classes should not be None')
    if ratio_type == 'iom':
        ratio_type = 'ios'
        warnings.warn(
            "ratio_type='iom' is deprecated and will be removed in the future. "
            "Use 'ios' instead.",
            DeprecationWarning,
        )
        
    boxes = np.asarray(boxes)
    scores = np.asarray(scores)
    if boxes.size == 0:
        return np.empty((0,), dtype=np.int64)
    if classes is not None:
        classes = np.asarray(classes)
        # strategy: in order to perform NMS independently per class,
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = np.max(boxes)
        offsets = classes * (max_coordinate + 1)
        if offsets.ndim == 1:
            offsets = offsets[:, None]
        # cannot use inplace add
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

        if isinstance(ratio_type, str):
            _ratio_type = ratio_type
        else:
            _ratio_type = ratio_type[classes[i]]
        if _ratio_type in ["iou", "union"]:
            intersect_areas /= (areas[i] + areas[order[1:]] - intersect_areas)
        elif _ratio_type in ["ios", "min"]:
            intersect_areas /= np.minimum(areas[i], areas[order[1:]])
        else:
            raise ValueError(f"Unsupported ratio_type. Got {_ratio_type}")
        
        if isinstance(thresh, float):
            _thresh = thresh
        else:
            _thresh = thresh[classes[i]]
        inds = np.nonzero(intersect_areas <= _thresh)[0]

        order = order[inds + 1]

    return np.asarray(keep)
