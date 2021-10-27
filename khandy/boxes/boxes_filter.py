import numpy as np


def filter_small_boxes(boxes, min_width, min_height):
    """Remove all boxes with side smaller than min size. 

    Args:
        boxes: a numpy array with shape [N, 4] holding N boxes.
        min_width (float): minimum width
        min_height (float): minimum height

    Returns:
        keep: indices of the boxes that have width larger than
            min_width and height larger than min_height.

    References:
        `_filter_boxes` in py-faster-rcnn
        `prune_small_boxes` in TensorFlow object detection API.
        `structures.Boxes.nonempty` in detectron2
        `ops.boxes.remove_small_boxes` in torchvision
    """
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1] 
    keep = (widths >= min_width)
    keep &= (heights >= min_height)
    return np.nonzero(keep)[0]
    