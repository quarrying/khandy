import numpy as np


def convert_xyxy_to_xywh(boxes, copy=True):
    """Convert [x_min, y_min, x_max, y_max] format to [x_min, y_min, width, height] format.
    """
    if copy:
        boxes = boxes.copy()
    boxes[..., 2:4] -= boxes[..., 0:2]
    return boxes


def convert_xywh_to_xyxy(boxes, copy=True):
    """Convert [x_min, y_min, width, height] format to [x_min, y_min, x_max, y_max] format.
    """
    if copy:
        boxes = boxes.copy()
    boxes[..., 2:4] += boxes[..., 0:2]
    return boxes


def convert_xywh_to_cxcywh(boxes, copy=True):
    """Convert [x_min, y_min, width, height] format to [cx, cy, width, height] format.
    """
    if copy:
        boxes = boxes.copy()
    boxes[..., 0:2] += boxes[..., 2:4] * 0.5
    return boxes
    
    
def convert_cxcywh_to_xywh(boxes, copy=True):
    """Convert [cx, cy, width, height] format to [x_min, y_min, width, height] format.
    """
    if copy:
        boxes = boxes.copy()
    boxes[..., 0:2] -= boxes[..., 2:4] * 0.5
    return boxes
    
    
def convert_xyxy_to_cxcywh(boxes, copy=True):
    """Convert [x_min, y_min, x_max, y_max] format to [cx, cy, width, height] format.
    """
    if copy:
        boxes = boxes.copy()
    boxes[..., 2:4] -= boxes[..., 0:2]
    boxes[..., 0:2] += boxes[..., 2:4] * 0.5
    return boxes


def convert_cxcywh_to_xyxy(boxes, copy=True):
    """Convert [cx, cy, width, height] format to [x_min, y_min, x_max, y_max] format.
    """
    if copy:
        boxes = boxes.copy()
    boxes[..., 0:2] -= boxes[..., 2:4] * 0.5
    boxes[..., 2:4] += boxes[..., 0:2]
    return boxes


def convert_boxes_format(boxes, in_fmt, out_fmt, copy=True):
    """Converts boxes from given in_fmt to out_fmt.

    Supported in_fmt and out_fmt are:
        'xyxy': boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right.
        'xywh' : boxes are represented via corner, width and height, x1, y2 being top left, w, h being width and height.
        'cxcywh' : boxes are represented via centre, width and height, cx, cy being center of box, w, h
            being width and height.

    Args:
        boxes: boxes which will be converted.
        in_fmt (str): Input format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh'].
        out_fmt (str): Output format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh']

    Returns:
        boxes: Boxes into converted format.

    References:
        torchvision.ops.box_convert
    """
    allowed_fmts = ("xyxy", "xywh", "cxcywh")
    if in_fmt not in allowed_fmts:
        raise ValueError(f"Unsupported `in_fmt`, got {in_fmt}")
    if out_fmt not in allowed_fmts:
        raise ValueError(f"Unsupported `out_fmt`, got {out_fmt}")
    if in_fmt == out_fmt:
        return boxes.copy() if copy else boxes

    convert_map = {
        ('xyxy', 'xywh'):    convert_xyxy_to_xywh,
        ('xywh', 'xyxy'):    convert_xywh_to_xyxy,
        ('xywh', 'cxcywh'):  convert_xywh_to_cxcywh,
        ('cxcywh', 'xywh'):  convert_cxcywh_to_xywh,
        ('xyxy', 'cxcywh'):  convert_xyxy_to_cxcywh,
        ('cxcywh', 'xyxy'):  convert_cxcywh_to_xyxy,
    }
    convert_fn = convert_map[(in_fmt, out_fmt)]
    return convert_fn(boxes, copy=copy)
