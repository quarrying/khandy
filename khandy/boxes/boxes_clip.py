import numpy as np


def clip_boxes(boxes, reference_box, copy=True):
    """Clip boxes to reference box.
    
    References:
        `clip_to_window` in TensorFlow object detection API.
    """
    if copy:
        boxes = boxes.copy()
    ref_x_min, ref_y_min, ref_x_max, ref_y_max = reference_box[:4]
    lower = np.array([ref_x_min, ref_y_min, ref_x_min, ref_y_min])
    upper = np.array([ref_x_max, ref_y_max, ref_x_max, ref_y_max])
    np.clip(boxes[..., :4], lower, upper, boxes[..., :4])
    return boxes
    
    
def clip_boxes_to_image(boxes, image_width, image_height, subpixel=True, copy=True):
    """Clip boxes to image boundaries.
    
    References:
        `clip_boxes` in py-faster-rcnn
        `core.boxes_op_list.clip_to_window` in TensorFlow object detection API.
        `structures.Boxes.clip` in detectron2
        
    Notes:
        Equivalent to `clip_boxes(boxes, [0,0,image_width-1,image_height-1], copy)`
    """
    if not subpixel:
        image_width -= 1
        image_height -= 1
    reference_box = [0, 0, image_width, image_height]
    return clip_boxes(boxes, reference_box, copy)
