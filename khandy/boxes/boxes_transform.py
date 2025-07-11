import numpy as np
from .boxes_utils import assert_and_normalize_shape


def flip_boxes(boxes, x_center=0, y_center=0, direction='h'):
    """
    Args:
        boxes: (N, 4+K)
        x_center: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
        y_center: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
        direction: str
    """
    assert direction in ['x', 'h', 'horizontal',
                         'y', 'v', 'vertical', 
                         'o', 'b', 'both']
    boxes = np.asarray(boxes, np.float32)
    ret_boxes = boxes.copy()
    
    x_center = np.asarray(x_center, np.float32)
    y_center = np.asarray(y_center, np.float32)
    x_center = assert_and_normalize_shape(x_center, boxes.shape[0])
    y_center = assert_and_normalize_shape(y_center, boxes.shape[0])
    
    if direction in ['o', 'b', 'both', 'x', 'h', 'horizontal']:
        ret_boxes[:, 0] = 2 * x_center - boxes[:, 2] 
        ret_boxes[:, 2] = 2 * x_center - boxes[:, 0]
    if direction in ['o', 'b', 'both', 'y', 'v', 'vertical']:
        ret_boxes[:, 1] = 2 * y_center - boxes[:, 3]
        ret_boxes[:, 3] = 2 * y_center - boxes[:, 1]
    return ret_boxes
    
    
def fliplr_boxes(boxes, x_center=0, y_center=0):
    """
    Args:
        boxes: (N, 4+K)
        x_center: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
        y_center: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
    """
    boxes = np.asarray(boxes, np.float32)
    ret_boxes = boxes.copy()
    
    x_center = np.asarray(x_center, np.float32)
    y_center = np.asarray(y_center, np.float32)
    x_center = assert_and_normalize_shape(x_center, boxes.shape[0])
    y_center = assert_and_normalize_shape(y_center, boxes.shape[0])
     
    ret_boxes[:, 0] = 2 * x_center - boxes[:, 2] 
    ret_boxes[:, 2] = 2 * x_center - boxes[:, 0]
    return ret_boxes
    
    
def flipud_boxes(boxes, x_center=0, y_center=0):
    """
    Args:
        boxes: (N, 4+K)
        x_center: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
        y_center: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
    """
    boxes = np.asarray(boxes, np.float32)
    ret_boxes = boxes.copy()
    
    x_center = np.asarray(x_center, np.float32)
    y_center = np.asarray(y_center, np.float32)
    x_center = assert_and_normalize_shape(x_center, boxes.shape[0])
    y_center = assert_and_normalize_shape(y_center, boxes.shape[0])
    
    ret_boxes[:, 1] = 2 * y_center - boxes[:, 3]
    ret_boxes[:, 3] = 2 * y_center - boxes[:, 1]
    return ret_boxes
    
    
def transpose_boxes(boxes, x_center=0, y_center=0):
    """
    Args:
        boxes: (N, 4+K)
        x_center: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
        y_center: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
    """
    boxes = np.asarray(boxes, np.float32)
    ret_boxes = boxes.copy()
    
    x_center = np.asarray(x_center, np.float32)
    y_center = np.asarray(y_center, np.float32)
    x_center = assert_and_normalize_shape(x_center, boxes.shape[0])
    y_center = assert_and_normalize_shape(y_center, boxes.shape[0])
    
    shift = x_center - y_center
    ret_boxes[:, 0] = boxes[:, 1] + shift
    ret_boxes[:, 1] = boxes[:, 0] - shift
    ret_boxes[:, 2] = boxes[:, 3] + shift
    ret_boxes[:, 3] = boxes[:, 2] - shift
    return ret_boxes


def flip_boxes_in_image(boxes, image_width, image_height, direction='h'):
    """
    Args:
        boxes: (N, 4+K)
        image_width: int
        image_width: int
        direction: str
        
    References:
        `core.bbox.bbox_flip` in mmdetection
        `datasets.pipelines.RandomFlip.bbox_flip` in mmdetection
    """
    x_center = (image_width - 1) * 0.5
    y_center = (image_height - 1) * 0.5
    ret_boxes = flip_boxes(boxes, x_center, y_center, direction=direction)
    return ret_boxes
    
    
def rot90_boxes_in_image(boxes, image_width, image_height, n=1):
    """Rotate boxes counter-clockwise by 90 degrees.
    
    References:
        np.rot90
        cv2.rotate
        tf.image.rot90
    """
    n = n % 4
    if n == 0:
        ret_boxes = boxes.copy()
    elif n == 1:
        ret_boxes = transpose_boxes(boxes)
        ret_boxes = flip_boxes_in_image(ret_boxes, image_width, image_height, 'v')
    elif n == 2:
        ret_boxes = flip_boxes_in_image(boxes, image_width, image_height, 'o')
    else:
        ret_boxes = transpose_boxes(boxes)
        ret_boxes = flip_boxes_in_image(ret_boxes, image_width, image_height, 'h');
    return ret_boxes
    
    
def translate_boxes(boxes, x_shift=0, y_shift=0, copy=True):
    """translate boxes coordinates in x and y dimensions.
    
    Args:
        boxes: (N, 4+K)
        x_shift: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
            shift in x dimension
        y_shift: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
            shift in y dimension
        copy: bool
        
    References:
        `datasets.pipelines.RandomCrop` in mmdetection
    """
    boxes = np.array(boxes, dtype=np.float32, copy=copy)
    
    x_shift = np.asarray(x_shift, np.float32)
    y_shift = np.asarray(y_shift, np.float32)

    x_shift = assert_and_normalize_shape(x_shift, boxes.shape[0])
    y_shift = assert_and_normalize_shape(y_shift, boxes.shape[0])
    
    boxes[:, 0] += x_shift
    boxes[:, 1] += y_shift
    boxes[:, 2] += x_shift
    boxes[:, 3] += y_shift
    return boxes
    
    
def adjust_boxes(boxes, x_min_shift, y_min_shift, x_max_shift, y_max_shift, copy=True):
    """
    Args:
        boxes: (N, 4+K)
        x_min_shift: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
            shift (x_min, y_min) in x dimension
        y_min_shift: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
            shift (x_min, y_min) in y dimension
        x_max_shift: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
            shift (x_max, y_max) in x dimension
        y_max_shift: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
            shift (x_max, y_max) in y dimension
        copy: bool
    """
    boxes = np.array(boxes, dtype=np.float32, copy=copy)

    x_min_shift = np.asarray(x_min_shift, np.float32)
    y_min_shift = np.asarray(y_min_shift, np.float32)
    x_max_shift = np.asarray(x_max_shift, np.float32)
    y_max_shift = np.asarray(y_max_shift, np.float32)

    x_min_shift = assert_and_normalize_shape(x_min_shift, boxes.shape[0])
    y_min_shift = assert_and_normalize_shape(y_min_shift, boxes.shape[0])
    x_max_shift = assert_and_normalize_shape(x_max_shift, boxes.shape[0])
    y_max_shift = assert_and_normalize_shape(y_max_shift, boxes.shape[0])
    
    boxes[:, 0] += x_min_shift
    boxes[:, 1] += y_min_shift
    boxes[:, 2] += x_max_shift
    boxes[:, 3] += y_max_shift
    return boxes
    
    
def inflate_or_deflate_boxes(boxes, width_delta=0, height_delta=0, copy=True):
    """
    Args:
        boxes: (N, 4+K)
        width_delta: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
        height_delta: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
        copy: bool
    """
    boxes = np.array(boxes, dtype=np.float32, copy=copy)

    width_delta = np.asarray(width_delta, np.float32)
    height_delta = np.asarray(height_delta, np.float32)

    width_delta = assert_and_normalize_shape(width_delta, boxes.shape[0])
    height_delta = assert_and_normalize_shape(height_delta, boxes.shape[0])
    
    half_width_delta = width_delta * 0.5
    half_height_delta = height_delta * 0.5
    boxes[:, 0] -= half_width_delta
    boxes[:, 1] -= half_height_delta
    boxes[:, 2] += half_width_delta
    boxes[:, 3] += half_height_delta
    return boxes
    

def inflate_boxes_to_square(boxes, copy=True):
    """Inflate boxes to square
    Args:
        boxes: (N, 4+K)
        copy: bool
    """
    boxes = np.array(boxes, dtype=np.float32, copy=copy)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    max_side_lengths = np.maximum(widths, heights)
    
    width_deltas = np.subtract(max_side_lengths, widths, widths)
    height_deltas = np.subtract(max_side_lengths, heights, heights)
    width_deltas *= 0.5
    height_deltas *= 0.5
    boxes[:, 0] -= width_deltas
    boxes[:, 1] -= height_deltas
    boxes[:, 2] += width_deltas
    boxes[:, 3] += height_deltas
    return boxes
    

def deflate_boxes_to_square(boxes, copy=True):
    """Deflate boxes to square
    Args:
        boxes: (N, 4+K)
        copy: bool
    """
    boxes = np.array(boxes, dtype=np.float32, copy=copy)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    min_side_lengths = np.minimum(widths, heights)
    
    width_deltas = np.subtract(min_side_lengths, widths, widths)
    height_deltas = np.subtract(min_side_lengths, heights, heights)
    width_deltas *= 0.5
    height_deltas *= 0.5
    boxes[:, 0] -= width_deltas
    boxes[:, 1] -= height_deltas
    boxes[:, 2] += width_deltas
    boxes[:, 3] += height_deltas
    return boxes


def scale_boxes(boxes, x_scale=1, y_scale=1, x_center=0, y_center=0, copy=True):
    """Scale boxes coordinates in x and y dimensions.
    
    Args:
        boxes: (N, 4+K)
        x_scale: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
            scale factor in x dimension
        y_scale: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
            scale factor in y dimension
        x_center: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
        y_center: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
        
    References:
        `core.box_list_ops.scale` in TensorFlow object detection API
        `utils.box_list_ops.scale` in TensorFlow object detection API
        `datasets.pipelines.Resize._resize_bboxes` in mmdetection
        `core.anchor.guided_anchor_target.calc_region` in mmdetection where comments may be misleading!
        `layers.mask_ops.scale_boxes` in detectron2
        `mmcv.bbox_scaling`
    """
    boxes = np.array(boxes, dtype=np.float32, copy=copy)

    x_scale = np.asarray(x_scale, np.float32)
    y_scale = np.asarray(y_scale, np.float32)
    x_scale = assert_and_normalize_shape(x_scale, boxes.shape[0])
    y_scale = assert_and_normalize_shape(y_scale, boxes.shape[0])
    
    x_center = np.asarray(x_center, np.float32)
    y_center = np.asarray(y_center, np.float32)
    x_center = assert_and_normalize_shape(x_center, boxes.shape[0])
    y_center = assert_and_normalize_shape(y_center, boxes.shape[0])
    
    x_shift = 1 - x_scale
    y_shift = 1 - y_scale
    x_shift *= x_center
    y_shift *= y_center
    
    boxes[:, 0] *= x_scale
    boxes[:, 1] *= y_scale
    boxes[:, 2] *= x_scale
    boxes[:, 3] *= y_scale
    boxes[:, 0] += x_shift
    boxes[:, 1] += y_shift
    boxes[:, 2] += x_shift
    boxes[:, 3] += y_shift
    return boxes
    
    
def scale_boxes_wrt_centers(boxes, x_scale=1, y_scale=1, copy=True):
    """
    Args:
        boxes: (N, 4+K)
        x_scale: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
            scale factor in x dimension
        y_scale: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
            scale factor in y dimension
            
    References:
        `core.anchor.guided_anchor_target.calc_region` in mmdetection where comments may be misleading!
        `layers.mask_ops.scale_boxes` in detectron2
        `mmcv.bbox_scaling`
    """
    boxes = np.array(boxes, dtype=np.float32, copy=copy)
    
    x_scale = np.asarray(x_scale, np.float32)
    y_scale = np.asarray(y_scale, np.float32)
    x_scale = assert_and_normalize_shape(x_scale, boxes.shape[0])
    y_scale = assert_and_normalize_shape(y_scale, boxes.shape[0])
    
    x_factor = (x_scale - 1) * 0.5
    y_factor = (y_scale - 1) * 0.5
    x_deltas = boxes[:, 2] - boxes[:, 0]
    y_deltas = boxes[:, 3] - boxes[:, 1]
    x_deltas *= x_factor
    y_deltas *= y_factor

    boxes[:, 0] -= x_deltas
    boxes[:, 1] -= y_deltas
    boxes[:, 2] += x_deltas
    boxes[:, 3] += y_deltas
    return boxes


def rotate_boxes(boxes, angle, x_center=0, y_center=0, scale=1, 
                 degrees=True, return_rotated_boxes=False):
    """
    Args:
        boxes: (N, 4+K)
        angle: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
        x_center: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
        y_center: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
        scale: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
            scale factor in x and y dimension
        degrees: bool
        return_rotated_boxes: bool
    """
    boxes = np.asarray(boxes, np.float32)
    
    angle = np.asarray(angle, np.float32)
    x_center = np.asarray(x_center, np.float32)
    y_center = np.asarray(y_center, np.float32)
    scale = np.asarray(scale, np.float32)
    
    angle = assert_and_normalize_shape(angle, boxes.shape[0])
    x_center = assert_and_normalize_shape(x_center, boxes.shape[0])
    y_center = assert_and_normalize_shape(y_center, boxes.shape[0])
    scale = assert_and_normalize_shape(scale, boxes.shape[0])
    
    if degrees:
        angle = np.deg2rad(angle)
    cos_val = scale * np.cos(angle)
    sin_val = scale * np.sin(angle)
    x_shift = x_center - x_center * cos_val + y_center * sin_val
    y_shift = y_center - x_center * sin_val - y_center * cos_val
    
    x_mins, y_mins = boxes[:,0], boxes[:,1]
    x_maxs, y_maxs = boxes[:,2], boxes[:,3]
    x00 = x_mins * cos_val - y_mins * sin_val + x_shift
    x10 = x_maxs * cos_val - y_mins * sin_val + x_shift
    x11 = x_maxs * cos_val - y_maxs * sin_val + x_shift
    x01 = x_mins * cos_val - y_maxs * sin_val + x_shift
    
    y00 = x_mins * sin_val + y_mins * cos_val + y_shift
    y10 = x_maxs * sin_val + y_mins * cos_val + y_shift
    y11 = x_maxs * sin_val + y_maxs * cos_val + y_shift
    y01 = x_mins * sin_val + y_maxs * cos_val + y_shift
    
    rotated_boxes = np.stack([x00, y00, x10, y10, x11, y11, x01, y01], axis=-1)
    ret_x_mins = np.min(rotated_boxes[:,0::2], axis=1)
    ret_y_mins = np.min(rotated_boxes[:,1::2], axis=1)
    ret_x_maxs = np.max(rotated_boxes[:,0::2], axis=1)
    ret_y_maxs = np.max(rotated_boxes[:,1::2], axis=1)
    
    if boxes.ndim == 4:
        ret_boxes = np.stack([ret_x_mins, ret_y_mins, ret_x_maxs, ret_y_maxs], axis=-1)
    else:
        ret_boxes = boxes.copy()
        ret_boxes[:, :4] = np.stack([ret_x_mins, ret_y_mins, ret_x_maxs, ret_y_maxs], axis=-1)
        
    if not return_rotated_boxes:
        return ret_boxes
    else:
        return ret_boxes, rotated_boxes
    
    
def rotate_boxes_wrt_centers(boxes, angle, scale=1, degrees=True,  
                             return_rotated_boxes=False):
    """
    Args:
        boxes: (N, 4+K)
        angle: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
        scale: array-like whose shape is (), (1,), (N,), (1, 1) or (N, 1)
            scale factor in x and y dimension
        degrees: bool
        return_rotated_boxes: bool
    """
    boxes = np.asarray(boxes, np.float32)
    
    angle = np.asarray(angle, np.float32)
    scale = np.asarray(scale, np.float32)
    angle = assert_and_normalize_shape(angle, boxes.shape[0])
    scale = assert_and_normalize_shape(scale, boxes.shape[0])
    
    if degrees:
        angle = np.deg2rad(angle)
    cos_val = scale * np.cos(angle)
    sin_val = scale * np.sin(angle)
    
    x_centers = boxes[:, 2] + boxes[:, 0]
    y_centers = boxes[:, 3] + boxes[:, 1]
    x_centers *= 0.5
    y_centers *= 0.5
    
    half_widths = boxes[:, 2] - boxes[:, 0]
    half_heights = boxes[:, 3] - boxes[:, 1]
    half_widths *= 0.5
    half_heights *= 0.5
    
    half_widths_cos = half_widths * cos_val
    half_widths_sin = half_widths * sin_val
    half_heights_cos = half_heights * cos_val
    half_heights_sin = half_heights * sin_val
    
    x00 = -half_widths_cos + half_heights_sin
    x10 = half_widths_cos + half_heights_sin
    x11 = half_widths_cos - half_heights_sin
    x01 = -half_widths_cos - half_heights_sin
    x00 += x_centers
    x10 += x_centers
    x11 += x_centers
    x01 += x_centers
    
    y00 = -half_widths_sin - half_heights_cos
    y10 = half_widths_sin - half_heights_cos
    y11 = half_widths_sin + half_heights_cos
    y01 = -half_widths_sin + half_heights_cos
    y00 += y_centers
    y10 += y_centers
    y11 += y_centers
    y01 += y_centers
    
    rotated_boxes = np.stack([x00, y00, x10, y10, x11, y11, x01, y01], axis=-1)
    ret_x_mins = np.min(rotated_boxes[:,0::2], axis=1)
    ret_y_mins = np.min(rotated_boxes[:,1::2], axis=1)
    ret_x_maxs = np.max(rotated_boxes[:,0::2], axis=1)
    ret_y_maxs = np.max(rotated_boxes[:,1::2], axis=1)
    
    if boxes.ndim == 4:
        ret_boxes = np.stack([ret_x_mins, ret_y_mins, ret_x_maxs, ret_y_maxs], axis=-1)
    else:
        ret_boxes = boxes.copy()
        ret_boxes[:, :4] = np.stack([ret_x_mins, ret_y_mins, ret_x_maxs, ret_y_maxs], axis=-1)
        
    if not return_rotated_boxes:
        return ret_boxes
    else:
        return ret_boxes, rotated_boxes
    

