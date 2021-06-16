import cv2
import numpy as np


def crop_or_pad(image, x_min, y_min, x_max, y_max, pad_val=None):
    """
    References:
        tf.image.resize_image_with_crop_or_pad
    """
    assert image.ndim in [2, 3]
    assert isinstance(x_min, int) and isinstance(y_min, int)
    assert isinstance(x_max, int) and isinstance(y_max, int)
    assert (x_min <= x_max) and (y_min <= y_max)
    
    src_height, src_width = image.shape[:2]
    dst_height, dst_width = y_max - y_min + 1, x_max - x_min + 1
    channels = 1 if image.ndim == 2 else image.shape[2]
    
    if pad_val is not None:
        if isinstance(pad_val, (int, float)):
            pad_val = [pad_val for _ in range(channels)]
        assert len(pad_val) == channels
        
    src_x_begin = max(x_min, 0)
    src_y_begin = max(y_min, 0, )
    src_x_end = min(x_max + 1, src_width)
    src_y_end = min(y_max + 1, src_height)
    dst_x_begin = src_x_begin - x_min
    dst_y_begin = src_y_begin - y_min
    dst_x_end = src_x_end - x_min
    dst_y_end = src_y_end - y_min
    
    if image.ndim == 2: 
        dst_image_shape = (dst_height, dst_width)
    else:
        dst_image_shape = (dst_height, dst_width, channels)
    if pad_val is None:
        dst_image = np.zeros(dst_image_shape, image.dtype)
    else:
        dst_image = np.full(dst_image_shape, pad_val, dtype=image.dtype)
    dst_image[dst_y_begin: dst_y_end, dst_x_begin: dst_x_end, ...] = \
        image[src_y_begin: src_y_end, src_x_begin: src_x_end, ...]
    return dst_image
    
    
def crop_or_pad_coords(boxes, image_width, image_height):
    """
    References:
        `mmcv.impad`
        `pad` in https://github.com/kpzhang93/MTCNN_face_detection_alignment
        `MtcnnDetector.pad` in https://github.com/AITTSMD/MTCNN-Tensorflow
    """
    x_mins = boxes[:, 0]
    y_mins = boxes[:, 1]
    x_maxs = boxes[:, 2]
    y_maxs = boxes[:, 3]

    src_x_begin = np.maximum(x_mins, 0)
    src_y_begin = np.maximum(y_mins, 0)
    src_x_end = np.minimum(x_maxs + 1, image_width)
    src_y_end = np.minimum(y_maxs + 1, image_height)
    
    dst_x_begin = src_x_begin - x_mins
    dst_y_begin = src_y_begin - y_mins
    dst_x_end = src_x_end - x_mins
    dst_y_end = src_y_end - y_mins

    coords = np.stack([src_x_begin, src_y_begin, src_x_end, src_y_end,
                       dst_x_begin, dst_y_begin, dst_x_end, dst_y_end], axis=1)
    return coords
    
    