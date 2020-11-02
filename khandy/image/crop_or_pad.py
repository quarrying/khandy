import cv2
import numpy as np


def crop_or_pad(image, crop_size, crop_center=None, pad_val=None):
    """
    References:
        tf.image.resize_image_with_crop_or_pad
    """
    assert image.ndim in [2, 3]
    
    src_height, src_width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    dst_height, dst_width = crop_size[1], crop_size[0]
    if crop_center is None:
        crop_center = [src_width // 2, src_height // 2]
    if pad_val is not None:
        if isinstance(pad_val, (int, float)):
            pad_val = [pad_val for _ in range(channels)]
        assert len(pad_val) == channels
        
    crop_begin_x = int(round(crop_center[0] - dst_width / 2.0))
    crop_begin_y = int(round(crop_center[1] - dst_height / 2.0))
    
    src_begin_x = max(0, crop_begin_x)
    src_begin_y = max(0, crop_begin_y)
    src_end_x = min(src_width, crop_begin_x + dst_width)
    src_end_y = min(src_height, crop_begin_y + dst_height)
    dst_begin_x = max(0, -crop_begin_x)
    dst_begin_y = max(0, -crop_begin_y)
    dst_end_x = dst_begin_x + src_end_x - src_begin_x
    dst_end_y = dst_begin_y + src_end_y - src_begin_y
    
    if image.ndim == 2: 
        cropped_image_shape = (dst_height, dst_width)
    else:
        cropped_image_shape = (dst_height, dst_width, channels)
    if pad_val is None:
        cropped = np.zeros(cropped_image_shape, image.dtype)
    else:
        cropped = np.full(cropped_image_shape, pad_val, dtype=image.dtype)
    if (src_end_x - src_begin_x <= 0) or (src_end_y - src_begin_y <= 0):
        return cropped
    else:
        cropped[dst_begin_y: dst_end_y, dst_begin_x: dst_end_x, ...] = \
            image[src_begin_y: src_end_y, src_begin_x: src_end_x, ...]
        return cropped
        
        
def crop_or_pad_coords(boxes, image_width, image_height):
    """
    References:
        `mmcv.impad`
        `pad` in https://github.com/kpzhang93/MTCNN_face_detection_alignment
        `MtcnnDetector.pad` in https://github.com/AITTSMD/MTCNN-Tensorflow
    """
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]
    
    src_x_begin = np.maximum(x_min, 0)
    src_y_begin = np.maximum(y_min, 0)
    src_x_end = np.minimum(x_max + 1, image_width)
    src_y_end = np.minimum(y_max + 1, image_height)
    
    dst_widths = x_max - x_min + 1
    dst_heights = y_max - y_min + 1
    dst_x_begin = np.maximum(-x_min, 0)
    dst_y_begin = np.maximum(-y_min, 0)
    dst_x_end = np.minimum(dst_widths, image_width - x_min)
    dst_y_end = np.minimum(dst_heights, image_height - y_min)
    
    coords = np.stack([src_x_begin, src_y_begin, src_x_end, src_y_end,
                       dst_x_begin, dst_y_begin, dst_x_end, dst_y_end], axis=1)
    return coords
    
    