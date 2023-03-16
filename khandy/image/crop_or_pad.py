import numbers
import warnings

import khandy
import numpy as np


def crop(image, x_min, y_min, x_max, y_max, border_value=0):
    """Crop the given image at specified rectangular area.
    
    See Also:
        translate_image
        
    References:
        PIL.Image.crop
        tf.image.resize_image_with_crop_or_pad
    """
    assert khandy.is_numpy_image(image)
    assert isinstance(x_min, numbers.Integral) and isinstance(y_min, numbers.Integral)
    assert isinstance(x_max, numbers.Integral) and isinstance(y_max, numbers.Integral)
    assert (x_min <= x_max) and (y_min <= y_max)
    
    src_height, src_width = image.shape[:2]
    dst_height, dst_width = y_max - y_min + 1, x_max - x_min + 1
    channels = 1 if image.ndim == 2 else image.shape[2]
    
    if isinstance(border_value, (tuple, list)):
        assert len(border_value) == channels, \
            'Expected the num of elements in tuple equals the channels ' \
            'of input image. Found {} vs {}'.format(
                len(border_value), channels)
    else:
        border_value = (border_value,) * channels
    dst_image = khandy.create_solid_color_image(
        dst_width, dst_height, border_value, dtype=image.dtype)

    src_x_begin = max(x_min, 0)
    src_x_end   = min(x_max + 1, src_width)
    dst_x_begin = src_x_begin - x_min
    dst_x_end   = src_x_end - x_min

    src_y_begin = max(y_min, 0)
    src_y_end   = min(y_max + 1, src_height)
    dst_y_begin = src_y_begin - y_min
    dst_y_end   = src_y_end - y_min
    
    if (src_x_begin >= src_x_end) or (src_y_begin >= src_y_end):
        return dst_image
    dst_image[dst_y_begin: dst_y_end, dst_x_begin: dst_x_end, ...] = \
        image[src_y_begin: src_y_end, src_x_begin: src_x_end, ...]
    return dst_image
    

def crop_or_pad(image, x_min, y_min, x_max, y_max, border_value=0):
    warnings.warn('crop_or_pad will be deprecated, use crop instead!')
    return crop(image, x_min, y_min, x_max, y_max, border_value)


def crop_coords(boxes, image_width, image_height):
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
    dst_widths = x_maxs - x_mins + 1
    dst_heights = y_maxs - y_mins + 1
    
    src_x_begin = np.maximum(x_mins, 0)
    src_x_end   = np.minimum(x_maxs + 1, image_width)
    dst_x_begin = src_x_begin - x_mins
    dst_x_end   = src_x_end - x_mins
    
    src_y_begin = np.maximum(y_mins, 0)
    src_y_end   = np.minimum(y_maxs + 1, image_height)
    dst_y_begin = src_y_begin - y_mins
    dst_y_end   = src_y_end - y_mins

    coords = np.stack([dst_y_begin, dst_y_end, dst_x_begin, dst_x_end, 
                       src_y_begin, src_y_end, src_x_begin, src_x_end, 
                       dst_heights, dst_widths], axis=0)
    return coords
    

def crop_or_pad_coords(boxes, image_width, image_height):
    warnings.warn('crop_or_pad_coords will be deprecated, use crop_coords instead!')
    return crop_coords(boxes, image_width, image_height)


def center_crop(image, dst_width, dst_height, strict=True):
    """
    strict: 
        when True, raise error if src size is less than dst size. 
        when False, remain unchanged if src size is less than dst size, otherwise center crop.
    """
    assert khandy.is_numpy_image(image)
    assert isinstance(dst_width, numbers.Integral) and isinstance(dst_height, numbers.Integral)
    src_height, src_width = image.shape[:2]
    if strict:
        assert (src_height >= dst_height) and (src_width >= dst_width)

    crop_top = max((src_height - dst_height) // 2, 0)
    crop_left = max((src_width - dst_width) // 2, 0)
    cropped = image[crop_top: dst_height + crop_top, 
                    crop_left: dst_width + crop_left, ...]
    return cropped


def center_pad(image, dst_width, dst_height, strict=True):
    """
    strict: 
        when True, raise error if src size is greater than dst size. 
        when False, remain unchanged if src size is greater than dst size, otherwise center pad.
    """
    assert khandy.is_numpy_image(image)
    assert isinstance(dst_width, numbers.Integral) and isinstance(dst_height, numbers.Integral)
    
    src_height, src_width = image.shape[:2]
    if strict:
        assert (src_height <= dst_height) and (src_width <= dst_width)
    
    padding_x = max(dst_width - src_width, 0)
    padding_y = max(dst_height - src_height, 0)
    padding_top = padding_y // 2
    padding_left = padding_x // 2
    if image.ndim == 2:
        padding = ((padding_top, padding_y - padding_top), 
                   (padding_left, padding_x - padding_left))
    else:
        padding = ((padding_top, padding_y - padding_top), 
                   (padding_left, padding_x - padding_left), (0, 0))
    return np.pad(image, padding, 'constant')

    