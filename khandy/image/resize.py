import warnings
from dataclasses import dataclass
from typing import Literal, Tuple

import cv2
import numpy as np

import khandy

interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def scale_image(image, x_scale, y_scale, return_scale=False, interpolation='bilinear'):
    """Scale image.
    
    Reference:
        mmcv.imrescale
    """
    assert khandy.is_numpy_image(image)
    src_height, src_width = image.shape[:2]
    dst_width = int(round(x_scale * src_width))
    dst_height = int(round(y_scale * src_height))
    
    resized_image = cv2.resize(image, (dst_width, dst_height), 
                               interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_image
    else:
        x_scale = dst_width / src_width
        y_scale = dst_height / src_height
        return resized_image, x_scale, y_scale


def resize_image(image, dst_width, dst_height, return_scale=False, interpolation='bilinear'):
    """Resize image to a given size.

    Args:
        image (ndarray): The input image.
        dst_width (int): Target width.
        dst_height (int): Target height.
        return_scale (bool): Whether to return `x_scale` and `y_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos".

    Returns:
        tuple or ndarray: (`resized_image`, `x_scale`, `y_scale`) or `resized_image`.
        
    Reference:
        mmcv.imresize
    """
    assert khandy.is_numpy_image(image)
    resized_image = cv2.resize(image, (dst_width, dst_height), 
                               interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_image
    else:
        src_height, src_width = image.shape[:2]
        x_scale = dst_width / src_width
        y_scale = dst_height / src_height
        return resized_image, x_scale, y_scale
    
    
def resize_image_short(image, dst_size, return_scale=False, interpolation='bilinear'):
    """Resize an image so that the length of shorter side is dst_size while 
    preserving the original aspect ratio.
    
    References:
        `resize_min` in `https://github.com/pjreddie/darknet/blob/master/src/image.c`
    """
    assert khandy.is_numpy_image(image)
    src_height, src_width = image.shape[:2]
    scale = max(dst_size / src_width, dst_size / src_height)
    dst_width = int(round(scale * src_width))
    dst_height = int(round(scale * src_height))
    
    resized_image = cv2.resize(image, (dst_width, dst_height), 
                               interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_image
    else:
        x_scale = dst_width / src_width
        y_scale = dst_height / src_height
        return resized_image, x_scale, y_scale
    
    
def resize_image_long(image, dst_size, return_scale=False, interpolation='bilinear'):
    """Resize an image so that the length of longer side is dst_size while 
    preserving the original aspect ratio.
    
    References:
        `resize_max` in `https://github.com/pjreddie/darknet/blob/master/src/image.c`
    """
    assert khandy.is_numpy_image(image)
    src_height, src_width = image.shape[:2]
    scale = min(dst_size / src_width, dst_size / src_height)
    dst_width = int(round(scale * src_width))
    dst_height = int(round(scale * src_height))
    
    resized_image = cv2.resize(image, (dst_width, dst_height), 
                               interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_image
    else:
        x_scale = dst_width / src_width
        y_scale = dst_height / src_height
        return resized_image, x_scale, y_scale
        
        
def resize_image_to_range(image, min_length, max_length, return_scale=False, interpolation='bilinear'):
    """Resizes an image so its dimensions are within the provided value.
    
    Rescale the shortest side of the image up to `min_length` pixels 
    while keeping the largest side below `max_length` pixels without 
    changing the aspect ratio. Often used in object detection (e.g. RCNN and SSH.)
    
    The output size can be described by two cases:
    1. If the image can be rescaled so its shortest side is equal to the
        `min_length` without the other side exceeding `max_length`, then do so.
    2. Otherwise, resize so the longest side is equal to `max_length`.
    
    Returns:
        resized_image: resized image so that
            min(dst_height, dst_width) == min_length or
            max(dst_height, dst_width) == max_length.
          
    References:
        `resize_to_range` in `models-master/research/object_detection/core/preprocessor.py`
        `prep_im_for_blob` in `py-faster-rcnn-master/lib/utils/blob.py`
        mmcv.imrescale
    """
    assert khandy.is_numpy_image(image)
    assert min_length < max_length
    src_height, src_width = image.shape[:2]
    
    min_side_length = min(src_width, src_height)
    max_side_length = max(src_width, src_height)
    scale = min_length / min_side_length
    if round(scale * max_side_length) > max_length:
        scale = max_length / max_side_length
    dst_width = int(round(scale * src_width))
    dst_height = int(round(scale * src_height))
    
    resized_image = cv2.resize(image, (dst_width, dst_height), 
                               interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_image
    else:
        x_scale = dst_width / src_width
        y_scale = dst_height / src_height
        return resized_image, x_scale, y_scale
        

@dataclass
class LetterBoxDetail:
    resized_w: int
    resized_h: int
    x_scale: float
    y_scale: float
    pad_top: int
    pad_left: int
    pad_bottom: int
    pad_right: int


LetterBoxLoc = Literal['top left', 'top center', 'top right',
                       'center left', 'center', 'center right', 
                       'bottom left', 'bottom center', 'bottom right']


def letterbox_image(image, dst_width, dst_height, border_value=0, interpolation='bilinear', 
                    loc: LetterBoxLoc = 'center') -> Tuple[np.ndarray, LetterBoxDetail]:
    """Resize an image while preserving its aspect ratio and pad it to match the desired dimensions.
  
    Args:  
        image (numpy.ndarray): The input image to be resized and padded.  
        dst_width (int): The desired width of the output image.  
        dst_height (int): The desired height of the output image.  
        border_value: The color value to use for padding. Defaults to 0 (black).  
        interpolation (str, optional): The interpolation method to use for resizing. Defaults to 'bilinear'.  
  
    Returns:  
        tuple: A tuple containing:  
            padded_image (numpy.ndarray): The resized and padded image.  
            lb_detail (LetterBoxDetail): An object containing details of the resizing and padding process.  

    References:
        letterbox_image` in `https://github.com/pjreddie/darknet/blob/master/src/image.c`
    """  
    assert khandy.is_numpy_image(image)
    src_height, src_width = image.shape[:2]
    scale = min(dst_width / src_width, dst_height / src_height)
    resized_w = int(round(scale * src_width))
    resized_h = int(round(scale * src_height))

    resized_image = cv2.resize(image, (resized_w, resized_h), interpolation=interp_codes[interpolation])
    
    height_diff = dst_height - resized_h
    width_diff = dst_width - resized_w
    pad_top, pad_left = {
        'top left': (0, 0),
        'top center': (0, width_diff // 2),
        'top right': (0, width_diff),
        'center left': (height_diff // 2, 0),
        'center': (height_diff // 2, width_diff // 2),
        'center right': (height_diff // 2, width_diff),
        'bottom left': (height_diff, 0),
        'bottom center': (height_diff, width_diff // 2),
        'bottom right': (height_diff, width_diff),
    }[loc]
    pad_bottom = height_diff - pad_top
    pad_right = width_diff - pad_left
    
    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=border_value)
    
    lb_detail = LetterBoxDetail(
        resized_w=resized_w,
        resized_h=resized_h,
        x_scale=resized_w / src_width,
        y_scale=resized_h / src_height,
        pad_top=pad_top,
        pad_left=pad_left,
        pad_bottom=pad_bottom,
        pad_right=pad_right
    )
    return padded_image, lb_detail
        

def letterbox_resize_image(image, dst_width, dst_height, border_value=0,
                           return_scale=False, interpolation='bilinear'):
    warnings.warn('letterbox_resize_image will be deprecated, use letterbox_image instead!')
    dst_image, lb_detail = letterbox_image(image, dst_width, dst_height, border_value, interpolation)
    if not return_scale:
        return dst_image
    else:
        return dst_image, lb_detail.x_scale, lb_detail.pad_top, lb_detail.pad_left
    