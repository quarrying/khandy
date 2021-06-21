import cv2
import numpy as np


interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def scale_image(image, x_scale, y_scale, interpolation='bilinear'):
    """Scale image.
    
    Reference:
        mmcv.imrescale
    """
    src_height, src_width = image.shape[:2]
    
    dst_width = int(round(x_scale * src_width))
    dst_height = int(round(y_scale * src_height))
    resized_image = cv2.resize(image, (dst_width, dst_height), 
                               interpolation=interp_codes[interpolation])
    return resized_image


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
    src_height, src_width = image.shape[:2]
    resized_image = cv2.resize(image, (dst_width, dst_height), 
                               interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_image
    else:
        x_scale = dst_width / src_width
        y_scale = dst_height / src_height
        return resized_image, x_scale, y_scale
    
    
def resize_image_short(image, dst_size, return_scale=False, interpolation='bilinear'):
    """Resize an image so that the length of shorter side is dst_size while 
    preserving the original aspect ratio.
    
    References:
        `resize_min` in `https://github.com/pjreddie/darknet/blob/master/src/image.c`
    """
    src_height, src_width = image.shape[:2]
    scale = max(dst_size / src_width, dst_size / src_height)
    dst_width = int(round(scale * src_width))
    dst_height = int(round(scale * src_height))
    
    resized_image = cv2.resize(image, (dst_width, dst_height), 
                               interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_image
    else:
        return resized_image, scale
    
    
def resize_image_long(image, dst_size, return_scale=False, interpolation='bilinear'):
    """Resize an image so that the length of longer side is dst_size while 
    preserving the original aspect ratio.
    
    References:
        `resize_max` in `https://github.com/pjreddie/darknet/blob/master/src/image.c`
    """
    src_height, src_width = image.shape[:2]
    scale = min(dst_size / src_width, dst_size / src_height)
    dst_width = int(round(scale * src_width))
    dst_height = int(round(scale * src_height))
    
    resized_image = cv2.resize(image, (dst_width, dst_height), 
                               interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_image
    else:
        return resized_image, scale
        
        
def letterbox_resize_image(image, new_width, new_height, pad_val=0,
                           return_scale=False, interpolation='bilinear'):
    """Resize an image preserving the original aspect ratio using padding.
    
    References:
        `letterbox_image` in `https://github.com/pjreddie/darknet/blob/master/src/image.c`
    """
    ori_height, ori_width = image.shape[:2]

    scale = min(new_width / ori_width, new_height / ori_height)
    resize_w = int(round(scale * ori_width))
    resize_h = int(round(scale * ori_height))

    resized_image = cv2.resize(image, (resize_w, resize_h), 
                               interpolation=interp_codes[interpolation])
    padded_shape = list(resized_image.shape)
    padded_shape[0] = new_height
    padded_shape[1] = new_width
    padded_image = np.full(padded_shape, pad_val, image.dtype)

    dw = int(round((new_width - resize_w) / 2.0))
    dh = int(round((new_height - resize_h) / 2.0))
    padded_image[dh: resize_h + dh, dw: resize_w + dw, ...] = resized_image
    
    if not return_scale:
        return padded_image
    else:
        return padded_image, scale, dw, dh
        
        
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
        resized_image: resized image (with bilinear interpolation) so that
            min(new_height, new_width) == min_length or
            max(new_height, new_width) == max_length.
          
    References:
        `resize_to_range` in `models-master/research/object_detection/core/preprocessor.py`
        `prep_im_for_blob` in `py-faster-rcnn-master/lib/utils/blob.py`
        mmcv.imrescale
    """
    assert min_length < max_length
    ori_height, ori_width = image.shape[:2]
    
    min_side_length = np.minimum(ori_width, ori_height)
    max_side_length = np.maximum(ori_width, ori_height)
    scale = min_length / min_side_length
    if round(scale * max_side_length) > max_length:
        scale = max_length / max_side_length
        
    new_width = int(round(scale * ori_width))
    new_height = int(round(scale * ori_height))
    resized_image = cv2.resize(image, (new_width, new_height), 
                               interpolation=interp_codes[interpolation])
    if not return_scale:
        return resized_image
    else:
        return resized_image, scale
        
        