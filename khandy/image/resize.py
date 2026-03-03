import sys
import warnings
from dataclasses import dataclass
from typing import Tuple, Union

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

import cv2
import numpy as np

import khandy

INTERP_CODES = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}

InterpolationType = Union[Literal['nearest', 'bilinear', 'bicubic', 'area', 'lanczos'], int]


def _normalize_interpolation(
    interpolation: InterpolationType = 'bilinear'
) -> int:
    if isinstance(interpolation, int):
        return interpolation
    
    if isinstance(interpolation, str):
        lowered = interpolation.lower()
        if lowered in INTERP_CODES:
            return INTERP_CODES[lowered]

    raise ValueError(
        f"Unsupported interpolation type: {interpolation}. "
        f"Supported strings are: {list(INTERP_CODES.keys())}"
    )

def resize_image(
    image: np.ndarray, 
    dst_width: int, 
    dst_height: int, 
    interpolation: InterpolationType = 'bilinear',
    *,
    return_scale: bool = False, 
    shrink_use_inter_area: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
    """Resizes an image to the specified dimensions.

    Args:
        image (np.ndarray): Input image as a NumPy array.
        dst_width (int): Target width of the resized image.
        dst_height (int): Target height of the resized image.
        interpolation (InterpolationType, optional): Interpolation method.
            Can be an integer (cv2 interpolation flag) or a string.
            Supported strings: 'area', 'nearest', 'bilinear', 'bicubic', 'lanczos'.
            Defaults to 'bilinear'.
        return_scale (bool, optional): Whether to return `x_scale` and `y_scale`.
            If True, returns a tuple of (`resized_image`, `x_scale`, `y_scale`).
            If False, returns just `resized_image`. Defaults to False.
        shrink_use_inter_area (bool, optional): If True and the image is being
            shrunk (both dimensions decrease), use cv2.INTER_AREA regardless
            of the interpolation setting. Defaults to True.

    Returns:
        tuple or ndarray: (`resized_image`, `x_scale`, `y_scale`) or `resized_image`.
        
    Raises:
        ValueError: If an unsupported string value is passed for interpolation.

    Reference:
        mmcv.imresize
        torch.nn.functional.interpolate
    """
    assert khandy.is_numpy_image(image)
    interpolation = _normalize_interpolation(interpolation)
    
    # In OpenCV (https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html)
    #   To shrink an image, it will generally look best with INTER_AREA interpolation.
    # In Pillow (https://pillow.readthedocs.io/en/stable/releasenotes/2.7.0.html)
    #   From the beginning BILINEAR and BICUBIC filters were based on affine transformations 
    #   and used a fixed number of pixels from the source image for every destination pixel 
    #   (2x2 pixels for BILINEAR and 4x4 for BICUBIC). This gave an unsatisfactory result for 
    #   downscaling. At the same time, a high quality convolutions-based algorithm with flexible 
    #   kernel was used for ANTIALIAS filter. Starting from Pillow 2.7.0, a high quality 
    #   convolutions-based algorithm is used for all of these three filters.
    src_height, src_width = image.shape[:2]
    if shrink_use_inter_area and (dst_width < src_width and dst_height < src_height):
        interpolation = cv2.INTER_AREA

    resized_image = cv2.resize(image, (dst_width, dst_height), interpolation=interpolation)

    if not return_scale:
        return resized_image
    else:
        x_scale = dst_width / src_width
        y_scale = dst_height / src_height
        return resized_image, x_scale, y_scale


def scale_image(
    image: np.ndarray,
    x_scale: Union[float, int],
    y_scale: Union[float, int],
    interpolation: InterpolationType = "bilinear",
    *,
    return_scale: bool = False,
    shrink_use_inter_area: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
    """Scale image.
    
    Reference:
        mmcv.imrescale
    """
    assert khandy.is_numpy_image(image)
    image_size = khandy.get_image_size(image)
    image_size = image_size.scale(x_scale, y_scale)
    return resize_image(
        image,
        image_size.width,
        image_size.height,
        interpolation,
        return_scale=return_scale,
        shrink_use_inter_area=shrink_use_inter_area
    )


def resize_image_short(
    image: np.ndarray,
    dst_size: int,
    interpolation: InterpolationType = "bilinear",
    *,
    return_scale: bool = False,
    shrink_use_inter_area: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
    """Resize an image so that the length of shorter side is dst_size while
    preserving the original aspect ratio.
    
    References:
        `resize_min` in `https://github.com/pjreddie/darknet/blob/master/src/image.c`
    """
    assert khandy.is_numpy_image(image)
    image_size = khandy.get_image_size(image)
    image_size = image_size.resize_short_to(dst_size)
    return resize_image(
        image,
        image_size.width,
        image_size.height,
        interpolation,
        return_scale=return_scale,
        shrink_use_inter_area=shrink_use_inter_area
    )


def resize_image_long(
    image: np.ndarray,
    dst_size: int,
    interpolation: InterpolationType = "bilinear",
    *,
    return_scale: bool = False,
    shrink_use_inter_area: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
    """Resize an image so that the length of longer side is dst_size while 
    preserving the original aspect ratio.
    
    References:
        `resize_max` in `https://github.com/pjreddie/darknet/blob/master/src/image.c`
    """
    assert khandy.is_numpy_image(image)
    image_size = khandy.get_image_size(image)
    image_size = image_size.resize_long_to(dst_size)
    return resize_image(
        image,
        image_size.width,
        image_size.height,
        interpolation,
        return_scale=return_scale,
        shrink_use_inter_area=shrink_use_inter_area
    )


def resize_image_to_range(
    image: np.ndarray,
    min_length: int,
    max_length: int,
    interpolation: InterpolationType = 'bilinear',
    *,
    return_scale: bool = False,
    shrink_use_inter_area: bool = True
) -> Union[np.ndarray, Tuple[np.ndarray, float, float]]:
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
    image_size = khandy.get_image_size(image)
    image_size = image_size.resize_to_range(min_length, max_length)
    return resize_image(
        image,
        image_size.width,
        image_size.height,
        interpolation,
        return_scale=return_scale,
        shrink_use_inter_area=shrink_use_inter_area
    )


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


def letterbox_image(
    image: np.ndarray,
    dst_width: int,
    dst_height: int,
    border_value: int = 0,
    interpolation: InterpolationType = 'bilinear',
    *,
    loc: LetterBoxLoc = 'center',
    shrink_use_inter_area: bool = True
) -> Tuple[np.ndarray, LetterBoxDetail]:
    """Resize an image while preserving its aspect ratio and pad it to match the desired dimensions.
  
    Args:  
        image (numpy.ndarray): The input image to be resized and padded.  
        dst_width (int): The desired width of the output image.  
        dst_height (int): The desired height of the output image.  
        border_value: The color value to use for padding. Defaults to 0 (black).  
        interpolation (str, optional): The interpolation method to use for resizing. Defaults to 'bilinear'.  
        shrink_use_inter_area (bool, optional): If True and the image is being
            shrunk (both dimensions decrease), use cv2.INTER_AREA regardless
            of the interpolation setting. Defaults to True.
            
    Returns:  
        tuple: A tuple containing:  
            padded_image (numpy.ndarray): The resized and padded image.  
            lb_detail (LetterBoxDetail): An object containing details of the resizing and padding process.  

    References:
        letterbox_image` in `https://github.com/pjreddie/darknet/blob/master/src/image.c`
    """  
    assert khandy.is_numpy_image(image)

    src_image_size = khandy.get_image_size(image)
    resized_image_size = src_image_size.fit_contain(khandy.ImageSize(dst_width, dst_height))
    resized_image = resize_image(
        image,
        resized_image_size.width,
        resized_image_size.height,
        interpolation,
        return_scale=False,
        shrink_use_inter_area=shrink_use_inter_area
    )

    height_diff = dst_height - resized_image_size.height
    width_diff = dst_width - resized_image_size.width
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

    padded_image = cv2.copyMakeBorder(
        resized_image,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=border_value,
    )

    lb_detail = LetterBoxDetail(
        resized_w=resized_image_size.width,
        resized_h=resized_image_size.height,
        x_scale=resized_image_size.width / src_image_size.width,
        y_scale=resized_image_size.height / src_image_size.height,
        pad_top=pad_top,
        pad_left=pad_left,
        pad_bottom=pad_bottom,
        pad_right=pad_right
    )
    return padded_image, lb_detail


def letterbox_resize_image(image, dst_width, dst_height, border_value=0,
                           return_scale=False, interpolation: InterpolationType = 'bilinear'):
    warnings.warn('letterbox_resize_image will be deprecated, use letterbox_image instead!')
    dst_image, lb_detail = letterbox_image(image, dst_width, dst_height, border_value, interpolation)
    if not return_scale:
        return dst_image
    else:
        return dst_image, lb_detail.x_scale, lb_detail.pad_top, lb_detail.pad_left
