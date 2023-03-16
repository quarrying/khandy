import numbers

import khandy


def translate_image(image, x_shift, y_shift, border_value=0):
    """Translate an image.
    
    Args:
        image (ndarray): Image to be translated with format (h, w) or (h, w, c).
        x_shift (int): The offset used for translate in horizontal
            direction. right is the positive direction.
        y_shift (int): The offset used for translate in vertical
            direction. down is the positive direction.
        border_value (int | tuple[int]): Value used in case of a 
            constant border.
            
    Returns:
        ndarray: The translated image.

    See Also:
        crop_or_pad
    """
    assert khandy.is_numpy_image(image)
    assert isinstance(x_shift, numbers.Integral)
    assert isinstance(y_shift, numbers.Integral)
    image_height, image_width = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    
    if isinstance(border_value, (tuple, list)):
        assert len(border_value) == channels, \
            'Expected the num of elements in tuple equals the channels ' \
            'of input image. Found {} vs {}'.format(
                len(border_value), channels)
    else:
        border_value = (border_value,) * channels
    dst_image = khandy.create_solid_color_image(
        image_height, image_width, border_value, dtype=image.dtype)
    
    if (abs(x_shift) >= image_width) or (abs(y_shift) >= image_height):
        return dst_image
        
    src_x_begin = max(-x_shift, 0)
    src_x_end   = min(image_width - x_shift, image_width)
    dst_x_begin = max(x_shift, 0)
    dst_x_end   = min(image_width + x_shift, image_width)
    
    src_y_begin = max(-y_shift, 0)
    src_y_end   = min(image_height - y_shift, image_height)
    dst_y_begin = max(y_shift, 0)
    dst_y_end   = min(image_height + y_shift, image_height)
    
    dst_image[dst_y_begin:dst_y_end, dst_x_begin:dst_x_end] = \
        image[src_y_begin:src_y_end, src_x_begin:src_x_end]
    return dst_image
    
    