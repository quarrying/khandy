import numpy as np


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
    """
    assert isinstance(x_shift, int)
    assert isinstance(y_shift, int)
    image_height, image_width = image.shape[:2]

    if image.ndim == 2:
        channels = 1
    elif image.ndim == 3:
        channels = image.shape[-1]
        
    if isinstance(border_value, int):
        new_image = np.full_like(image, border_value)
    elif isinstance(border_value, tuple):
        assert len(border_value) == channels, \
            'Expected the num of elements in tuple equals the channels' \
            'of input image. Found {} vs {}'.format(
                len(border_value), channels)
        if channels == 1:
            new_image = np.full_like(image, border_value[0])
        else:
            border_value = np.asarray(border_value, dtype=image.dtype)
            new_image = np.empty_like(image)
            new_image[:] = border_value
    else:
        raise ValueError(
            'Invalid type {} for `border_value`.'.format(type(border_value)))
        
    if (abs(x_shift) >= image_width) or (abs(y_shift) >= image_height):
        return new_image
        
    src_x_start = max(0, -x_shift)
    src_x_end   = min(image_width, image_width - x_shift)
    dst_x_start = max(0, x_shift)
    dst_x_end   = min(image_width, image_width + x_shift)
    
    src_y_start = max(0, -y_shift)
    src_y_end   = min(image_height, image_height - y_shift)
    dst_y_start = max(0, y_shift)
    dst_y_end   = min(image_height, image_height + y_shift)
    
    new_image[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        image[src_y_start:src_y_end, src_x_start:src_x_end]
    return new_image
    
    