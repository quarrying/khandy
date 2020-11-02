import numpy as np


def translate_image(image, x_shift, y_shift):
    image_height, image_width = image.shape[:2]
    assert abs(x_shift) < image_width
    assert abs(y_shift) < image_height
    
    new_image = np.zeros_like(image)
    if x_shift < 0:
        src_x_start = -x_shift
        src_x_end = image_width
        dst_x_start = 0
        dst_x_end = image_width + x_shift
    else:
        src_x_start = 0
        src_x_end = image_width - x_shift
        dst_x_start = x_shift
        dst_x_end = image_width
        
    if y_shift < 0:
        src_y_start = -y_shift
        src_y_end = image_height
        dst_y_start = 0
        dst_y_end = image_height + y_shift
    else:
        src_y_start = 0
        src_y_end = image_height - y_shift
        dst_y_start = y_shift
        dst_y_end = image_height
        
    new_image[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
        image[src_y_start:src_y_end, src_x_start:src_x_end]
    return new_image
