import os
import imghdr
import numbers
import warnings
from io import BytesIO

import cv2
import khandy
import numpy as np
from PIL import Image


def imread(file_or_buffer, flags=-1):
    """Improvement on cv2.imread, make it support filename including chinese character.
    """
    try:
        if isinstance(file_or_buffer, bytes):
            return cv2.imdecode(np.frombuffer(file_or_buffer, dtype=np.uint8), flags)
        else:
            # support type: file or str or Path
            return cv2.imdecode(np.fromfile(file_or_buffer, dtype=np.uint8), flags)
    except Exception as e:
        print(e)
        return None
    

def imread_cv(file_or_buffer, flags=-1):
    warnings.warn('khandy.imread_cv will be deprecated, use khandy.imread instead!')
    return imread(file_or_buffer, flags)


def imwrite(filename, image, params=None):
    """Improvement on cv2.imwrite, make it support filename including chinese character.
    """
    cv2.imencode(os.path.splitext(filename)[-1], image, params)[1].tofile(filename)


def imwrite_cv(filename, image, params=None):
    warnings.warn('khandy.imwrite_cv will be deprecated, use khandy.imwrite instead!')
    return imwrite(filename, image, params)


def imread_pil(file_or_buffer, to_mode=None):
    """Improvement on Image.open to avoid ResourceWarning.
    """
    try:
        if isinstance(file_or_buffer, bytes):
            buffer = BytesIO()
            buffer.write(file_or_buffer)
            buffer.seek(0)
            file_or_buffer = buffer

        if hasattr(file_or_buffer, 'read'):
            image = Image.open(file_or_buffer)
            if to_mode is not None:
                image = image.convert(to_mode)
        else:
            # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
            with open(file_or_buffer, 'rb') as f:
                image = Image.open(f)
                # If convert outside with statement, will raise "seek of closed file" as
                # https://github.com/microsoft/Swin-Transformer/issues/66
                if to_mode is not None:
                    image = image.convert(to_mode)
        return image
    except Exception as e:
        print(e)
        return None
        
        
def imwrite_bytes(filename, image_bytes: bytes, update_extension: bool = True):
    """Write image bytes to file.
    
    Args:
        filename: str
            filename which image_bytes is written into.
        image_bytes: bytes
            image content to be written.
        update_extension: bool
            whether update extension according to image_bytes or not.
            the cost of update extension is smaller than update image format.
    """
    extension = imghdr.what('', image_bytes)
    file_extension = khandy.get_path_extension(filename)
    # imghdr.what fails to determine image format sometimes!
    # so when its return value is None, never update extension.
    if extension is None:
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
        image_bytes = cv2.imencode(file_extension, image)[1]
    elif (extension.lower() != file_extension.lower()[1:]):
        if update_extension:
            filename = khandy.replace_path_extension(filename, extension)
        else:
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
            image_bytes = cv2.imencode(file_extension, image)[1]
    
    with open(filename, "wb") as f:
        f.write(image_bytes)
    return filename


def normalize_image_dtype(image, keep_num_channels=False):
    """Normalize image dtype to uint8 (usually for visualization).
    
    Args:
        image : ndarray
            Input image.
        keep_num_channels : bool, optional
            If this is set to True, the result is an array which has 
            the same shape as input image, otherwise the result is 
            an array whose channels number is 3.
            
    Returns:
        out: ndarray
            Image whose dtype is np.uint8.
    """
    assert (image.ndim == 3 and image.shape[-1] in [1, 3]) or (image.ndim == 2)

    image = image.astype(np.float32)
    image = khandy.minmax_normalize(image, axis=None, copy=False)
    image = np.array(image * 255, dtype=np.uint8)
    
    if not keep_num_channels:
        if image.ndim == 2:
            image = np.expand_dims(image, -1)
        if image.shape[-1] == 1:
            image = np.tile(image, (1,1,3))
    return image
    
    
def normalize_image_shape(image, swap_rb=False):
    """Normalize image shape to (h, w, 3).
    
    Args:
        image : ndarray
            Input image.
        swap_rb : bool, optional
            whether swap red and blue channel or not
            
    Returns:
        out: ndarray
            Image whose shape is (h, w, 3).
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3:
        num_channels = image.shape[-1]
        if num_channels == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif num_channels == 3:
            if swap_rb:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif num_channels == 4:
            if swap_rb:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError('Unsupported!')
    else:
        raise ValueError('Unsupported!')
    return image


def stack_image_list(image_list, dtype=np.float32):
    """Join a sequence of image along a new axis before first axis.

    References:
        `im_list_to_blob` in `py-faster-rcnn-master/lib/utils/blob.py`
    """
    assert isinstance(image_list, (tuple, list))

    max_dimension = np.array([image.ndim for image in image_list]).max()
    assert max_dimension in [2, 3]
    max_shape = np.array([image.shape[:2] for image in image_list]).max(axis=0)
    
    num_channels = []
    for image in image_list:
        if image.ndim == 2:
            num_channels.append(1)
        else:
            num_channels.append(image.shape[-1])
    assert len(set(num_channels) - set([1])) in [0, 1]
    max_num_channels = np.max(num_channels)
    
    blob = np.empty((len(image_list), max_shape[0], max_shape[1], max_num_channels), dtype=dtype)
    for k, image in enumerate(image_list):
        blob[k, :image.shape[0], :image.shape[1], :] = np.atleast_3d(image).astype(dtype, copy=False)
    if max_dimension == 2:
        blob = np.squeeze(blob, axis=-1)
    return blob
    

def is_numpy_image(image):
    return isinstance(image, np.ndarray) and image.ndim in {2, 3}


def is_gray_image(image, tol=3):
    assert is_numpy_image(image)
    if image.ndim == 2:
        return True
    elif image.ndim == 3:
        num_channels = image.shape[-1]
        if num_channels == 1:
            return True
        elif num_channels == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            mae = np.mean(cv2.absdiff(rgb, gray3))
            return mae <= tol
        elif num_channels == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            mae = np.mean(cv2.absdiff(image, gray3))
            return mae <= tol
        else:
            return False
    else:
        return False
        

def is_solid_color_image(image, tol=4):
    assert is_numpy_image(image)
    mean = np.array(cv2.mean(image)[:-1], dtype=np.float32)
    mae = np.mean(np.abs(image - mean))
    return mae <= tol


def create_solid_color_image(image_width, image_height, color, dtype=None):
    if isinstance(color, numbers.Real):
        image = np.full((image_height, image_width), color, dtype=dtype)
    elif isinstance(color, (tuple, list)):
        if len(color) == 1:
            image = np.full((image_height, image_width), color[0], dtype=dtype)
        elif len(color) in (3, 4):
            image = np.full((1, 1, len(color)), color, dtype=dtype)
            image = cv2.copyMakeBorder(image, 0, image_height-1, 0, image_width-1, 
                                       cv2.BORDER_CONSTANT, value=color)
        else:
            color = np.asarray(color, dtype=dtype)
            image = np.empty((image_height, image_width, len(color)), dtype=dtype)
            image[:] = color
    else:
        raise ValueError(f'Invalid type {type(color)} for `color`.')
    return image
