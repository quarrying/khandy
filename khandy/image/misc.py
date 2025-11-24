
import codecs
import os
import numbers
import warnings
from dataclasses import dataclass, field
from io import BytesIO

import cv2
import khandy
import numpy as np
from PIL import Image


@dataclass
class ImageSize:
    width: int
    height: int

    @property
    def cols(self) -> int:
        return self.width
    
    @property
    def rows(self) -> int:
        return self.height
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height
    
    @property
    def ar(self) -> float:
        return self.width / self.height
    
    
def get_image_size(obj) -> ImageSize:
    if isinstance(obj, (str, os.PathLike)):
        with open(obj, 'rb') as f:
            image = Image.open(f)
            return ImageSize(width=image.width, height=image.height)
    elif isinstance(obj, Image.Image):
        return ImageSize(width=obj.width, height=obj.height)
    elif isinstance(obj, np.ndarray) and obj.ndim >= 2:
        return ImageSize(width=obj.shape[1], height=obj.shape[0])
    else:
        raise TypeError(f"Unexpected type {type(obj)}")
    

def _is_isobmff(buf):
    if len(buf) < 16 or buf[4:8] != b'ftyp':
        return False
    if len(buf) < int(codecs.encode(buf[0:4], 'hex'), 16):
        return False
    return True


def _get_ftyp(buf):
    ftyp_len = int(codecs.encode(buf[0:4], 'hex'), 16)
    major_brand = buf[8:12].decode(errors='ignore')
    minor_version = int(codecs.encode(buf[12:16], 'hex'), 16)
    compatible_brands = []
    for i in range(16, ftyp_len, 4):
        compatible_brands.append(buf[i:i+4].decode(errors='ignore'))
    return major_brand, minor_version, compatible_brands


def match_heic(buf):
    if _is_isobmff(buf):
        major_brand, minor_version, compatible_brands = _get_ftyp(buf)
        if major_brand == 'heic':
            return True
        if major_brand in ['mif1', 'msf1'] and 'heic' in compatible_brands:
            return True


def get_image_file_type(file_or_buffer):
    """Determine the image file type based on the file header.

    Args:
        file_or_buffer (Union[BinaryIO, bytes]): file_or_buffer: A file path or buffer object.

    Returns:
        Union[str, None]: A lowercase string representing the image file type ('bmp', 'gif', 'jpeg', 'png', 'tiff', 'webp', 'heic')
        or None if the file type is not recognized.

    References:
        https://bugs.python.org/issue28591
        https://github.com/h2non/filetype.py/blob/master/filetype/types/image.py
        https://github.com/kovidgoyal/calibre/blob/master/src/calibre/utils/imghdr.py
        https://peps.python.org/pep-0594/#imghdr
        https://github.com/sphinx-doc/sphinx/blob/a502e7523376e0344c1c9cc8a9d128143cd98b2d/sphinx/util/images.py
    """
    header = khandy.get_file_header(file_or_buffer, 32)

    # https://en.wikipedia.org/wiki/BMP_file_format#Bitmap_file_header
    if header.startswith(b'BM'):
        return 'bmp'

    # https://en.wikipedia.org/wiki/GIF#File_format
    if header.startswith((b'GIF87a', b'GIF89a')):
        return 'gif'

    # https://en.wikipedia.org/wiki/JPEG_File_Interchange_Format#File_format_structure
    if header.startswith(b'\xFF\xD8\xFF'):
        return 'jpeg'

    # https://en.wikipedia.org/wiki/PNG#File_header
    if header.startswith(b'\x89PNG\r\n\x1A\n'):
        return 'png'

    # https://en.wikipedia.org/wiki/TIFF#Byte_order
    if header.startswith((b'MM', b'II')):
        return 'tiff'

    # https://en.wikipedia.org/wiki/WebP#Technology
    if header.startswith(b'RIFF') and header[8:12] == b'WEBP':
        return 'webp'

    # https://en.wikipedia.org/wiki/High_Efficiency_Image_File_Format
    header = khandy.get_file_header(file_or_buffer, 8192)
    if match_heic(header):
        return 'heic'

    return None


def get_image_extension(file_or_buffer):
    warnings.warn('khandy.get_image_extension will be deprecated, use khandy.get_image_file_type instead!')
    return get_image_file_type(file_or_buffer)


def imread(file_or_buffer, flags=cv2.IMREAD_COLOR):
    """Improvement on cv2.imread, make it support filename including chinese character.
    
    Args:
        flags: same as OpenCV
    """
    try:
        if isinstance(file_or_buffer, (bytes, bytearray, memoryview)):
            return cv2.imdecode(np.frombuffer(file_or_buffer, dtype=np.uint8), flags)
        else:
            # support type: file or str or Path
            return cv2.imdecode(np.fromfile(file_or_buffer, dtype=np.uint8), flags)
    except Exception as e:
        warnings.warn(f'{e}')
        return None
    

def imread_cv(file_or_buffer, flags=cv2.IMREAD_COLOR):
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
        warnings.warn(f'{e}')
        return None
        
        
def imwrite_bytes(
    filename: str,
    image_bytes: bytes,
    raise_if_not_image: bool = True,
    use_adpative_ext: bool = True
) -> str:
    """Write image bytes to file.
    
    Args:
        filename: str
            filename which image_bytes is written into.
        image_bytes: bytes
            image content to be written.
        raise_if_not_image: bool
            whether to raise if image_bytes is not image. Defaults to True.
        use_adpative_ext: bool
            whether to adaptively handle the file extension. Defaults to True.
    """
    extension = get_image_file_type(image_bytes)
    if extension is None and raise_if_not_image:
        raise Exception('image_bytes is not image!')
    name_extension = khandy.get_path_extension(filename)
    if name_extension == '':
        filename = f'{filename}.{extension}'
        name_extension = f'.{extension}'
        
    if extension is not None:
        if name_extension.lower() == '.jpg' and extension.lower() == 'jpeg':
            extension == 'jpg'
        if (extension.lower() != name_extension.lower()[1:]):
            if use_adpative_ext:
                filename = khandy.replace_path_extension(filename, extension)
            else:
                image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
                image_bytes = cv2.imencode(name_extension, image)[1]
    
    with open(filename, "wb") as f:
        f.write(image_bytes)
    return filename


def rescale_image(image: np.ndarray, rescale_factor='auto', dst_dtype=np.float32):
    """Rescale image by rescale_factor.

    Args:
        img (ndarray): Image to be rescaled.
        rescale_factor (str, int or float, *optional*, defaults to `'auto'`): 
            rescale the image by the specified scale factor. When is `'auto'`, 
            rescale the image to [0, 1).
        dtype (np.dtype, *optional*, defaults to `np.float32`):
            The dtype of the output image. Defaults to `np.float32`.

    Returns:
        ndarray: The rescaled image.
    """
    if rescale_factor == 'auto':
        if np.issubdtype(image.dtype, np.unsignedinteger):
            rescale_factor = 1. / np.iinfo(image.dtype).max
        else:
            raise TypeError(f'Only support uint dtype ndarray when `rescale_factor` is `auto`, got {image.dtype}')
    elif issubclass(rescale_factor, (int, float)):
        pass
    else:
        raise TypeError('rescale_factor must be "auto", int or float')
    image = image.astype(dst_dtype, copy=True)
    image *= rescale_factor
    image = image.astype(dst_dtype)
    return image


def normalize_image_value(image: np.ndarray, mean, std, rescale_factor=None):
    """Normalize image values using provided mean and standard deviation.  
  
    Args:
        image (np.ndarray): The input image to be normalized.
        mean (int, float, Sequence[int], Sequence[float], np.ndarray): The mean to be used for normalize.
        std (int, float, Sequence[int], Sequence[float], np.ndarray): The std to be used for normalize.
        rescale_factor (Union[int, float, str, None], optional): Rescale factor for mean and std.
            If set to 'auto', it will scale the mean and std based on the maximum value of the image dtype (only for unsigned integers).
            If int or float, the mean and std will be multiplied by the given factor.
            If None, no rescaling will be done. Defaults to None.
  
    Returns:
        np.ndarray: The normalized image with dtype np.float32.
  
    Raises:
        TypeError: If the `rescale_factor` is not one of the supported types (int, float, str, None).
        TypeError: If the image dtype is not unsigned integer when `rescale_factor` is set to 'auto'.
    """  
    dst_dtype = np.float32
    mean = np.array(mean, dtype=dst_dtype).flatten()
    std = np.array(std, dtype=dst_dtype).flatten()
    if rescale_factor == 'auto':
        if np.issubdtype(image.dtype, np.unsignedinteger):
            mean *= np.iinfo(image.dtype).max
            std *= np.iinfo(image.dtype).max
        else:
            raise TypeError(f'Only support uint dtype ndarray when `rescale_factor` is `auto`, got {image.dtype}')
    elif isinstance(rescale_factor, (int, float)):
        mean *= rescale_factor
        std *= rescale_factor
    image = image.astype(dst_dtype, copy=True)
    image -= mean
    image /= std
    return image


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
    
    
def normalize_image_channel(image, swap_rb=False):
    """Normalize image channel number and order to RGB or BGR.
    
    Args:
        image : ndarray
            Input image.
        swap_rb : bool, optional
            whether swap red and blue channel or not
            
    Returns:
        out: ndarray
            Image whose shape is (..., 3).
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
            raise ValueError(f'Unsupported image channel number, only support 1, 3 and 4, got {num_channels}!')
    else:
        raise ValueError(f'Unsupported image ndarray ndim, only support 2 and 3, got {image.ndim}!')
    return image


def normalize_image_shape(image, swap_rb=False):
    warnings.warn('khandy.normalize_image_shape will be deprecated, use khandy.normalize_image_channel instead!')
    return normalize_image_channel(image, swap_rb)


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
        elif num_channels == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            mae = np.mean(cv2.absdiff(image, gray3))
            return mae <= tol
        elif num_channels == 4:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            gray3 = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            mae = np.mean(cv2.absdiff(rgb, gray3))
            return mae <= tol
        else:
            return False
    else:
        return False
        

def is_solid_color_image(image, tol=4):
    assert is_numpy_image(image)
    mean = np.array(cv2.mean(image)[:-1], dtype=np.float32)
    
    if image.ndim == 2:
        mae = np.mean(np.abs(image - mean[0]))
        return mae <= tol
    elif image.ndim == 3:
        num_channels = image.shape[-1]
        if num_channels == 1:
            mae = np.mean(np.abs(image - mean[0]))
            return mae <= tol
        elif num_channels == 3:
            mae = np.mean(np.abs(image - mean))
            return mae <= tol
        elif num_channels == 4:
            mae = np.mean(np.abs(image[:,:,:-1] - mean))
            return mae <= tol
        else:
            return False
    else:
        return False


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
        raise TypeError(f'Invalid type {type(color)} for `color`.')
    return image
