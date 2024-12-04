import argparse
import builtins
import json
import logging
import numbers
import os
import socket
import subprocess
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import requests


def is_number_between(value: numbers.Real, 
                      lower: Optional[numbers.Real], upper: Optional[numbers.Real], 
                      lower_close: bool = True, upper_close: bool = False) -> bool:
    """Compares a value to upper and lower bounds, considering the bounds are closed or open.
    
    Args:
        value: A value of type numbers.Real, which represents the number to be checked.
        lower: An optional value of type numbers.Real, which represents the lower bound of the range. 
            It can be None if no lower bound is provided.
        upper: An optional value of type numbers.Real, which represents the upper bound of the range. 
            It can be None if no upper bound is provided.
        lower_close: A boolean value that indicates whether the lower bound should be considered closed or open. 
            If True, the lower bound is considered closed, meaning the value is included in the range. 
            If False, it is considered open, meaning the value is not included. Defaults to True.
        upper_close: A boolean value that indicates whether the upper bound should be considered closed or open. 
            If True, the upper bound is considered closed, meaning the value is included in the range. 
            If False, it is considered open, meaning the value is not included. Defaults to False.

    Notes:
        When `lower` is None, it can be understood as negative infinity. 
        When `upper` is None, it can be understood as positive infinity. 
        When the interval formed by `lower` and `upper` is not valid, the interval can be understood as an empty set.
    """
    assert isinstance(value, numbers.Real)
    assert lower is None or isinstance(lower, numbers.Real)
    assert upper is None or isinstance(upper, numbers.Real)
    assert isinstance(lower_close, bool)
    assert isinstance(upper_close, bool)
    
    if upper is not None:
        if upper_close:
            upper_result = value <= upper
        else:
            upper_result = value < upper
    else:
        upper_result = True
    
    if lower is not None:
        if lower_close:
            lower_result = lower <= value
        else:
            lower_result = lower < value
    else:
        lower_result = True
    return upper_result and lower_result


def all_of(iterable, pred):
    """Returns whether all elements in the iterable satisfy the predicate.

    Args:
        iterable (Iterable): An iterable to check.
        pred (callable): A predicate to apply to each element.

    Returns:
        bool: True if all elements satisfy the predicate, False otherwise.

    References:
        https://en.cppreference.com/w/cpp/algorithm/all_any_none_of
    """
    return all(pred(element) for element in iterable)


def any_of(iterable, pred):
    """Returns whether any element in the iterable satisfies the predicate.

    Args:
        iterable (Iterable): An iterable to check.
        pred (callable): A predicate to apply to each element.

    Returns:
        bool: True if any element satisfies the predicate, False otherwise.

    References:
        https://en.cppreference.com/w/cpp/algorithm/all_any_none_of
    """
    return any(pred(element) for element in iterable)


def none_of(iterable, pred):
    """Returns whether no elements in the iterable satisfy the predicate.

    Args:
        iterable (Iterable): An iterable to check.
        pred (callable): A predicate to apply to each element.

    Returns:
        bool: True if no elements satisfy the predicate, False otherwise.

    References:
        https://en.cppreference.com/w/cpp/algorithm/all_any_none_of
    """
    return not any(pred(element) for element in iterable)


def print_with_no(obj):
    if hasattr(obj, '__len__'):
        for k, item in enumerate(obj):
            print('[{}/{}] {}'.format(k+1, len(obj), item)) 
    elif hasattr(obj, '__iter__'):
        for k, item in enumerate(obj):
            print('[{}] {}'.format(k+1, item)) 
    else:
        print('[1] {}'.format(obj))
        
        
@dataclass
class TypeShapeSpec:
    name: str
    type: Optional[builtins.type] = None
    shape: Optional[Union[Tuple[int], int]] = None
    dtype: Optional[str] = None
    value: Optional[Union[int, bool, float]] = None
    
    def __repr__(self):
        type = self.type
        if self.dtype is not None:
            type = f'{type}[{self.dtype}]'
        repr_str = f"{self.name}: {type}, {self.shape}"
        if self.value is not None:
            repr_str = f'{repr_str}, {self.value}'
        return repr_str
    
    
def get_type_and_shape(x, name='x', max_seq_len=10) -> List[TypeShapeSpec]:
    """Determines the type and shape (sometimes dtype and value) of the given object x and returns a list of TypeShapeSpec objects.
  
    Args:  
        x (Any): The object to analyze.
        name (str, optional): The name of the object. Defaults to 'x'.
        max_seq_len (int, optional): The maximum sequence length to analyze for sequences. Defaults to 10.
  
    Returns:  
        List[TypeShapeSpec]: A list of TypeShapeSpec objects representing the type, shape, dtype, and value of x.
    """
    if (hasattr(x, 'shape') and not callable(x.shape) and
        hasattr(x, 'dtype') and not callable(x.dtype)):
        # e.g. torch.Tensor and numpy.ndarray
        results = [TypeShapeSpec(name, type(x), x.shape, str(x.dtype))]
    elif (hasattr(x, 'shape') and not callable(x.shape)):
        results = [TypeShapeSpec(name, type(x), x.shape)]
    elif isinstance(x, (tuple, list)):
        results = [TypeShapeSpec(name, type(x), len(x))]
        if max_seq_len is not None:
            seq_len = min(max_seq_len, len(x))
        for i in range(seq_len):
            results += get_type_and_shape(x[i], f"  {name}[{i}]", max_seq_len)
        if len(x) > seq_len:
            results += [TypeShapeSpec(f"  {name}[...]")]
    elif isinstance(x, dict):
        results = [TypeShapeSpec(name, type(x))]
        for k, v in x.items():
            results += get_type_and_shape(v, f"  {name}.{k}", max_seq_len)
    elif isinstance(x, str):
        results = [TypeShapeSpec(name, type(x), len(x))]
    elif isinstance(x, (int, bool, float)):
        results = [TypeShapeSpec(name, type(x), value=x)]
    else:
        results = [TypeShapeSpec(name, type(x))]
    return results


def get_file_line_count(filename, encoding='utf-8'):
    line_count = 0
    buffer_size = 1024 * 1024 * 8
    with open(filename, 'r', encoding=encoding) as f:
        while True:
            data = f.read(buffer_size)
            if not data:
                break
            line_count += data.count('\n')
    return line_count


def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip
    

def set_logger(filename, level=logging.INFO, logger_name=None, formatter=None, with_print=True):
    logger = logging.getLogger(logger_name) 
    logger.setLevel(level)
    
    if formatter is None:
        formatter = logging.Formatter('%(message)s')

    # Never mutate (insert/remove elements) the list you're currently iterating on. 
    # If you need, make a copy.
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
        # FileHandler is subclass of StreamHandler, so isinstance(handler,
        # logging.StreamHandler) is True even if handler is FileHandler.
        # if (type(handler) == logging.StreamHandler) and (handler.stream == sys.stderr):
        elif type(handler) == logging.StreamHandler:
            logger.removeHandler(handler)
            
    file_handler = logging.FileHandler(filename, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if with_print:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def print_arguments(args, sort_keys=True, indent=4):
    assert isinstance(args, argparse.Namespace)
    args = vars(args)
    print(json.dumps(args, indent=indent, sort_keys=sort_keys))


def save_arguments(filename, args, sort_keys=True, indent=4):
    assert isinstance(args, argparse.Namespace)
    args = vars(args)
    with open(filename, 'w') as f:
        json.dump(args, f, indent=indent, sort_keys=sort_keys)


class DownloadStatusCode(Enum):
    FILE_SIZE_TOO_LARGE = (-100, 'the size of file from url is too large')
    FILE_SIZE_TOO_SMALL = (-101, 'the size of file from url is too small')
    FILE_SIZE_IS_ZERO = (-102, 'the size of file from url is zero')
    URL_IS_NOT_IMAGE = (-103, 'URL is not an image')
    
    @property
    def code(self):
        return self.value[0]

    @property
    def message(self):
        return self.value[1]


class DownloadError(Exception):
    def __init__(self, status_code: DownloadStatusCode, extra_str: Optional[str] = None):
        self.status_code = status_code
        self.extra_str = extra_str
        
        self.name = status_code.name
        self.code = status_code.code
        if extra_str is None:
            self.message = status_code.message
        else:
            self.message = f'{status_code.message}: {extra_str}'
        Exception.__init__(self)

    def __repr__(self):
        return f'[{self.__class__.__name__} {self.code}] {self.message}'
    
    __str__ = __repr__

    
def download_image(image_url, min_filesize=0, max_filesize=100*1024*1024, 
                   params=None, **kwargs) -> bytes:
    """
    References:
        https://httpwg.org/specs/rfc9110.html#field.content-length
        https://requests.readthedocs.io/en/latest/user/advanced/#body-content-workflow
    """
    stream = kwargs.pop('stream', True)
    
    with requests.get(image_url, stream=stream, params=params, **kwargs) as response:
        response.raise_for_status()

        content_type = response.headers.get('content-type')
        if content_type is None:
            warnings.warn('No Content-Type!')
        else:
            if not content_type.startswith(('image/', 'application/octet-stream')):
                raise DownloadError(DownloadStatusCode.URL_IS_NOT_IMAGE)
        
        # when Transfer-Encoding == chunked, Content-Length does not exist.
        content_length = response.headers.get('content-length')
        if content_length is None:
            warnings.warn('No Content-Length!')
        else:
            content_length = int(content_length)
            if content_length > max_filesize:
                raise DownloadError(DownloadStatusCode.FILE_SIZE_TOO_LARGE)
            if content_length < min_filesize:
                raise DownloadError(DownloadStatusCode.FILE_SIZE_TOO_SMALL)
        
        filesize = 0
        chunks = []
        for chunk in response.iter_content(chunk_size=10*1024):
            chunks.append(chunk)
            filesize += len(chunk)
            if filesize > max_filesize:
                raise DownloadError(DownloadStatusCode.FILE_SIZE_TOO_LARGE)
        if filesize < min_filesize:
            raise DownloadError(DownloadStatusCode.FILE_SIZE_TOO_SMALL)
        image_bytes = b''.join(chunks)

    return image_bytes
    

def download_file(url, min_filesize=0, max_filesize=100*1024*1024, 
                  params=None, **kwargs) -> bytes:
    """
    References:
        https://httpwg.org/specs/rfc9110.html#field.content-length
        https://requests.readthedocs.io/en/latest/user/advanced/#body-content-workflow
    """
    stream = kwargs.pop('stream', True)
    
    with requests.get(url, stream=stream, params=params, **kwargs) as response:
        response.raise_for_status()

        # when Transfer-Encoding == chunked, Content-Length does not exist.
        content_length = response.headers.get('content-length')
        if content_length is None:
            warnings.warn('No Content-Length!')
        else:
            content_length = int(content_length)
            if content_length > max_filesize:
                raise DownloadError(DownloadStatusCode.FILE_SIZE_TOO_LARGE)
            if content_length < min_filesize:
                raise DownloadError(DownloadStatusCode.FILE_SIZE_TOO_SMALL)
        
        filesize = 0
        chunks = []
        for chunk in response.iter_content(chunk_size=10*1024):
            chunks.append(chunk)
            filesize += len(chunk)
            if filesize > max_filesize:
                raise DownloadError(DownloadStatusCode.FILE_SIZE_TOO_LARGE)
        if filesize < min_filesize:
            raise DownloadError(DownloadStatusCode.FILE_SIZE_TOO_SMALL)
        file_bytes = b''.join(chunks)

    return file_bytes
    

def get_git_repo_root(path: Optional[str] = None) -> Optional[str]:
    """Get the root directory of a Git repository.

    Args:
        path (Optional[str]): The path to the directory where the Git repository is located.
            If not provided, the current working directory will be used.

    Returns:
        Optional[str]: The absolute path of the top-level directory of the Git repository.
            If the Git repository is not found or an error occurs, None will be returned.
    """
    if path is None:
        path = os.getcwd()
    try:
        # Show the absolute path of the top-level directory of the working tree.
        output = os.popen(f'git -C "{path}" rev-parse --show-toplevel').read().strip()
        if output:
            return output
        else:
            return None
    except Exception as e:
        print(f"Error occurred while getting Git repo root: {e}")
        return None
    

def get_gpu_count() -> int:
    """Return the count of GPUs available on the system.

    This function uses the 'nvidia-smi' command to query the system for NVIDIA GPUs.
    It counts the occurrences of 'UUID' in the output to estimate the number of GPUs.

    Returns:
        int: The number of GPUs found, or 0 if an error occurs while executing the 'nvidia-smi' command.
    """
    try:
        output = subprocess.check_output(['nvidia-smi', '--list-gpus'], text=True)
        return output.count('UUID')
    except subprocess.CalledProcessError:
        return 0
    
    
def get_sw_slices(length: int, window_size: int, stride: Optional[int] = None,
                  drop_last: bool = False, adjust_last: bool = True) -> List[Tuple[int, int]]:
    """Generate a list of tuples representing sliding window slices of a sequence of given length.

    Args:
        length (int): The total length of the sequence to be sliced.
        window_size (int): The size of each sliding window.
        stride (Optional[int], optional): The step size (or stride) between the start of each window.
            If None, it defaults to the window_size. Defaults to None.
        drop_last (bool, optional): Whether to drop the last window if it is incomplete.
            Defaults to False.
        adjust_last (bool, optional): Whether to adjust the start index of the last window
            if it is incomplete and drop_last is False. This ensures the last window's size
            equals the window_size if possible. Defaults to True.

    Returns:
        List[Tuple[int, int]]: A list of tuples, where each tuple (start, end) represents
            a sliding window slice of the sequence. The end index is exclusive.

    Raises:
        TypeError: If `length`, `window_size`, or `stride` are not of the expected type.
        ValueError: If `length`, `stride`, or `window_size` are zero or negative.
    """
    if not isinstance(length, int):
        raise TypeError(f'`length` must be int, got {type(length)}')
    if not isinstance(window_size, int):
        raise TypeError(f'`window_size` must be int, got {type(window_size)}')
    if not isinstance(stride, int) and (stride is not None):
        raise TypeError(f'`stride` must be int or None, got {type(stride)}')
    if length <= 0:
        raise ValueError(f'`length` cannot be zero or negative, got {length}')
    if stride is not None and stride <= 0:
        raise ValueError(f'`stride` cannot be zero or negative, got {stride}')
    if window_size <= 0:
        raise ValueError(f'`window_size` cannot be zero or negative, got {window_size}')

    stride = stride or window_size

    slices = []
    max_index = min_index = 0
    while max_index < length:
        max_index = min_index + window_size
        if max_index > length:
            if not drop_last:
                max_index = length
                if adjust_last:
                    min_index = max(0, max_index - window_size)
                slices.append((min_index, max_index))
        else:
            slices.append((min_index, max_index))
        min_index = min_index + stride
    return slices
