import json
import socket
import logging
import argparse
import warnings
from enum import Enum

import requests


def print_with_no(obj):
    if hasattr(obj, '__len__'):
        for k, item in enumerate(obj):
            print('[{}/{}] {}'.format(k+1, len(obj), item)) 
    elif hasattr(obj, '__iter__'):
        for k, item in enumerate(obj):
            print('[{}] {}'.format(k+1, item)) 
    else:
        print('[1] {}'.format(obj))
        
      
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


def print_arguments(args):
    assert isinstance(args, argparse.Namespace)
    arg_list = sorted(vars(args).items())
    for key, value in arg_list:
        print('{}: {}'.format(key, value))


def save_arguments(filename, args, sort=True):
    assert isinstance(args, argparse.Namespace)
    args = vars(args)
    with open(filename, 'w') as f:
        json.dump(args, f, indent=4, sort_keys=sort)


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
    def __init__(self, status_code: DownloadStatusCode, extra_str: str=None):
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
    

