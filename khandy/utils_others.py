
import re
import time
import json
import socket
import imghdr
import logging
import argparse
import numbers
import datetime
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
        
      
def get_file_line_count(filename):
    line_count = 0
    buffer_size = 1024 * 1024 * 8
    with open(filename, 'r') as f:
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
    
 
class ContextTimer(object):
    """
    References:
        WithTimer in https://github.com/uber/ludwig/blob/master/ludwig/utils/time_utils.py
    """
    def __init__(self, name=None, use_log=False, quiet=False):
        self.use_log = use_log
        self.quiet = quiet
        if name is None:
            self.name = ''
        else:
            self.name = '{}, '.format(name.rstrip())
                
    def __enter__(self):
        self.start_time = time.time()
        if not self.quiet:
            self._print_or_log('{}{} starts'.format(self.name, self._now_time_str))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.quiet:
            self._print_or_log('{}elapsed_time = {:.5}s'.format(self.name, self.get_eplased_time()))
            self._print_or_log('{}{} ends'.format(self.name, self._now_time_str))
            
    @property
    def _now_time_str(self):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    
    def _print_or_log(self, output_str):
        if self.use_log:
            logging.info(output_str)
        else:
            print(output_str)
            
    def get_eplased_time(self):
        return time.time() - self.start_time
        
    def enter(self):
        """Manually trigger enter"""
        self.__enter__()
        

def set_logger(filename, level=logging.INFO, logger_name=None):
    logger = logging.getLogger(logger_name) 
    logger.setLevel(level)
    
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
            
    file_handler = logging.FileHandler(filename)
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
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


def strip_content_in_paren(string):
    """
    Notes:
        strip_content_in_paren cannot process nested paren correctly
    """
    return re.sub(r"\([^)]*\)|（[^）]*）", "", string)


def _to_timestamp(val):
    if val is None:
        timestamp = time.time()
    elif isinstance(val, numbers.Real):
        timestamp = float(val)
    elif isinstance(val, time.struct_time):
        timestamp = time.mktime(val)
    elif isinstance(val, datetime.datetime):
        timestamp = val.timestamp()
    elif isinstance(val, datetime.date):
        dt = datetime.datetime.combine(val, datetime.time())
        timestamp = dt.timestamp()
    elif isinstance(val, str):
        try:
            # The full format looks like 'YYYY-MM-DD HH:MM:SS.mmmmmm'.
            dt = datetime.datetime.fromisoformat(val)
            timestamp = dt.timestamp()
        except:
            raise TypeError('when argument is str, it should conform to isoformat')
    else:
        raise TypeError('unsupported type!')
    return timestamp


def get_timestamp(time_val=None, rounded=True):
    """timestamp in seconds
    """
    timestamp = _to_timestamp(time_val)
    if rounded:
        timestamp = round(timestamp)
    return timestamp


def get_timestamp_ms(time_val=None, rounded=True):
    """timestamp in milliseconds
    """
    timestamp = _to_timestamp(time_val) * 1000
    if rounded:
        timestamp = round(timestamp)
    return timestamp


def get_utc8now():
    tz = datetime.timezone(datetime.timedelta(hours=8))
    utc8now = datetime.datetime.now(tz)
    return utc8now


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

    
def download_image(image_url, min_filesize=None, max_filesize=None, 
                   imghdr_check=False, params=None, **kwargs) -> bytes:
    """
    References:
        https://httpwg.org/specs/rfc9110.html#field.content-length
        https://requests.readthedocs.io/en/latest/user/advanced/#body-content-workflow
    """
    stream = kwargs.pop('stream', True)
    min_filesize = min_filesize or 0
    max_filesize = max_filesize or 100 * 1024 * 1024
    
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
        first_chunk = True
        chunks = []
        for chunk in response.iter_content(chunk_size=10*1024):
            if imghdr_check and first_chunk:
                # imghdr.what fails to determine image format sometimes!
                extension = imghdr.what('', chunk[:64])
                if extension is None:
                    raise DownloadError(DownloadStatusCode.URL_IS_NOT_IMAGE)
                chunks.append(chunk)
                first_chunk = False
            else:
                chunks.append(chunk)
            
            filesize += len(chunk)
            if filesize > max_filesize:
                raise DownloadError(DownloadStatusCode.FILE_SIZE_TOO_LARGE)
        if filesize < min_filesize:
            raise DownloadError(DownloadStatusCode.FILE_SIZE_TOO_SMALL)
        image_bytes = b''.join(chunks)

    return image_bytes
    
