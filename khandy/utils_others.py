
import re
import time
import json
import socket
import logging
import argparse
import numbers
import datetime


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
    utcnow = datetime.datetime.now(tz)
    return utcnow

