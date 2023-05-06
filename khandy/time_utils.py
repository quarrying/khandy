import time
import logging
import numbers
import datetime


def _to_timestamp(val, multiplier=1, rounded=False):
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
    timestamp = timestamp * multiplier
    if rounded:
        # The return value is an integer if ndigits is omitted or None.
        timestamp = round(timestamp)
    return timestamp


def get_timestamp(time_val=None, rounded=True):
    """timestamp in seconds.
    """
    return _to_timestamp(time_val, multiplier=1, rounded=rounded)


def get_timestamp_ms(time_val=None, rounded=True):
    """timestamp in milliseconds.
    """
    return _to_timestamp(time_val, multiplier=1000, rounded=rounded)


def get_timestamp_us(time_val=None, rounded=True):
    """timestamp in microseconds.
    """
    return _to_timestamp(time_val, multiplier=1000000, rounded=rounded)


def get_utc8now() -> datetime.datetime:
    """get current UTC-8 time or Beijing time
    """
    tz = datetime.timezone(datetime.timedelta(hours=8))
    utc8now = datetime.datetime.now(tz)
    return utc8now


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

