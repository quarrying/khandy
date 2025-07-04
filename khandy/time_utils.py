import time
import logging
import numbers
import datetime
from dataclasses import dataclass
from typing import Callable, Any, Dict, Tuple, Optional

import numpy as np


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
    """get current UTC+8 time or Beijing time
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


@dataclass
class BenchmarkStats:
    avg: float
    std: float
    min: float
    max: float
    sum: float
    cnt: int = 0
    
    @property
    def mean(self) -> float:
        """Alias for mean"""
        return self.avg
    
    @property
    def stddev(self) -> float:
        """Alias for std"""
        return self.std

    @property
    def total(self) -> float:
        """Alias for sum"""
        return self.sum
    
    @property
    def num_repeats(self) -> int:
        """Alias for cnt"""
        return self.cnt
    
    
def benchmark(
    func: Callable[..., Any], 
    args: tuple = (),
    kwargs: Dict[str, Any] = {},
    num_repeats: int = 100,
    num_repeats_burn_in: int = 2,
    display_interval: Optional[int] = None,
    display_desc: str = 'benchmark'
) -> Tuple[Any, BenchmarkStats]:
    """Run func several times to obtain the timing stats.
    
    References:
        tutorials/image/alexnet/alexnet_benchmark.py in tensorflow-models
    """
    assert isinstance(num_repeats, int) and num_repeats > 0
    assert isinstance(num_repeats_burn_in, int) and num_repeats_burn_in >= 0
    assert display_interval is None or (isinstance(display_interval, int) and display_interval > 0)

    duration_list = []
    for i in range(num_repeats + num_repeats_burn_in):
        start_time = time.perf_counter()
        output = func(*args, **kwargs)
        duration = time.perf_counter() - start_time
        if i >= num_repeats_burn_in:
            duration_list.append(duration)
            actual_step = i - num_repeats_burn_in + 1
            if display_interval is not None and actual_step % display_interval == 0:
                print(f'[{display_desc}][{actual_step}/{num_repeats}] {duration:.3f}s')
    
    stats = BenchmarkStats(
        avg=np.mean(duration_list).item(),
        std=np.std(duration_list).item(),
        max=np.max(duration_list).item(),
        min=np.min(duration_list).item(),
        sum=np.sum(duration_list).item(),
        cnt=len(duration_list)
    )
    return output, stats

