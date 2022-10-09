import numbers
from collections import Sequence

import numpy as np


def split_by_num(x, num_splits, strict=True):
    """
    Args:
        num_splits: an integer indicating the number of splits

    References:
        numpy.split and numpy.array_split
    """
    # NB: np.ndarray is not Sequence
    assert isinstance(x, (Sequence, np.ndarray))
    assert isinstance(num_splits, numbers.Integral)
    
    if strict:
        assert len(x) % num_splits == 0
    split_size = (len(x) + num_splits - 1) // num_splits
    out_list = []
    for i in range(0, len(x), split_size):
        out_list.append(x[i: i + split_size])
    return out_list
    
    
def split_by_size(x, sizes):
    """
    References:
        tf.split
        https://github.com/open-mmlab/mmcv/blob/master/mmcv/utils/misc.py
    """
    # NB: np.ndarray is not Sequence
    assert isinstance(x, (Sequence, np.ndarray))
    assert isinstance(sizes, (list, tuple))
     
    assert sum(sizes) == len(x)
    out_list = []
    start_index = 0
    for size in sizes:
        out_list.append(x[start_index: start_index + size])
        start_index += size
    return out_list
    
    
def split_by_slice(x, slices):
    """
    References:
        SliceLayer in Caffe, and numpy.split
    """
    # NB: np.ndarray is not Sequence
    assert isinstance(x, (Sequence, np.ndarray))
    assert isinstance(slices, (list, tuple))
    
    out_list = []
    indices = [0] + list(slices) + [len(x)]
    for i in range(len(slices) + 1):
        out_list.append(x[indices[i]: indices[i + 1]])
    return out_list


def split_by_ratio(x, ratios):
    # NB: np.ndarray is not Sequence
    assert isinstance(x, (Sequence, np.ndarray))
    assert isinstance(ratios, (list, tuple))
    
    pdf = [k / sum(ratios) for k in ratios]
    cdf = [sum(pdf[:k]) for k in range(len(pdf) + 1)]
    indices = [int(round(len(x) * k)) for k in cdf]
    return [x[indices[i]: indices[i + 1]] for i in range(len(ratios))]
    
    