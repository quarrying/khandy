import collections.abc
import itertools
import numbers
import random
import warnings

import numpy as np


def to_list(obj):
    if obj is None:
        return None
    elif hasattr(obj, '__iter__') and not isinstance(obj, str):
        try:
            return list(obj)
        except:
            return [obj]
    else:
        return [obj]


def convert_lists_to_record(*list_objs, delimiter=None):
    warnings.warn('convert_lists_to_record will be deprecated, use convert_table_to_records or/with transpose_table instead!')

    assert len(list_objs) >= 1, 'list_objs length must >= 1.'
    delimiter = delimiter or ','

    assert isinstance(list_objs[0], (tuple, list))
    number = len(list_objs[0])
    for item in list_objs[1:]:
        assert isinstance(item, (tuple, list))
        assert len(item) == number, '{} != {}'.format(len(item), number)
        
    records = []
    record_list = zip(*list_objs)
    for record in record_list:
        record_str = [str(item) for item in record]
        records.append(delimiter.join(record_str))
    return records


def convert_table_to_records(*table, delimiter=None):
    """Convert a table of tuples or lists into a list of records separated by a delimiter.  
  
    Args:  
        *table: A variable number of tuples or lists representing rows in the table.  
        delimiter (str, optional): The delimiter to use for separating values in each record. Defaults to ','.  
  
    Returns:  
        list: A list of strings, where each string is a record representing a row in the table.  
  
    Raises:  
        AssertionError: If the length of 'table' is less than 1.  
        AssertionError: If the first item in 'table' is not a tuple or list.  
        AssertionError: If any subsequent item in 'table' is not a tuple or list of the same length as the first item.  
    """  
    assert len(table) >= 1, 'table length must >= 1.'
    delimiter = delimiter or ','

    assert isinstance(table[0], (tuple, list))
    number = len(table[0])
    for item in table[1:]:
        assert isinstance(item, (tuple, list))
        assert len(item) == number, '{} != {}'.format(len(item), number)
        
    records = [delimiter.join(str(item) for item in row) for row in table]
    return records


def shuffle_table(table):
    """
    Notes:
        table can be seen as sequence of sequence which have equal items.
    """
    shuffled = list(zip(*table))
    random.shuffle(shuffled)
    dst_table = tuple(zip(*shuffled))
    return dst_table
    
    
def transpose_table(table):
    """
    Notes:
        table can be seen as sequence of sequence which have equal items.
    """
    return tuple(zip(*table))


def concat_list(in_list):
    """Concatenate a list of list into a single list.

    Args:
        in_list (list): The list of list to be merged.

    Returns:
        list: The concatenated flat list.
    
    References:
        mmcv.concat_list
    """
    return list(itertools.chain(*in_list))
    

def split_by_num(x, num_splits, strict=True):
    """
    Args:
        num_splits: an integer indicating the number of splits

    References:
        numpy.split and numpy.array_split
    """
    # NB: np.ndarray is not Sequence
    assert isinstance(x, (collections.abc.Sequence, np.ndarray))
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
    assert isinstance(x, (collections.abc.Sequence, np.ndarray))
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
    assert isinstance(x, (collections.abc.Sequence, np.ndarray))
    assert isinstance(slices, (list, tuple))
    
    out_list = []
    indices = [0] + list(slices) + [len(x)]
    for i in range(len(slices) + 1):
        out_list.append(x[indices[i]: indices[i + 1]])
    return out_list


def split_by_ratio(x, ratios):
    # NB: np.ndarray is not Sequence
    assert isinstance(x, (collections.abc.Sequence, np.ndarray))
    assert isinstance(ratios, (list, tuple))
    
    pdf = [k / sum(ratios) for k in ratios]
    cdf = [sum(pdf[:k]) for k in range(len(pdf) + 1)]
    indices = [int(round(len(x) * k)) for k in cdf]
    return [x[indices[i]: indices[i + 1]] for i in range(len(ratios))]
    
    
def to_ntuple(x, n):
    """Convert the input into a tuple of length n.

    Args:
        x (Any): The input to be converted.
        n (numbers.Integral): The length of the resulting tuple, which must be a positive integer.

    Returns:
        tuple: A tuple of length n.

    Raises:
        ValueError: If n is not a positive integer.
        AssertionError: If len(x) is not equal to n when x is a non-str sequence.
    """
    if not (isinstance(n, numbers.Integral) and n > 0):
        raise ValueError(f'n must be positive integer, got {n}.')
    if isinstance(x, collections.abc.Sequence) and not isinstance(x, str):
        assert len(x) == n, f'len(x) is not equal to {n}.'
        return tuple(x)
    return tuple(x for _ in range(n))


def to_1tuple(x):
    return to_ntuple(x, 1)


def to_2tuple(x):
    return to_ntuple(x, 2)


def to_3tuple(x):
    return to_ntuple(x, 3)


def to_4tuple(x):
    return to_ntuple(x, 4)


def is_seq_of(seq, item_type, seq_type=None) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        item_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.

    Returns:
        bool: Whether the sequence is valid.
    
    References:
        mmcv
    """
    if seq_type is None:
        exp_seq_type = collections.abc.Sequence
    else:
        assert issubclass(seq_type, collections.abc.Sequence)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, item_type):
            return False
    return True


def is_list_of(seq, item_type):
    """Check whether it is a list of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, item_type, seq_type=list)


def is_tuple_of(seq, item_type):
    """Check whether it is a tuple of some type.

    A partial method of :func:`is_seq_of`.
    """
    return is_seq_of(seq, item_type, seq_type=tuple)

