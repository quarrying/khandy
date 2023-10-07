
import collections.abc
import itertools
import numbers
import random


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


def shuffle_table(*table):
    """
    Notes:
        table can be seen as list of list which have equal items.
    """
    shuffled_list = list(zip(*table))
    random.shuffle(shuffled_list)
    tuple_list = zip(*shuffled_list)
    return [list(item) for item in tuple_list]
    
    
def transpose_table(table):
    """
    Notes:
        table can be seen as list of list which have equal items.
    """
    m, n = len(table), len(table[0])
    return [[table[i][j] for i in range(m)] for j in range(n)]


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

