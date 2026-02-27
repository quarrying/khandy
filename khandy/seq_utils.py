import collections.abc
import itertools
import numbers
import random
import warnings
from typing import (
    Any,
    List,
    Optional,
    Union,
    Tuple,
    Iterable,
    TypeVar,
    Callable,
    Literal,
)

import numpy as np


def to_list(obj) -> Optional[List]:
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


def concat_list(input_data: List[Iterable[Any]]) -> List[Any]:
    """Concatenate a list of iterables into a single flat list.

    Args:
        input_data (List[Iterable[Any]]): A list containing iterable objects to be concatenated.

    Returns:
        List[Any]: A single flat list containing all elements from the input iterables.

    Raises:
        TypeError: If input_data is not a list or its elements are not iterable.
        
    References:
        mmcv.concat_list
    """
    return list(itertools.chain.from_iterable(input_data))


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


T = TypeVar('T')
def split_by(
    iterable: Iterable[T], 
    condition: Callable[[int, T], bool], 
    mode: Literal['first', 'last', 'drop'] = 'first'
) -> List[List[T]]:
    """Split an iterable into sublists based on a condition.
    
    References:
        more_itertools.split_at
        more_itertools.split_before
        more_itertools.split_after
    """
    if mode not in ['first', 'last', 'drop']:
        raise ValueError("mode must be one of 'first', 'last', or 'drop'")
    
    result: List[List[T]] = []
    current_sublist: List[T] = []
    
    for index, item in enumerate(iterable):
        if condition(index, item):
            if mode == 'first':
                # Add current sublist if not empty, then start new sublist with the matching item
                if current_sublist:
                    result.append(current_sublist)
                current_sublist = [item]
            elif mode == 'last':
                # Add the matching item to current sublist and save it
                current_sublist.append(item)
                result.append(current_sublist)
                current_sublist = []
            elif mode == 'drop':
                # Add current sublist, drop the matching item
                result.append(current_sublist)
                current_sublist = []
        else:
            # Regular item, add to current sublist
            current_sublist.append(item)
    
    if mode == 'drop':
        result.append(current_sublist)
    elif current_sublist:
        result.append(current_sublist)
    return result


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


def is_seq_of(
    seq: Any, 
    item_type: Union[type, Tuple[type, ...]], 
    seq_type: Union[type, Tuple[type, ...], None] = None
) -> bool:
    """Checks if the provided sequence is of a specified item type and optionally of a specified sequence type.

    Args:
        seq (Any): The sequence to check.
        item_type (Union[type, Tuple[type, ...]]): The expected type(s) of items in the sequence.
        seq_type (Union[type, Tuple[type, ...], None]): The expected type(s) of the sequence itself. 
            If None, defaults to collections.abc.Sequence.

    Returns:
        bool: True if the sequence is of the specified item type and sequence type, False otherwise.
    
    Raises:
        AssertionError: If seq_type is provided and is not a subclass of collections.abc.Sequence.
        
    References:
        mmcv
    """
    if seq_type is None:
        exp_seq_type = collections.abc.Sequence
    else:
        if isinstance(seq_type, tuple):
            for item in seq_type:
                assert issubclass(item, collections.abc.Sequence), \
                    f'{item} is not a subclass of collections.abc.Sequence'
        else:
            assert issubclass(seq_type, collections.abc.Sequence), \
                f'{seq_type} is not a subclass of collections.abc.Sequence'
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


class EqLenSequences:
    """Class for managing sequences of equal length.

    References:
        https://github.com/facebookresearch/detectron2/blob/main/detectron2/structures/instances.py
        https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/data_structures/instance_data.py
    """
    def __init__(self, **kwargs) -> None:
        """Initialize the class object with keyword arguments.

        Args:
            **kwargs: Keyword arguments where the key is the attribute name and the value is the attribute value.
        """
        self._fields = set()
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override the default attribute assignment.
        Only allows assignment of sequences (objects with __len__ and __getitem__ attributes).

        Args:
            name (str): Name of the attribute to set.
            value (Any): Value of the attribute, which should be a sequence.

        Raises:
            AttributeError: If trying to change the private attribute '_fields'.
            AssertionError: If the value is not a sequence or its length is inconsistent with existing sequences.
        """
        if name == '_fields':
            # hasattr is implemented by calling getattr(object, name) and
            # seeing whether it raises an AttributeError or not.
            if not hasattr(self, name):
                super().__setattr__(name, value)
            else:
                raise AttributeError(f'{name} has been used as a private attribute, which is immutable.')
        else:
            # colllections.abc.Sequence is not used here to avoid issues with
            # np.ndarray and torch.Tensor which is not a Sequence
            assert hasattr(value, '__len__'), 'value must contain `__len__` attribute'
            assert hasattr(value, '__getitem__'), 'value must contain `__getitem__` attribute'
            if len(self._fields) > 0:
                msg = f'{name} len is not consistent with len(self), {len(value)} vs {len(self)}'
                assert len(value) == len(self), msg
            self._fields.add(name)
            super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        """Override the default attribute deletion.

        Args:
            name (str): Name of the attribute to delete.

        Raises:
            AttributeError: If trying to delete the private attribute '_fields'.
        """
        if name == '_fields':
            raise AttributeError(f'{name} has been used as a private attribute, which is immutable.')
        super().__delattr__(name)
        self._fields.remove(name)

    def __len__(self) -> int:
        """Returns the length of the sequences stored in the object.

        Returns:
            int: The length of the sequences, or 0 if there are no fields.

        Note:
            All sequences must have the same length.
        """
        for name in self._fields:
            return len(getattr(self, name))
        return 0

    def __contains__(self, name: str) -> bool:
        """Checks if the given name is a field (sequence) in the object.

        Args:
            name (str): Name of the field to check.

        Returns:
            bool: True if the field exists, False otherwise.
        """
        return name in self._fields

    def __getitem__(self, key: Union[numbers.Integral, slice, List[numbers.Integral], np.ndarray]):
        """Supports indexing and slicing to get a new object
        with sequences indexed or sliced by the given key.

        Args:
            key (Union[numbers.Integral, slice, List[numbers.Integral]], np.ndarray): Index, slice, 
                or list of indices to use for getting sequences.

        Returns:
            A new object of the same class with sequences indexed or sliced according to the key.

        Raises:
            IndexError: If the index, slice, or any index in the list is out of range.
            TypeError: If the key is not an integer, slice, list of integers, or a 1D numpy array.
            AssertionError: If the key is a numpy array but not 1D, or if the length of the sequence 
                does not match the length of the key when using a boolean mask.
        """
        if isinstance(key, numbers.Integral):
            if key >= len(self) or key < -len(self):
                raise IndexError(f"{self.__class__.__name__} index {key} out of range!")
            else:
                key = slice(key, None, len(self))
        elif isinstance(key, np.ndarray):
            assert key.ndim == 1, f"Key must be a 1D array, but got {key.ndim}D array."
        elif not isinstance(key, (slice, list)):
            raise TypeError(f"Unsupported key type: {type(key)}")
            
        kwargs = {}
        for name in self._fields:
            value = getattr(self, name)
            try:
                kwargs[name] = value[key]
            except:
                if isinstance(key, np.ndarray) and key.dtype == np.bool_:
                    assert len(value) == len(key), \
                        f"Length of {name} ({len(value)}) does not match length of key ({len(key)})"
                    # If key is a boolean array, use it to index the sequence
                    kwargs[name] = [value[i] for i, elem in enumerate(key) if elem]
                else:
                    kwargs[name] = [value[i] for i in key]
        return self.__class__(**kwargs)

    def __str__(self) -> str:
        """Return a string representation of the EqLenSequences object.

        Returns:
            str: A string representation of the object.
        """
        s = self.__class__.__name__ + "("
        s += "num_instances={}, ".format(len(self))
        s += "fields={{{}}})".format(", ".join((f"{name}: {getattr(self, name)}" for name in self._fields)))
        return s

    __repr__ = __str__

    def get_fields(self) -> set:
        """Return a set of attribute names (fields) stored in the object.

        Returns:
            set: A set of attribute names.
        """
        return self._fields

    def filter_(self, index: Union[numbers.Integral, slice, List[numbers.Integral], np.ndarray]):
        """Filter the EqLenSequences object by indexing the stored sequences using the provided index.

        This method allows you to filter the sequences stored in the EqLenSequences object by
        specifying an integer, slice, or list of integers as the index. Depending on the type of
        index provided, the corresponding elements or slices will be retained in the object.

        Args:
            index (Union[numbers.Integral, slice, List[numbers.Integral], np.ndarray]): The index or slice to use for filtering.
                - If an integer is provided, it should be within the valid range of the object's length.
                - If a slice is provided, it will be used to slice the sequences.
                - If a list of integers is provided, the sequences at the specified indices will be retained.
                - If a numpy array is provided, it should be a 1D array of integers or booleans:
                    - If a boolean numpy array is provided, it will be used to filter the sequences based on the mask.
                    - If a list of integers is provided, it can be used to index the sequences.

        Returns:
            EqLenSequences: The filtered EqLenSequences object with the selected sequences.

        Raises:
            IndexError: If the provided index is out of range (i.e., an integer index is greater than
                or equal to the length of the object, or less than negative the length of the object).
            TypeError: If the provided index is not an integer, slice, list of integers, or a 1D numpy array.
            AssertionError: If the provided index is a numpy array but not 1D, or if the length of the sequence
                does not match the length of the key when using a boolean mask.
        """
        if isinstance(index, numbers.Integral):
            if index >= len(self) or index < -len(self):
                raise IndexError(f"{self.__class__.__name__} index {index} out of range!")
            else:
                index = slice(index, None, len(self))
        elif isinstance(index, np.ndarray):
            assert index.ndim == 1, f"Index must be a 1D array, but got {index.ndim}D array."
        elif not isinstance(index, (slice, list)):
            raise TypeError(f"Unsupported index type: {type(index)}")

        for name in self._fields:
            value = getattr(self, name)
            try:
                super().__setattr__(name, value[index])
            except:
                # NB: used to avoid the following errors:
                # TypeError: only integer scalar arrays can be converted to a scalar index
                # TypeError: list indices must be integers or slices, not list
                if isinstance(index, np.ndarray) and index.dtype == np.bool_:
                    assert len(value) == len(index), \
                        f"Length of {name} ({len(value)}) does not match length of key ({len(index)})"
                    # If key is a boolean array, use it to index the sequence
                    super().__setattr__(name, [value[i] for i, elem in enumerate(index) if elem])
                else:
                    super().__setattr__(name, [value[i] for i in index])
        return self
    
    def filter(self, index: Union[numbers.Integral, slice, List[numbers.Integral], np.ndarray], inplace: bool):
        if inplace:
            return self.filter_(index)
        else:
            return self[index]
