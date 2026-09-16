from __future__ import annotations

import random
import warnings
from typing import Any, Callable, Iterable, Iterator, Sequence, TypeVar

K = TypeVar('K')
V = TypeVar('V')


def get_dict_first_item(
    dict_obj: dict,
    raise_if_empty: bool = False,
) -> tuple[Any, Any] | None:
    """Retrieve the first key-value pair from a dictionary.

    Args:
        dict_obj (dict): The dictionary to retrieve the first item from.
        raise_if_empty (bool): Whether to raise an error if the dictionary is empty.

    Returns:
        tuple[Any, Any] | None: The first key-value pair, or None if the dictionary
            is empty and raise_if_empty is False.

    Raises:
        TypeError: If dict_obj is not a dictionary.
        ValueError: If dict_obj is empty and raise_if_empty is True.
    """
    if not isinstance(dict_obj, dict):
        raise TypeError(f'dict_obj should be a dict, got {type(dict_obj)}')
    if not dict_obj: # empty dict
        if raise_if_empty:
            raise ValueError('dict_obj is empty, cannot retrieve the first item')
        return None
    # next(iter(...)) is more efficient than `for`
    return next(iter(dict_obj.items()))


def sort_dict(
    dict_obj: dict[Any, Any],
    key: Callable[[tuple[Any, Any]], Any] | None = None,
    reverse: bool = False,
) -> dict[Any, Any]:
    return dict(sorted(dict_obj.items(), key=key, reverse=reverse))


def _zip_equal_length(
    keys: Iterable[K],
    values: Iterable[V],
) -> Iterator[tuple[K, V]]:
    """Iterate (key, value) pairs, raising ValueError if lengths differ."""
    keys_list = list(keys)
    values_list = list(values)
    if len(keys_list) != len(values_list):
        raise ValueError(
            f"'keys' and 'values' must have the same length, "
            f"got {len(keys_list)} and {len(values_list)}"
        )
    return zip(keys_list, values_list)


def create_multidict(
    keys: Iterable[K],
    values: Iterable[V],
) -> dict[K, list[V]]:
    multidict_obj: dict[K, list[V]] = {}
    for key, value in _zip_equal_length(keys, values):
        multidict_obj.setdefault(key, []).append(value)
    return multidict_obj


def create_multidict_unique(
    keys: Iterable[K],
    values: Iterable[V],
) -> dict[K, set[V]]:
    multidict_obj: dict[K, set[V]] = {}
    for key, value in _zip_equal_length(keys, values):
        multidict_obj.setdefault(key, set()).add(value)
    return multidict_obj


def convert_multidict_to_list(
    multidict_obj: dict[Any, list[Any]],
) -> tuple[list[Any], list[Any]]:
    key_list, value_list = [], []
    for key, value in multidict_obj.items():
        key_list += [key] * len(value)
        value_list += value
    return key_list, value_list


def convert_multidict_to_records(
    multidict_obj: dict[Any, list[Any]],
    value_first: bool = True,
) -> list[str]:
    records = []
    for key, values in multidict_obj.items():
        for value in values:
            if value_first:
                records.append(f'{value},{key}')
            else:
                records.append(f'{key},{value}')
    return records


def rekey_multidict(
    multidict_obj: dict[Any, list[Any]],
    key_map: dict[Any, Any],
    raise_if_key_error: bool = True,
) -> dict[Any, list[Any]]:
    """Apply key mapping to a multidict object.

    Args:
        multidict_obj: Input multidict object where each key maps to a list of values
        key_map: Dictionary mapping old keys to new keys
        raise_if_key_error: Whether to raise an error if a key is not found in key_map

    Returns:
        If key_map is None, returns the original multidict.
        Otherwise, returns a new multidict with remapped keys.
    """
    if key_map is None:
        return multidict_obj

    result = {}
    for key, values in multidict_obj.items():
        if raise_if_key_error:
            mapped_key = key_map[key]  # This will raise KeyError if key not in key_map
        else:
            mapped_key = key_map.get(key, key)  # Use original key if not in key_map
        result.setdefault(mapped_key, []).extend(values)
    return result


def remap_multidict_keys(
    multidict_obj: dict[Any, list[Any]],
    key_map: dict[Any, Any],
    raise_if_key_error: bool = True,
) -> dict[Any, list[Any]]:
    warnings.warn('`remap_multidict_keys` will be deprecated, use `rekey_multidict` instead!')
    return rekey_multidict(multidict_obj, key_map, raise_if_key_error)


def sample_multidict(
    multidict_obj: dict[Any, list[Any]],
    num_keys: int | None = None,
    num_per_key: int | None = None,
) -> dict[Any, list[Any]]:
    """Randomly samples key-value pairs from a multi-dict object.

    Args:
        multidict_obj (dict[Any, list[Any]]): Input multi-dict object where each key maps to a list of values.
        num_keys (int | None): Number of keys to sample. If None or less than 1, all keys are used.
        num_per_key (int | None): Number of values to sample per key. If None or less than 1, all values
            for each selected key are retained.

    Returns:
        dict[Any, list[Any]]: A new multi-dict containing the sampled key-value pairs.
    """
    if num_keys is None or num_keys < 1:
        num_keys = len(multidict_obj)
    else:
        num_keys = min(num_keys, len(multidict_obj))
    sub_keys = random.sample(list(multidict_obj), num_keys)
    if num_per_key is None:
        sub_mdict = {key: multidict_obj[key] for key in sub_keys}
    else:
        sub_mdict = {}
        for key in sub_keys:
            if num_per_key >= len(multidict_obj[key]) or num_per_key < 1:
                sub_mdict[key] = multidict_obj[key]
            else:
                sub_mdict[key] = random.sample(multidict_obj[key], num_per_key)
    return sub_mdict


def split_multidict_on_key(
    multidict_obj: dict[Any, list[Any]],
    split_ratio: Sequence[float],
    use_shuffle: bool = False,
) -> list[dict[Any, list[Any]]]:
    """Split multidict_obj on its key.
    """
    if not isinstance(multidict_obj, dict):
        raise TypeError(f'multidict_obj should be a dict, got {type(multidict_obj)}')
    if not isinstance(split_ratio, (list, tuple)):
        raise TypeError(f'split_ratio should be a list or tuple, got {type(split_ratio)}')

    pdf = [k / float(sum(split_ratio)) for k in split_ratio]
    cdf = [sum(pdf[:k]) for k in range(len(pdf) + 1)]
    indices = [int(round(len(multidict_obj) * k)) for k in cdf]
    dict_keys = list(multidict_obj)
    if use_shuffle:
        random.shuffle(dict_keys)

    be_split_list = []
    for i in range(len(split_ratio)):
        part_keys = dict_keys[indices[i]: indices[i + 1]]
        part_dict = dict([(key, multidict_obj[key]) for key in part_keys])
        be_split_list.append(part_dict)
    return be_split_list


def split_multidict_on_value(
    multidict_obj: dict[Any, list[Any]],
    split_ratio: Sequence[float],
    use_shuffle: bool = False,
) -> list[dict[Any, list[Any]]]:
    """Split multidict_obj on its value.
    """
    if not isinstance(multidict_obj, dict):
        raise TypeError(f'multidict_obj should be a dict, got {type(multidict_obj)}')
    if not isinstance(split_ratio, (list, tuple)):
        raise TypeError(f'split_ratio should be a list or tuple, got {type(split_ratio)}')

    pdf = [k / float(sum(split_ratio)) for k in split_ratio]
    cdf = [sum(pdf[:k]) for k in range(len(pdf) + 1)]
    be_split_list = [dict() for k in range(len(split_ratio))]
    for key, value in multidict_obj.items():
        indices = [int(round(len(value) * k)) for k in cdf]
        cloned = value[:]
        if use_shuffle:
            random.shuffle(cloned)
        for i in range(len(split_ratio)):
            be_split_list[i][key] = cloned[indices[i]: indices[i + 1]]
    return be_split_list


def get_multidict_info(
    multidict_obj: dict[Any, list[Any]],
    with_print: bool = False,
    desc: str | None = None,
) -> dict[str, float | int]:
    num_list = [len(val) for val in multidict_obj.values()]
    num_keys = len(num_list)
    num_values = sum(num_list)
    max_values_per_key = max(num_list, default=0)
    min_values_per_key = min(num_list, default=0)
    if num_keys == 0:
        avg_values_per_key = 0
    else:
        avg_values_per_key = num_values / num_keys
    info = {
        'num_keys': num_keys,
        'num_values': num_values,
        'max_values_per_key': max_values_per_key,
        'min_values_per_key': min_values_per_key,
        'avg_values_per_key': avg_values_per_key,
    }
    if with_print:
        desc = desc or '<unknown>'
        print('{} key number:    {}'.format(desc, info['num_keys']))
        print('{} value number:    {}'.format(desc, info['num_values']))
        print('{} max number per-key: {}'.format(desc, info['max_values_per_key']))
        print('{} min number per-key: {}'.format(desc, info['min_values_per_key']))
        print('{} avg number per-key: {:.2f}'.format(desc, info['avg_values_per_key']))
    return info


def filter_multidict_by_number(
    multidict_obj: dict[Any, list[Any]],
    lower: int | None = None,
    upper: int | None = None,
) -> dict[Any, list[Any]]:
    if lower is None and upper is None:
        return multidict_obj.copy()
    elif lower is None:
        return {key: value for key, value in multidict_obj.items()
                if len(value) <= upper}
    elif upper is None:
        return {key: value for key, value in multidict_obj.items()
                if lower <= len(value) }
    else:
        if lower > upper:
            raise ValueError('lower must not be greater than upper')
        return {key: value for key, value in multidict_obj.items()
                if lower <= len(value) <= upper }


def sort_multidict_by_number(
    multidict_obj: dict[Any, list[Any]],
    num_keys_to_keep: int | None = None,
    reverse: bool = True,
) -> dict[Any, list[Any]]:
    """
    Args:
        reverse: sort in ascending order when is True.
    """
    if num_keys_to_keep is None:
        num_keys_to_keep = len(multidict_obj)
    else:
        num_keys_to_keep = min(num_keys_to_keep, len(multidict_obj))
    sorted_items = sorted(multidict_obj.items(), key=lambda x: len(x[1]), reverse=reverse)
    filtered_dict = {}
    for i in range(num_keys_to_keep):
        filtered_dict[sorted_items[i][0]] = sorted_items[i][1]
    return filtered_dict


def merge_multidict(
    *mdicts: dict[Any, list[Any]],
) -> dict[Any, list[Any]]:
    merged_multidict: dict[Any, list[Any]] = {}
    for item in mdicts:
        for key, value in item.items():
            merged_multidict.setdefault(key, []).extend(value)
    return merged_multidict


def invert_multidict(
    multidict_obj: dict[Any, list[Any]],
) -> dict[Any, list[Any]]:
    inverted_dict: dict[Any, list[Any]] = {}
    for key, value in multidict_obj.items():
        for item in value:
            inverted_dict.setdefault(item, []).append(key)
    return inverted_dict
