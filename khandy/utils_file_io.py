import json
import base64
import numbers
import warnings
from collections import OrderedDict


def load_list(filename, encoding='utf-8', start=0, stop=None):
    assert isinstance(start, numbers.Integral) and start >= 0
    assert (stop is None) or (isinstance(stop, numbers.Integral) and stop > start)
    
    lines = []
    with open(filename, 'r', encoding=encoding) as f:
        for _ in range(start):
            f.readline()
        for k, line in enumerate(f):
            if (stop is not None) and (k + start > stop):
                break
            lines.append(line.rstrip('\n'))
    return lines


def save_list(filename, list_obj, encoding='utf-8', append_break=True):
    with open(filename, 'w', encoding=encoding) as f:
        if append_break:
            for item in list_obj:
                f.write(str(item) + '\n')
        else:
            for item in list_obj:
                f.write(str(item))


def load_json(filename, encoding='utf-8'):
    with open(filename, 'r', encoding=encoding) as f:
        data = json.load(f, object_pairs_hook=OrderedDict)
    return data


def save_json(filename, data, encoding='utf-8', indent=4, cls=None, sort_keys=False):
    if not filename.endswith('.json'):
        filename = filename + '.json'
    with open(filename, 'w', encoding=encoding) as f:
        json.dump(data, f, indent=indent, separators=(',',': '),
                  ensure_ascii=False, cls=cls, sort_keys=sort_keys)


def load_bytes(filename, use_base64: bool = False) -> bytes:
    """Open the file in bytes mode, read it, and close the file.
    
    References:
        pathlib.Path.read_bytes
    """
    with open(filename, 'rb') as f:
        data = f.read()
    if use_base64:
        data = base64.b64encode(data)
    return data


def save_bytes(filename, data: bytes, use_base64: bool = False) -> int:
    """Open the file in bytes mode, write to it, and close the file.
    
    References:
        pathlib.Path.write_bytes
    """
    if use_base64:
        data = base64.b64decode(data)
    with open(filename, 'wb') as f:
        ret = f.write(data)
    return ret


def load_as_base64(filename) -> bytes:
    warnings.warn('khandy.load_as_base64 will be deprecated, use khandy.load_bytes instead!')
    return load_bytes(filename, True)

