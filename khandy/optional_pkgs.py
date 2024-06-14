from typing import Union

import numpy as np

try:
    import torch
    T = torch.Tensor
    _torch_available = True
except:
    T = object
    _torch_available = False
    

KArray = Union[np.ndarray, T]


def is_torch_available():
    return _torch_available


def is_torch_tensor(x) -> bool:
    return is_torch_available() and isinstance(x, torch.Tensor)


def import_torch():
    """
    References:
        https://github.com/ray-project/maze-raylit/blob/master/rllib/utils/framework.py
    """
    try:
        import torch
        return torch
    except ImportError as e:
        print(f"PyTorch is not installed. Please install PyTorch to continue. Error: {e}")
        return None
