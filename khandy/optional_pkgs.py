import warnings
from typing import Union

import numpy as np

try:
    import torch
    T = torch.Tensor
except:
    torch = None
    T = np.ndarray
KArray = Union[np.ndarray, T]


def is_torch_available() -> bool:
    return torch is not None


def is_torch_tensor(x) -> bool:
    return (torch is not None) and isinstance(x, torch.Tensor)


def import_torch():
    """
    References:
        https://github.com/ray-project/maze-raylit/blob/master/rllib/utils/framework.py
    """
    try:
        import torch
        return torch
    except ImportError as e:
        warnings.warn(f"PyTorch is not installed: {e}")
        return None
