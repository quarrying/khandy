import numpy as np


def assert_and_normalize_shape(x, length):
    """
    Args:
        x: ndarray
        length: int
    """
    if x.ndim == 0:
        return x
    elif x.ndim == 1:
        if len(x) == 1:
            return x
        elif len(x) == length:
            return x
        else:
            raise ValueError('Incompatible shape!')
    elif x.ndim == 2:
        if x.shape == (1, 1):
            return np.squeeze(x, axis=-1)
        elif x.shape == (length, 1):
            return np.squeeze(x, axis=-1)
        else:
            raise ValueError('Incompatible shape!') 
    else:
        raise ValueError('Incompatible ndim!')
        
