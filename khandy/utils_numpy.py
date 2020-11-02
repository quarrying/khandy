import numpy as np


def sigmoid(x):
    return 1. / (1 + np.exp(-x))
    
    
def softmax(x, axis=-1, copy=True):
    """
    Args:
        copy: Copy x or not.
        
    Referneces:
        `from sklearn.utils.extmath import softmax`
    """
    if copy:
        x = np.copy(x)
    max_val = np.max(x, axis=axis, keepdims=True)
    x -= max_val
    np.exp(x, x)
    sum_exp = np.sum(x, axis=axis, keepdims=True)
    x /= sum_exp
    return x
    
    
def log_sum_exp(x, axis=-1, keepdims=False):
    """
    References:
        numpy.logaddexp
        numpy.logaddexp2
        scipy.misc.logsumexp
    """
    max_val = np.max(x, axis=axis, keepdims=True)
    x -= max_val
    np.exp(x, x)
    sum_exp = np.sum(x, axis=axis, keepdims=keepdims)
    lse = np.log(sum_exp, sum_exp)
    if not keepdims:
        max_val = np.squeeze(max_val, axis=axis)
    return max_val + lse
    
    
def l2_normalize(x, axis=0, epsilon=1e-12, copy=True):
    """L2 normalize an array along an axis.
    
    Args:
        x : array_like of floats
            Input data.
        axis : None or int or tuple of ints, optional
            Axis or axes along which to operate.
        epsilon: float, optional
            A small value such as to avoid division by zero.
        copy : bool, optional
            Copy X or not.
    """
    if copy:
        x = np.copy(x)
    x /= np.maximum(np.linalg.norm(x, axis=axis, keepdims=True), epsilon)
    return x
    
    
def minmax_normalize(x, axis=0, copy=True):
    """minmax normalize an array along a given axis.
    
    Args:
        x : array_like of floats
            Input data.
        axis : None or int or tuple of ints, optional
            Axis or axes along which to operate.
        copy : bool, optional
            Copy X or not.
    """
    if copy:
        x = np.copy(x)
    
    minval = np.min(x, axis=axis, keepdims=True)
    maxval = np.max(x, axis=axis, keepdims=True)
    maxval -= minval
    maxval = np.maximum(maxval, 1e-5)
    
    x -= minval
    x /= maxval
    return x

    
def get_order_of_magnitude(number):
    number = np.where(number == 0, 1, number)
    oom = np.floor(np.log10(np.abs(number)))
    return oom.astype(np.int32)
    
    