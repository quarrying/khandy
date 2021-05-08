from collections import OrderedDict

import numpy as np

from .utils_dict import get_dict_first_item as _get_dict_first_item


def convert_feature_dict_to_array(feature_dict):
    key_list = []
    one_feature = _get_dict_first_item(feature_dict)[1]
    feature_array = np.empty((len(feature_dict), len(one_feature)), one_feature.dtype)
    for k, (key, value) in enumerate(feature_dict.items()):
        key_list.append(key)
        feature_array[k] = value
    return key_list, feature_array
    
    
def convert_feature_array_to_dict(key_list, feature_array):
    assert len(feature_array) == len(key_list)
    feature_dict = OrderedDict()
    for k, key in enumerate(key_list):
        feature_dict[key] = feature_array[k]
    return feature_dict
    

def pairwise_distances(x, y, squared=True):
    """Compute pairwise (squared) Euclidean distances.
    
    References:
        [2016 CVPR] Deep Metric Learning via Lifted Structured Feature Embedding
        `euclidean_distances` from sklearn
    """
    assert isinstance(x, np.ndarray) and x.ndim == 2
    assert isinstance(y, np.ndarray) and y.ndim == 2
    assert x.shape[1] == y.shape[1]
    
    x_square = np.expand_dims(np.einsum('ij,ij->i', x, x), axis=1)
    if x is y:
        y_square = x_square.T
    else:
        y_square = np.expand_dims(np.einsum('ij,ij->i', y, y), axis=0)
    distances = np.dot(x, y.T)
    # use inplace operation to accelerate
    distances *= -2
    distances += x_square
    distances += y_square
    # result maybe less than 0 due to floating point rounding errors.
    np.maximum(distances, 0, distances)
    if x is y:
        # Ensure that distances between vectors and themselves are set to 0.0.
        # This may not be the case due to floating point rounding errors.
        distances.flat[::distances.shape[0] + 1] = 0.0
    if not squared:
        np.sqrt(distances, distances)
    return distances
    
    