import khandy
import numpy as np


def sort_points(points, axes=-1, reverse=False):
    assert isinstance(points, np.ndarray) and points.ndim == 2

    axes = khandy.to_list(axes)
    if isinstance(reverse, bool):
        reverse = [reverse for _ in range(len(axes))]
    elif khandy.is_seq_of(reverse, bool):
        assert len(reverse) == len(axes)
    else:
        raise TypeError('reverse type should be bool or sequence of bool')
    
    sorted_inds = np.arange(len(points))
    for k in range(len(axes)):
        inds = np.argsort(points[sorted_inds, axes[k]], kind='stable')
        # # Following codes is used for test.
        # print(np.allclose(points[sorted_inds[inds], axes[k]], 
        #                   np.sort(points[sorted_inds, axes[k]])))
        sorted_inds = sorted_inds[inds]
        if reverse[k]:
            sorted_inds = sorted_inds[::-1]
    return sorted_inds, points[sorted_inds]

