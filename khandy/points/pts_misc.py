from typing import Sequence, Tuple, Union

import numpy as np

import khandy


def sort_points(points: np.ndarray, axes: Union[int, Sequence[int]] = -1, 
                reverse: Union[bool, Sequence[bool]] = False) -> Tuple[np.ndarray, np.ndarray]:
    assert isinstance(points, np.ndarray) and points.ndim == 2

    axes = khandy.to_list(axes)
    if isinstance(reverse, bool):
        reverse = [reverse for _ in range(len(axes))]
    elif khandy.is_seq_of(reverse, bool):
        assert len(reverse) == len(axes)
    else:
        raise TypeError('reverse type should be bool or sequence of bool')

    maxs = np.max(points, axis=0)
    mins = np.min(points, axis=0)
    pos = np.zeros((len(points),), dtype=points.dtype)
    for k, (axis, reverse_on_axis) in enumerate(zip(axes, reverse)):
        if k != 0:
            pos *= maxs[axes[k-1]] - mins[axes[k-1]]
        if reverse_on_axis:
            pos += maxs[axis] - points[:, axis]
        else:
            pos += points[:, axis] - mins[axis]

    sorted_inds = np.argsort(pos)
    return sorted_inds, points[sorted_inds]

