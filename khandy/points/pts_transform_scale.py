import numpy as np

__all__ = ['scale_2d_points']


def scale_2d_points(points, x_scale=1, y_scale=1, x_center=0, y_center=0, copy=True):
    """Scale 2d points.
    
    Args:
        points: (..., 2N)
        x_scale: scale factor in x dimension
        y_scale: scale factor in y dimension
        x_center: scale center in x dimension
        y_center: scale center in y dimension
    """
    points = np.array(points, dtype=np.float32, copy=copy)
    x_scale = np.asarray(x_scale, np.float32)
    y_scale = np.asarray(y_scale, np.float32)
    x_center = np.asarray(x_center, np.float32)
    y_center = np.asarray(y_center, np.float32)
    
    x_shift = 1 - x_scale
    y_shift = 1 - y_scale
    x_shift *= x_center
    y_shift *= y_center
    
    points[..., 0::2] *= x_scale
    points[..., 1::2] *= y_scale
    points[..., 0::2] += x_shift
    points[..., 1::2] += y_shift
    return points
    
    
