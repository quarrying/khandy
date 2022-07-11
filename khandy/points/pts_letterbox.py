__all__ = ['letterbox_2d_points', 'unletterbox_2d_points']


def letterbox_2d_points(points, scale=1.0, pad_left=0, pad_top=0, copy=True):
    if copy:
        points = points.copy()
    points[..., 0::2] = points[..., 0::2] * scale + pad_left
    points[..., 1::2] = points[..., 1::2] * scale + pad_top
    return points
    
    
def unletterbox_2d_points(points, scale=1.0, pad_left=0, pad_top=0, copy=True):
    if copy:
        points = points.copy()
        
    points[..., 0::2] = (points[..., 0::2] - pad_left) / scale
    points[..., 1::2] = (points[..., 1::2] - pad_top) / scale
    return points
    
