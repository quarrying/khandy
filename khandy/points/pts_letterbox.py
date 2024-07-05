__all__ = ['letterbox_2d_points', 'unletterbox_2d_points']


def letterbox_2d_points(points, lb_detail, copy=True):
    if copy:
        points = points.copy()
    points[..., 0::2] = points[..., 0::2] * lb_detail.x_scale + lb_detail.pad_left
    points[..., 1::2] = points[..., 1::2] * lb_detail.y_scale + lb_detail.pad_top
    return points


def unletterbox_2d_points(points, lb_detail, copy=True):
    if copy:
        points = points.copy()
        
    points[..., 0::2] = (points[..., 0::2] - lb_detail.pad_left) / lb_detail.x_scale
    points[..., 1::2] = (points[..., 1::2] - lb_detail.pad_top) / lb_detail.y_scale
    return points
    
