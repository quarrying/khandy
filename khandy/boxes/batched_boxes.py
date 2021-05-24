import numpy as np


def _concat(arr_list, axis=0):
    """Avoids a copy if there is only a single element in a list.
    """
    if len(arr_list) == 1:
        return arr_list[0]
    return np.concatenate(arr_list, axis)
    
    
def convert_boxes_list_to_batched_boxes(boxes_list):
    """
    Args:
        boxes_list: list or tuple of ndarray with shape (N_i, 4+K)
        
    Returns:
        ndarray with shape (M, 5+K) where M is sum of N_i.
        
    References:
        `mmdet.core.bbox.bbox2roi` in mmdetection
        `convert_boxes_to_roi_format` in TorchVision
        `modeling.poolers.convert_boxes_to_pooler_format` in detectron2
    """
    assert isinstance(boxes_list, (list, tuple))
    concat_boxes = _concat(boxes_list, axis=0)
    indices_list = [np.full((len(b), 1), i, concat_boxes.dtype) 
                    for i, b in enumerate(boxes_list)]
    indices = _concat(indices_list, axis=0)
    batched_boxes = np.hstack([indices, concat_boxes])
    return batched_boxes
    
    
def convert_batched_boxes_to_boxes_list(batched_boxes):
    """
    References:
        `mmdet.core.bbox.roi2bbox` in mmdetection
        `convert_boxes_to_roi_format` in TorchVision
        `modeling.poolers.convert_boxes_to_pooler_format` in detectron2
    """
    assert isinstance(batched_boxes, np.ndarray)
    assert batched_boxes.ndim == 2 and batched_boxes.shape[-1] >= 5
    
    boxes_list = []
    indices = np.unique(batched_boxes[:, 0])
    for index in indices:
        inds = (batched_boxes[:, 0] == index)
        boxes = batched_boxes[inds, 1:]
        boxes_list.append(boxes)
    return boxes_list
    
    