import numpy as np


class FasterRcnnBoxCoder:
    """Faster RCNN box coder.
    
    Notes:
        boxes should be in cxcywh format.
    """
    
    def __init__(self, stddevs=None):
        """Constructor for FasterRcnnBoxCoder.
      
        Args:
          stddevs: List of 4 positive scalars to scale ty, tx, th and tw.
            If set to None, does not perform scaling. For Faster RCNN,
            the open-source implementation recommends using [0.1, 0.1, 0.2, 0.2].
        """
        if stddevs:
            assert len(stddevs) == 4
            for scalar in stddevs:
                assert scalar > 0
        self.stddevs = stddevs

    def encode(self, boxes, reference_boxes, copy=True):
        """Encode boxes with respect to reference boxes.
        """
        if copy:
            boxes = boxes.copy()
            
        boxes[..., 2:4] += 1e-8
        reference_boxes[..., 2:4] += 1e-8
        
        boxes[..., 0:2] -= reference_boxes[..., 0:2]
        boxes[..., 0:2] /= reference_boxes[..., 2:4]
        boxes[..., 2:4] /= reference_boxes[..., 2:4]
        boxes[..., 2:4] = np.log(boxes[..., 2:4], boxes[..., 2:4])
        if self.stddevs:
            boxes[..., 0:4] /= self.stddevs
        return boxes

    def decode(self, rel_boxes, reference_boxes, copy=True):
        """Decode relative codes to boxes.
        """
        if copy:
            rel_boxes = rel_boxes.copy()
            
        if self.stddevs:
            rel_boxes[..., 0:4] *= self.stddevs
        
        rel_boxes[..., 0:2] *= reference_boxes[..., 2:4]
        rel_boxes[..., 0:2] += reference_boxes[..., 0:2]
        rel_boxes[..., 2:4] = np.exp(rel_boxes[..., 2:4], rel_boxes[..., 2:4])
        rel_boxes[..., 2:4] *= reference_boxes[..., 2:4]
        return rel_boxes
    
    def decode_points(self, rel_points, reference_boxes, copy=True):
        """Decode relative codes to points.
        """
        if copy:
            rel_points = rel_points.copy()
        if self.stddevs:
            rel_points[..., 0::2] *= self.stddevs[0]
            rel_points[..., 1::2] *= self.stddevs[1]
        rel_points[..., 0::2] *= reference_boxes[..., 2:3]
        rel_points[..., 1::2] *= reference_boxes[..., 3:4]
        rel_points[..., 0::2] += reference_boxes[..., 0:1]
        rel_points[..., 1::2] += reference_boxes[..., 1:2]
        return rel_points
