import unittest
import numpy as np

import khandy


class TestFilterBoxesByOverlap(unittest.TestCase):
    def test_filter_boxes_by_overlap_ioa(self):
        boxes = np.array([
            [0, 0, 9, 9],
            [0, 0, 11, 11],
            [9, 9, 20, 20]
        ])
        reference_box = [0, 0, 10, 10]
        result = khandy.filter_boxes_by_overlap(boxes, reference_box, ratio_type='ioa', overlap_ratio=0.5)
        np.testing.assert_array_equal(result, [0, 1])

        
if __name__ == '__main__':
    unittest.main()