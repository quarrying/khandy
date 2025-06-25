import unittest
import numpy as np
import khandy


class DummyDetector(khandy.model.BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, image, **kwargs):
        # 返回一个简单的DetObjects实例
        return khandy.model.DetObjects(
            boxes=np.array([[0, 0, 10, 10], [5, 5, 15, 15]]),
            confs=np.array([0.8, 0.6]),
            classes=np.array([0, 1]),
            class_names=['a', 'b']
        )


class TestBaseDetector(unittest.TestCase):
    def setUp(self):
        self.detector = DummyDetector(num_classes=2)
        self.image = np.zeros((20, 20, 3), dtype=np.uint8)

    def test_initialization(self):
        with self.assertRaises(AssertionError):
            DummyDetector(num_classes=0)
        with self.assertRaises(AssertionError):
            DummyDetector(num_classes=2, class_names=['a'])

    def test_property_setters(self):
        self.detector.conf_thresh = 0.5
        self.assertEqual(self.detector.conf_thresh, 0.5)
        self.detector.iou_thresh = 0.3
        self.assertEqual(self.detector.iou_thresh, 0.3)
        self.detector.min_width = 5
        self.assertEqual(self.detector.min_width, 5)
        self.detector.min_height = 6
        self.assertEqual(self.detector.min_height, 6)
        self.detector.min_area = 10
        self.assertEqual(self.detector.min_area, 10)
        self.detector.class_names = ['a', 'b']
        self.assertEqual(self.detector.class_names, ['a', 'b'])
        self.detector.sort_by = khandy.model.DetObjectSortBy.BY_CONF
        self.assertEqual(self.detector.sort_by, khandy.model.DetObjectSortBy.BY_CONF)
        self.detector.sort_dir = khandy.model.DetObjectSortDir.ASC
        self.assertEqual(self.detector.sort_dir, khandy.model.DetObjectSortDir.ASC)

    def test_filter_methods(self):
        det_objects = self.detector.forward(self.image)
        # filter_by_conf
        self.detector.conf_thresh = 0.7
        filtered = self.detector.filter_by_conf(det_objects)
        self.assertEqual(len(filtered), 1)
        # filter_by_size
        self.detector.min_width = 11
        filtered = self.detector.filter_by_size(det_objects)
        self.assertEqual(len(filtered), 0)
        # filter_by_area
        self.detector.min_area = 100
        filtered = self.detector.filter_by_area(det_objects)
        self.assertEqual(len(filtered), 2)

    def test_call_and_sort(self):
        self.detector.sort_by = khandy.model.DetObjectSortBy.BY_CONF
        self.detector.sort_dir = khandy.model.DetObjectSortDir.ASC
        det_objects = self.detector(self.image)
        self.assertTrue(np.all(np.diff(det_objects.confs) >= 0))


if __name__ == '__main__':
    unittest.main()
