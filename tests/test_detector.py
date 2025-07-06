import unittest
import numpy as np
import khandy


class DummyDetector(khandy.model.BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, image, **kwargs):
        return khandy.model.DetObjects(
            boxes=np.array([[0, 0, 10, 10], [5, 5, 15, 15]]),
            confs=np.array([0.8, 0.6]),
            classes=np.array([0, 1]),
            class_names=['a', 'b']
        )


class EmptyDetector(khandy.model.BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, image, **kwargs):
        return khandy.model.DetObjects()
            
class TestBaseDetector(unittest.TestCase):
    def setUp(self):
        self.detector = DummyDetector(num_classes=2)
        self.image = np.zeros((20, 20, 3), dtype=np.uint8)

    def test_initialization(self):
        # test num_classes initialization
        with self.assertRaisesRegex(AssertionError, 'num_classes must be a positive integer*'):
            DummyDetector(num_classes=0.5)
        with self.assertRaisesRegex(AssertionError, 'num_classes must be a positive integer*'):
            DummyDetector(num_classes=0)
        
        # test class_names initialization
        DummyDetector(num_classes=2, class_names=['a', 'b'])
        DummyDetector(num_classes=2, class_names=('a', 'b'))
        with self.assertRaises(TypeError):
            DummyDetector(num_classes=2, class_names='a')
        with self.assertRaisesRegex(AssertionError, 'num_classes must be set before*'):
            DummyDetector(num_classes=None, class_names=['a', 'b'])
        with self.assertRaisesRegex(AssertionError, 'class_names must be a list or tuple of strings*'):
            DummyDetector(num_classes=2, class_names=['a', 0])
        with self.assertRaisesRegex(AssertionError, 'class_names must have length*'):
            DummyDetector(num_classes=2, class_names=['a'])

        # test conf_thresh initialization
        DummyDetector(num_classes=2, conf_thresh=0.5)
        DummyDetector(num_classes=2, conf_thresh=[0.5, 0.5])
        DummyDetector(num_classes=2, conf_thresh=(0.5, 0.5))
        DummyDetector(num_classes=2, conf_thresh=np.array([0.5, 0.5]))
        DummyDetector(num_classes=2, conf_thresh=np.array([[0.5], [0.5]]))
        with self.assertRaises(TypeError):
            DummyDetector(num_classes=2, conf_thresh='0.5')
        with self.assertRaisesRegex(AssertionError, 'num_classes must be set before*'):
            DummyDetector(num_classes=None, conf_thresh=[0.5, 0.5])
        with self.assertRaisesRegex(AssertionError, 'conf_thresh must be a list or tuple of floats*'):
            DummyDetector(num_classes=2, conf_thresh=[0.5, '0.5'])
        with self.assertRaisesRegex(AssertionError, 'conf_thresh must have length*'):
            DummyDetector(num_classes=2, conf_thresh=[0.5])
            
        with self.assertRaisesRegex(AssertionError, 'num_classes must be set before*'):
            DummyDetector(num_classes=None, conf_thresh=np.array([0.5, 0.5]))
        with self.assertRaisesRegex(AssertionError, 'conf_thresh must be a numpy array of floats*'):
            DummyDetector(num_classes=2, conf_thresh=np.array([True, False]))
        with self.assertRaisesRegex(AssertionError, 'conf_thresh shape must be*'):
            DummyDetector(num_classes=2, conf_thresh=np.array([0.5, 0.5, 0.5]))

        with self.assertRaisesRegex(AssertionError, 'conf_thresh must be >= 0*'):
            DummyDetector(num_classes=2, conf_thresh=-0.5)
        with self.assertRaisesRegex(AssertionError, 'conf_thresh must be >= 0*'):
            DummyDetector(num_classes=2, conf_thresh=[-0.5, 0.5])
        with self.assertRaisesRegex(AssertionError, 'conf_thresh must be >= 0*'):
            DummyDetector(num_classes=2, conf_thresh=np.array([-0.5, 0.5]))
            
        with self.assertRaisesRegex(AssertionError, 'conf_thresh must be <= 1*'):
            DummyDetector(num_classes=2, conf_thresh=1.5)
        with self.assertRaisesRegex(AssertionError, 'conf_thresh must be <= 1*'):
            DummyDetector(num_classes=2, conf_thresh=[1.5, 0.5])
        with self.assertRaisesRegex(AssertionError, 'conf_thresh must be <= 1*'):
            DummyDetector(num_classes=2, conf_thresh=np.array([1.5, 0.5]))
            
    def test_property_setters(self):
        self.detector.conf_thresh = 0.5
        self.assertEqual(self.detector.conf_thresh, 0.5)
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

    def test_detect_in_det_objects(self):
        # 构造一个包含两个目标的 DetObjects
        det_objects = khandy.model.DetObjects(
            boxes=np.array([[1, 2, 11, 12], [5, 5, 15, 15]]),
            confs=np.array([0.9, 0.8]),
            classes=np.array([0, 1]),
            class_names=['a', 'b']
        )
        image = np.ones((20, 20, 3), dtype=np.uint8)
        detector = DummyDetector(num_classes=2)

        result = khandy.model.detect_in_det_objects(detector, image, det_objects)
        self.assertIsInstance(result, khandy.model.DetObjects)
        self.assertEqual(len(result), 4)
        self.assertTrue(np.all(result.boxes[:2, 0] >= 1))
        self.assertTrue(np.all(result.boxes[:2, 1] >= 2))
        self.assertTrue(np.all(result.boxes[2:, 0] >= 5))
        self.assertTrue(np.all(result.boxes[2:, 1] >= 5))

    def test_max_class_index(self):
        detector = DummyDetector(num_classes=1)
        with self.assertRaisesRegex(AssertionError, 'max of det_objects.classes must be*'):
            detector(self.image)

        detector = DummyDetector(num_classes=2)
        self.assertTrue(len(detector(self.image)) ==2)

        detector = EmptyDetector(num_classes=1)
        self.assertTrue(len(detector(self.image)) == 0)


class Dummy3ClassesDetector(khandy.model.BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(num_classes=3, class_names=['a', 'b', 'c'], **kwargs)
    def forward(self, image, **kwargs):
        return khandy.model.DetObjects(
            boxes=np.array([[0, 0, 10, 10], [5, 5, 15, 15], [1, 1, 2, 2]]),
            confs=np.array([0.8, 0.6, 0.9]),
            classes=np.array([0, 1, 2]),
            class_names=['a', 'b', 'c']
        )

class TestSubsetDetector(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((20, 20, 3), dtype=np.uint8)
        self.detector = Dummy3ClassesDetector()

    def test_subset_detector(self):
        subset = khandy.model.SubsetDetector(self.detector, ['a', 'c'])
        det_objects = subset(self.image)
        self.assertEqual(len(det_objects), 2)
        self.assertListEqual(det_objects.class_names, ['a', 'c'])
        self.assertTrue(np.all(det_objects.classes == [0, 1]))
        self.assertTrue(np.allclose(det_objects.boxes[0], [0, 0, 10, 10]))
        self.assertTrue(np.allclose(det_objects.boxes[1], [1, 1, 2, 2]))
        self.assertTrue(np.allclose(det_objects.confs, [0.8, 0.9]))

    def test_invalid_subset(self):
        with self.assertRaises(AssertionError):
            khandy.model.SubsetDetector(self.detector, ['a', 'a'])
        with self.assertRaises(AssertionError):
            khandy.model.SubsetDetector(self.detector, ['a', 'd'])


class KwargsDetector(khandy.model.BaseDetector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, image, **kwargs):
        return khandy.model.DetObjects(
            boxes=np.array([[0, 0, 1, 1]]),
            confs=np.array([1.0]),
            classes=np.array([0]),
            class_names=['a']
        )


class TestBaseDetectorKwargs(unittest.TestCase):
    def test_kwargs_passed(self):
        detector = KwargsDetector(num_classes=1, foo='bar', bar=123)
        self.assertEqual(detector.foo, 'bar')
        self.assertEqual(detector.bar, 123)
 
        # Ensure normal detection still works
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        det_objects = detector(image)
        self.assertEqual(len(det_objects), 1)


if __name__ == '__main__':
    unittest.main()
