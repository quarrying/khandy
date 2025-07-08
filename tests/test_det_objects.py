import unittest
import warnings

import khandy
import numpy as np


def random_boxes(x_min, y_min, x_max, y_max, num_boxes=1):
    random_xs = np.random.uniform(x_min, x_max, size=(num_boxes, 2))
    random_ys = np.random.uniform(y_min, y_max, size=(num_boxes, 2))
    x_mins, x_maxs = [], []
    for x in random_xs:
        x_mins.append(np.min(x))
        x_maxs.append(np.max(x))
    y_mins, y_maxs = [], []
    for y in random_ys:
        y_mins.append(np.min(y))
        y_maxs.append(np.max(y))
    return np.stack([x_mins, y_mins, x_maxs, y_maxs], axis=-1)


class TestDetObjects(unittest.TestCase):
    def setUp(self):
        self.det_objects = khandy.model.DetObjects(
            boxes=random_boxes(0, 0, 10, 10, num_boxes=10),
            classes=np.array([0, 1, 1, 3, 3, 5, 6, 7, 8, 9]),
            confs=np.linspace(0.1, 1.0, 10),
            class_names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
            extras=np.arange(10)
        )

    def test_getitem(self):
        # Test single index
        item = self.det_objects[0]
        self.assertEqual(item.class_index, self.det_objects.classes[0])
        self.assertAlmostEqual(item.conf, self.det_objects.confs[0])
        self.assertEqual(item.class_name, self.det_objects.class_names[0])
        self.assertEqual(item.extras, [self.det_objects.extras[0]])
        
        # Test slice
        sliced = self.det_objects[:3]
        self.assertEqual(len(sliced), 3)
        self.assertTrue(np.array_equal(sliced.boxes, self.det_objects.boxes[:3]))
        self.assertTrue(np.array_equal(sliced.classes, self.det_objects.classes[:3]))
        self.assertTrue(np.array_equal(sliced.confs, self.det_objects.confs[:3]))
        self.assertEqual(sliced.class_names, self.det_objects.class_names[:3])
        self.assertTrue(np.array_equal(sliced.extras, self.det_objects.extras[:3]))

        self.assertEqual(self.det_objects[np.int32(0)].class_index, self.det_objects.classes[0])
        self.assertEqual(self.det_objects[np.int64(0)].class_index, self.det_objects.classes[0])

    def test_filter_by_class_names(self):
        result = self.det_objects.filter_by_class_names(inplace=False)
        self.assertEqual(result.class_names, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

        result = self.det_objects.filter_by_class_names(ignored='d', inplace=False)
        self.assertEqual(result.class_names, ['a', 'b', 'c', 'e', 'f', 'g', 'h', 'i', 'j'])

        result = self.det_objects.filter_by_class_names(interested='a', inplace=False)
        self.assertEqual(result.class_names, ['a'])

        result = self.det_objects.filter_by_class_names(ignored=['d', 'e'], inplace=False)
        self.assertEqual(result.class_names, ['a', 'b', 'c', 'f', 'g', 'h', 'i', 'j'])

        result = self.det_objects.filter_by_class_names(interested=['a', 'b', 'c'], inplace=False)
        self.assertEqual(result.class_names, ['a', 'b', 'c'])

        result = self.det_objects.filter_by_class_names(ignored=['d', 'e', 'x'], inplace=False)
        self.assertEqual(result.class_names, ['a', 'b', 'c', 'f', 'g', 'h', 'i', 'j'])

        result = self.det_objects.filter_by_class_names(interested=['a', 'b', 'c', 'x'], inplace=False)
        self.assertEqual(result.class_names, ['a', 'b', 'c'])

    def test_filter_by_class_indices(self):
        result = self.det_objects.filter_by_class_indices(inplace=False)
        self.assertTrue(np.array_equal(result.classes, [0, 1, 1, 3, 3, 5, 6, 7, 8, 9]))

        result = self.det_objects.filter_by_class_indices(ignored=0, inplace=False)
        self.assertTrue(np.array_equal(result.classes, [1, 1, 3, 3, 5, 6, 7, 8, 9]))

        result = self.det_objects.filter_by_class_indices(ignored=1, inplace=False)
        self.assertTrue(np.array_equal(result.classes, [0, 3, 3, 5, 6, 7, 8, 9]))

        result = self.det_objects.filter_by_class_indices(interested=0, inplace=False)
        self.assertTrue(np.array_equal(result.classes, [0]))

        result = self.det_objects.filter_by_class_indices(interested=1, inplace=False)
        self.assertTrue(np.array_equal(result.classes, [1, 1]))

        result = self.det_objects.filter_by_class_indices(ignored=[0, 1], inplace=False)
        self.assertTrue(np.array_equal(result.classes, [3, 3, 5, 6, 7, 8, 9]))

        result = self.det_objects.filter_by_class_indices(interested=[3, 4, 5], inplace=False)
        self.assertTrue(np.array_equal(result.classes, [3, 3, 5]))

        result = self.det_objects.filter_by_class_indices(ignored=[0, 1, 100], inplace=False)
        self.assertTrue(np.array_equal(result.classes, [3, 3, 5, 6, 7, 8, 9]))

        result = self.det_objects.filter_by_class_indices(interested=[3, 4, 5, 100], inplace=False)
        self.assertTrue(np.array_equal(result.classes, [3, 3, 5]))

    def _assert_equal(self, a, b):
        self.assertEqual(len(a), len(b))
        self.assertTrue(np.array_equal(a.boxes, b.boxes))
        self.assertTrue(np.array_equal(a.classes, b.classes))
        self.assertTrue(np.array_equal(a.confs, b.confs))
        self.assertEqual(a.class_names, b.class_names)

    def test_filter_misc(self):
        warnings.filterwarnings("ignore", category=UserWarning)
        
        num_objects = 100
        det_objects = khandy.model.DetObjects(
            boxes=random_boxes(0, 0, 100, 100, num_boxes=num_objects),
            confs=np.random.uniform(0, 1, size=(num_objects,)),
        )
        boxes = khandy.Boxes(det_objects.boxes)
        self._assert_equal(det_objects.filter_by_conf(0.5), 
                           det_objects[det_objects.confs > 0.5])
        self._assert_equal(det_objects.filter_by_size(min_width=50), 
                           det_objects[boxes.widths[:, 0] >= 50])
        self._assert_equal(det_objects.filter_by_size(min_height=50), 
                           det_objects[boxes.heights[:, 0] >= 50])
        self._assert_equal(det_objects.filter_by_area(2500), 
                           det_objects[boxes.areas[:, 0] >= 2500])
        
    def test_filter_by_func(self):
        det_objects = khandy.model.DetObjects(
            boxes=np.array([[0, 0, 1, 1], [1, 1, 2, 2]]),
            confs=np.array([0.8, 0.9]),
            classes=np.array([0, 1]),
            class_names=["class_0", "class_1"]
        )
        
        def func(item: khandy.model.DetObjectItem) -> bool:
            return item.conf > 0.85
        
        filtered_objects = det_objects.filter_by_func(func)
        self.assertEqual(len(filtered_objects), 1)
        self.assertAlmostEqual(filtered_objects.confs[0], 0.9)
        self.assertEqual(len(det_objects), 2)
        self.assertAlmostEqual(det_objects.confs[0], 0.8)
        
        det_objects.filter_by_func(func, inplace=True)
        self.assertEqual(len(filtered_objects), 1)
        self.assertAlmostEqual(filtered_objects.confs[0], 0.9)
        self.assertEqual(len(det_objects), 1)
        self.assertAlmostEqual(det_objects.confs[0], 0.9)
        
    def test_filter_by_ar(self):
        boxes = np.array([
            [0, 0, 10, 10],   # ar = 1.0
            [0, 0, 20, 10],   # ar = 2.0
            [0, 0, 10, 20],   # ar = 0.5
            [0, 0, 0, 10],    # ar = 0.0 (width=0)
            [0, 0, 10, 0],    # ar = inf (height=0, will be filtered out)
        ])
        det_objects = khandy.model.DetObjects(
            boxes=boxes,
            confs=np.ones(5),
            classes=np.arange(5),
            class_names=[str(i) for i in range(5)]
        )
        filtered = det_objects.filter_by_ar(min_ar=0.8, max_ar=1.5)
        self.assertEqual(len(filtered), 1)
        self.assertTrue(np.allclose(filtered.boxes, [[0, 0, 10, 10]]))

        filtered = det_objects.filter_by_ar(min_ar=0.5, max_ar=2.0)
        self.assertEqual(len(filtered), 3)
        self.assertTrue(np.allclose(filtered.boxes, boxes[[0,1,2]]))

        det_objects2 = khandy.model.DetObjects(
            boxes=boxes,
            confs=np.ones(5),
            classes=np.arange(5),
            class_names=[str(i) for i in range(5)]
        )
        det_objects2.filter_by_ar(min_ar=1.8, inplace=True)
        self.assertEqual(len(det_objects2), 1)
        self.assertTrue(np.allclose(det_objects2.boxes, [[0, 0, 20, 10]]))

    def test_filter_with_empty_det_objects(self):
        def func(item: khandy.model.DetObjectItem) -> bool:
            return item.conf > 0.85

        det_objects = khandy.model.DetObjects()
        det_objects2 = det_objects.filter_by_conf(conf_thresh=0.5)
        self.assertTrue(len(det_objects2) == 0)
        det_objects2 = det_objects.filter_by_area(min_area=10)
        self.assertTrue(len(det_objects2) == 0)
        det_objects2 = det_objects.filter_by_size(min_width=10, min_height=10)
        self.assertTrue(len(det_objects2) == 0)
        det_objects2 = det_objects.filter_by_ar(min_ar=1.8)
        self.assertEqual(len(det_objects2), 0)
        det_objects2 = det_objects.filter_by_class_names(ignored=['car'])
        self.assertTrue(len(det_objects2) == 0)
        det_objects2 = det_objects.filter_by_class_indices(ignored=[0])
        self.assertTrue(len(det_objects2) == 0)
        det_objects2 = det_objects.filter_by_func(func)
        self.assertTrue(len(det_objects2) == 0)

        det_objects2 = det_objects.filter_by_conf(conf_thresh=0.5, inplace=True)
        self.assertTrue(len(det_objects2) == 0)
        det_objects2 = det_objects.filter_by_area(min_area=10, inplace=True)
        self.assertTrue(len(det_objects2) == 0)
        det_objects2 = det_objects.filter_by_size(min_width=10, min_height=10, inplace=True)
        self.assertTrue(len(det_objects2) == 0)
        det_objects2 = det_objects.filter_by_ar(min_ar=1.8, inplace=True)
        self.assertEqual(len(det_objects2), 0)
        det_objects2 = det_objects.filter_by_class_names(ignored=['car'], inplace=True)
        self.assertTrue(len(det_objects2) == 0)
        det_objects2 = det_objects.filter_by_class_indices(ignored=[0], inplace=True)
        self.assertTrue(len(det_objects2) == 0)
        det_objects2 = det_objects.filter_by_func(func, inplace=True)
        self.assertTrue(len(det_objects2) == 0)


class TestConcatDetObjects(unittest.TestCase):
    def setUp(self):
        self.boxes1 = np.random.randn(3, 4)
        self.boxes2 = np.random.randn(3, 4)
        self.boxes3 = np.random.randn(3, 4)
        self.det_objects1 = khandy.model.DetObjects(boxes=self.boxes1)
        self.det_objects2 = khandy.model.DetObjects(boxes=self.boxes2)
        self.det_objects3 = khandy.model.DetObjects(boxes=self.boxes3, extras=[1, 2, 5])
    
    def test_concat_all_fields(self):
        result = khandy.model.concat_det_objects([self.det_objects1, self.det_objects2])
        self.assertTrue(np.array_equal(result.boxes, np.concatenate([self.boxes1, self.boxes2])))
    
    def test_concat_common_fields(self):
        result = khandy.model.concat_det_objects([self.det_objects1, self.det_objects2, self.det_objects3], only_common_fields=True)
        self.assertTrue(np.array_equal(result.boxes, np.concatenate([self.boxes1, self.boxes2, self.boxes3])))
    
    def test_empty_list(self):
        self.assertTrue(len(khandy.model.concat_det_objects([])) == 0)
        self.assertTrue(len(khandy.model.concat_det_objects([khandy.model.DetObjects()])) == 0)
        self.assertTrue(len(khandy.model.concat_det_objects([khandy.model.DetObjects(), khandy.model.DetObjects()])) == 0)

    def test_different_fields(self):
        with self.assertRaises(ValueError):
            khandy.model.concat_det_objects([self.det_objects1, self.det_objects2, self.det_objects3], only_common_fields=False)

    def test_concat_with_det_object_item(self):
        item = khandy.model.DetObjectItem(
            x_min=1.0, y_min=2.0, x_max=3.0, y_max=4.0,
            conf=0.9, class_index=0, class_name="a",
            class_extras=['c']
        )
        det_objects = khandy.model.DetObjects(
            boxes=np.array([[5.0, 6.0, 7.0, 8.0]]),
            confs=np.array([0.8]),
            classes=np.array([1]),
            class_names=["b"],
            class_extras=['d']
        )
        result = khandy.model.concat_det_objects([item, det_objects])
        self.assertEqual(len(result), 2)
        self.assertTrue(np.allclose(result.boxes[0], [1.0, 2.0, 3.0, 4.0]))
        self.assertTrue(np.allclose(result.boxes[1], [5.0, 6.0, 7.0, 8.0]))
        self.assertTrue(np.allclose(result.confs, [0.9, 0.8]))
        self.assertTrue(result.class_extras, ['c', 'd'])


if __name__ == '__main__':
    unittest.main()
    
    