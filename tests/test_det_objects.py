import unittest
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
            class_names=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
        )

    def test_filter_by_class_names(self):
        result1 = self.det_objects.filter_by_class_names(inplace=False)
        self.assertEqual(result1.class_names, ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

        result2 = self.det_objects.filter_by_class_names(ignored=['d', 'e'], inplace=False)
        self.assertEqual(result2.class_names, ['a', 'b', 'c', 'f', 'g', 'h', 'i', 'j'])

        result3 = self.det_objects.filter_by_class_names(interested=['a', 'b', 'c'], inplace=False)
        self.assertEqual(result3.class_names, ['a', 'b', 'c'])

        result4 = self.det_objects.filter_by_class_names(ignored=['d', 'e', 'x'], inplace=False)
        self.assertEqual(result4.class_names, ['a', 'b', 'c', 'f', 'g', 'h', 'i', 'j'])

        result5 = self.det_objects.filter_by_class_names(interested=['a', 'b', 'c', 'x'], inplace=False)
        self.assertEqual(result5.class_names, ['a', 'b', 'c'])

    def test_filter_by_class_indices(self):
        result1 = self.det_objects.filter_by_class_indices(inplace=False)
        self.assertTrue(np.array_equal(result1.classes.flatten(), [0, 1, 1, 3, 3, 5, 6, 7, 8, 9]))

        result2 = self.det_objects.filter_by_class_indices(ignored=[0, 1], inplace=False)
        self.assertTrue(np.array_equal(result2.classes.flatten(), [3, 3, 5, 6, 7, 8, 9]))

        result3 = self.det_objects.filter_by_class_indices(interested=[3, 4, 5], inplace=False)
        self.assertTrue(np.array_equal(result3.classes.flatten(), [3, 3, 5]))

        result4 = self.det_objects.filter_by_class_indices(ignored=[0, 1, 100], inplace=False)
        self.assertTrue(np.array_equal(result4.classes.flatten(), [3, 3, 5, 6, 7, 8, 9]))

        result5 = self.det_objects.filter_by_class_indices(interested=[3, 4, 5, 100], inplace=False)
        self.assertTrue(np.array_equal(result5.classes.flatten(), [3, 3, 5]))

    def _assert_equal(self, a, b):
        self.assertEqual(len(a), len(b))
        self.assertTrue(np.array_equal(a.boxes, b.boxes))
        self.assertTrue(np.array_equal(a.classes, b.classes))
        self.assertTrue(np.array_equal(a.confs, b.confs))
        self.assertEqual(a.class_names, b.class_names)

    def test_filter_misc(self):
        num_objects = 100
        det_objects = khandy.model.DetObjects(
            boxes=random_boxes(0, 0, 100, 100, num_boxes=num_objects),
            confs=np.random.uniform(0, 1, size=(num_objects,)),
        )
        boxes = khandy.Boxes(det_objects.boxes)
        self._assert_equal(det_objects.filter_by_conf(0.5), 
                           det_objects[det_objects.confs > 0.5])
        self._assert_equal(det_objects.filter_by_min_size(min_width=50), 
                           det_objects[boxes.widths[:, 0] >= 50])
        self._assert_equal(det_objects.filter_by_min_size(min_height=50), 
                           det_objects[boxes.heights[:, 0] >= 50])
        self._assert_equal(det_objects.filter_by_min_area(2500), 
                           det_objects[boxes.areas[:, 0] >= 2500])
        

if __name__ == '__main__':
    unittest.main()
    
    