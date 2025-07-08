import unittest
import khandy


class TestDetObjectItem(unittest.TestCase):
    def setUp(self):
        self.item = khandy.model.DetObjectItem(
            x_min=10.0, y_min=20.0, x_max=30.0, y_max=40.0,
            conf=0.95, class_index=1, class_name="person",
            colors=["red"], sizes=["large"]
        )
        print(self.item.__dict__)
        print(self.item.__annotations__)

    def test_field_access(self):
        self.assertEqual(self.item.x_min, 10.0)
        self.assertEqual(self.item.y_min, 20.0)
        self.assertEqual(self.item.x_max, 30.0)
        self.assertEqual(self.item.y_max, 40.0)
        self.assertEqual(self.item.conf, 0.95)
        self.assertEqual(self.item.class_index, 1)
        self.assertEqual(self.item.class_name, "person")
        self.assertEqual(self.item.colors, ["red"])
        self.assertEqual(self.item.sizes, ["large"])
        with self.assertRaises(AttributeError):
            _ = self.item.non_existent_field
            
    def test_init(self):
        with self.assertRaises(AssertionError):
            khandy.model.DetObjectItem(
                x_min=10.0, y_min=20.0, x_max=30.0, y_max=40.0,
                conf=0.95, class_index=1, class_name="person",
                colors=0
            )
        with self.assertRaises(AssertionError):
            khandy.model.DetObjectItem(
                x_min=10.0, y_min=20.0, x_max=30.0, y_max=40.0,
                conf=0.95, class_index=1, class_name="person",
                colors=["red", 'green']
            )
        with self.assertRaises(AssertionError):
            khandy.model.DetObjectItem(
                x_min=10.0, y_min=20.0, x_max=30.0, y_max=40.0,
                conf=0.95, class_index=1, class_name="person",
                **{"colors": 0}
            )
        with self.assertRaises(AssertionError):
            khandy.model.DetObjectItem(
                x_min=10.0, y_min=20.0, x_max=30.0, y_max=40.0,
                conf=0.95, class_index=1, class_name="person",
                **{"colors": ["red", 'green']}
            )
        with self.assertRaises(TypeError):
            khandy.model.DetObjectItem(
                x_min=10.0, y_min=20.0, x_max=30.0, y_max=40.0,
                conf=0.95, class_index=1, class_name="person",
                **{"class_name": ["a"]}
            )
        with self.assertRaises(AttributeError):
            khandy.model.DetObjectItem(
                x_min=10.0, y_min=20.0, x_max=30.0, y_max=40.0,
                conf=0.95, class_index=1, class_name="person",
                **{"area": ["a"]}
            )

    def test_field_modification(self):
        self.item.x_min = 10.5
        self.assertAlmostEqual(self.item.x_min, 10.5)
        self.item.colors = ["blue"]
        self.assertEqual(self.item.colors, ["blue"])
        self.item.foobar = ["new_value"]
        self.assertEqual(self.item.foobar, ["new_value"])
        
        with self.assertRaises(AssertionError):
            self.item.colors = "blue"
        with self.assertRaises(AssertionError):
            self.item.colors = ["blue", 'green']
       
        with self.assertRaises(AssertionError):
            self.item.new_attr1 = "blue"
        with self.assertRaises(AssertionError):
            self.item.new_attr2 = ["blue", 'green']
            
    def test_to_det_objects(self):
        det_objects = self.item.to_det_objects()
        self.assertIsInstance(det_objects, khandy.model.DetObjects)
        self.assertEqual(len(det_objects), 1)
        self.assertIsInstance(det_objects[0], khandy.model.DetObjectItem)

    def test_properties(self):
        item = khandy.model.DetObjectItem(
            x_min=10.0, y_min=10.0, x_max=30.0, y_max=40.0,
            conf=0.95, class_index=1, class_name="person"
        )
        self.assertAlmostEqual(item.area, 600.0)
        self.assertAlmostEqual(item.width, 20.0)
        self.assertAlmostEqual(item.height, 30.0)
        self.assertAlmostEqual(item.x_center, 20.0)
        self.assertAlmostEqual(item.y_center, 25.0)
        with self.assertRaises(AttributeError):
            # AttributeError: can't set attribute
            item.area = 100.0



if __name__ == '__main__':
    unittest.main()
    
