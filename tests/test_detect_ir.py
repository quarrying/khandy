import unittest
import khandy


class TestFliterDetectByLabel(unittest.TestCase):
    def setUp(self):
        self.detect_ir_objects = [
            khandy.label.DetectIrObject(label='label1', x_min=0, y_min=0, x_max=10, y_max=10), 
            khandy.label.DetectIrObject(label='label1', x_min=0, y_min=0, x_max=10, y_max=10), 
            khandy.label.DetectIrObject(label='label2', x_min=0, y_min=0, x_max=10, y_max=10)]  # Add more test objects

    def test_fliter_with_single_label(self):
        result = khandy.label.fliter_detect_by_label(self.detect_ir_objects, 'label1')
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].label, 'label1')
        self.assertEqual(result[1].label, 'label1')
        result = khandy.label.fliter_detect_by_label(self.detect_ir_objects, 'label2')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].label, 'label2')
        
    def test_fliter_with_multiple_labels(self):
        result = khandy.label.fliter_detect_by_label(self.detect_ir_objects, ['label1', 'label2'])
        self.assertEqual(len(result), 3)

    def test_fliter_with_callable(self):
        result = khandy.label.fliter_detect_by_label(self.detect_ir_objects, lambda x: x.startswith('label'))
        self.assertEqual(len(result), 3)

    def test_fliter_with_no_match(self):
        result = khandy.label.fliter_detect_by_label(self.detect_ir_objects, 'label3')
        self.assertEqual(len(result), 0)

    def test_fliter_with_empty_input(self):
        result = khandy.label.fliter_detect_by_label([], 'label1')
        self.assertEqual(result, [])


class TestDetectIrObject(unittest.TestCase):
    def test_initialization(self):
        obj =  khandy.label.DetectIrObject(label="test", x_min=1.0, y_min=1.0, x_max=2.0, y_max=2.0)
        self.assertEqual(obj.label, "test")
        self.assertEqual(obj.x_min, 1.0)
        self.assertEqual(obj.y_min, 1.0)
        self.assertEqual(obj.x_max, 2.0)
        self.assertEqual(obj.y_max, 2.0)

    def test_area(self):
        obj =  khandy.label.DetectIrObject(label="test", x_min=1.0, y_min=1.0, x_max=3.0, y_max=4.0)
        self.assertEqual(obj.area, 6.0)

    def test_swapping_values(self):
        with self.assertWarns(Warning):
            obj =  khandy.label.DetectIrObject(label="test", x_min=3.0, y_min=4.0, x_max=1.0, y_max=2.0)
        self.assertEqual(obj.x_min, 1.0)
        self.assertEqual(obj.x_max, 3.0)
        self.assertEqual(obj.y_min, 2.0)
        self.assertEqual(obj.y_max, 4.0)

    def test_properties(self):
        obj =  khandy.label.DetectIrObject(label="test", x_min=1.0, y_min=1.0, x_max=3.0, y_max=4.0)
        self.assertAlmostEqual(obj.width, 2.0)
        self.assertAlmostEqual(obj.height, 3.0)
        self.assertAlmostEqual(obj.x_center, 2.0)
        self.assertAlmostEqual(obj.y_center, 2.5)
        self.assertAlmostEqual(obj.area, 6.0)


class TestReplaceDetectLabel(unittest.TestCase):
    def setUp(self):
        self.record = khandy.label.DetectIrRecord(
            filename="",
            width=10,
            height=10,
            objects=[
                khandy.label.DetectIrObject("cat", 0, 0, 10, 10),
                khandy.label.DetectIrObject("dog", 0, 0, 10, 10),
                khandy.label.DetectIrObject("bird", 0, 0, 10, 10),
            ]
        )
        self.label_map_dict = {"cat": "feline", "dog": "canine"}
        self.label_map_callable = lambda label: self.label_map_dict.get(label)

    def test_replace_with_dict_label_map(self):
        result = khandy.label.replace_detect_label(self.record, self.label_map_dict, ignore_none=False)
        self.assertEqual(result.objects[0].label, "feline")
        self.assertEqual(result.objects[1].label, "canine")
        self.assertEqual(result.objects[2].label, "bird")  # Unmapped label remains unchanged
        self.assertEqual(self.record.objects[0].label, "cat")
        self.assertEqual(self.record.objects[1].label, "dog")
        self.assertEqual(self.record.objects[2].label, "bird")
        
    def test_replace_with_callable_label_map(self):
        result = khandy.label.replace_detect_label(self.record, self.label_map_callable, ignore_none=False)
        self.assertEqual(result.objects[0].label, "feline")
        self.assertEqual(result.objects[1].label, "canine")
        self.assertEqual(result.objects[2].label, "bird")  # Unmapped label remains unchanged
        self.assertEqual(self.record.objects[0].label, "cat")
        self.assertEqual(self.record.objects[1].label, "dog")
        self.assertEqual(self.record.objects[2].label, "bird")
        
    def test_ignore_none_true(self):
        result = khandy.label.replace_detect_label(self.record, self.label_map_dict, ignore_none=True)
        self.assertEqual(len(result.objects), 2)  # Only mapped labels are kept
        self.assertEqual(result.objects[0].label, "feline")
        self.assertEqual(result.objects[1].label, "canine")
        self.assertEqual(self.record.objects[0].label, "cat")
        self.assertEqual(self.record.objects[1].label, "dog")
        self.assertEqual(self.record.objects[2].label, "bird")
        
    def test_ignore_none_false(self):
        result = khandy.label.replace_detect_label(self.record, self.label_map_dict, ignore_none=False)
        self.assertEqual(len(result.objects), 3)  # All objects are kept
        self.assertEqual(result.objects[2].label, "bird")  # Unmapped label remains unchanged
        self.assertEqual(self.record.objects[0].label, "cat")
        self.assertEqual(self.record.objects[1].label, "dog")
        self.assertEqual(self.record.objects[2].label, "bird")
        
    def test_invalid_input_type(self):
        with self.assertRaises(TypeError):
            khandy.label.replace_detect_label("invalid_input", self.label_map_dict)


if __name__ == '__main__':
    unittest.main()
