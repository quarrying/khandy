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


if __name__ == '__main__':
    unittest.main()
    
    