import collections
import unittest

import khandy


def concat_list_alternate(input_data):
    return sum(input_data, [])


class TestConcatList(unittest.TestCase):
    def test_normal_case(self):
        input_data = [[1, 2], [3, 4], [5]]
        expected_output = [1, 2, 3, 4, 5]
        self.assertEqual(khandy.concat_list(input_data), expected_output)

    def test_alternate_implement(self):
        input_data = [[1, 2], [3, 4], [5]]
        self.assertEqual(khandy.concat_list(input_data), concat_list_alternate(input_data))

        input_data = [(1, 2), {3, 4}, '56']
        with self.assertRaises(TypeError):
            concat_list_alternate(input_data)
        
    def test_empty_list(self):
        input_data = []
        expected_output = []
        self.assertEqual(khandy.concat_list(input_data), expected_output)

    def test_nested_empty_lists(self):
        input_data = [[], [], []]
        expected_output = []
        self.assertEqual(khandy.concat_list(input_data), expected_output)

    def test_non_list_iterables(self):
        input_data = [(1, 2), {3, 4}, '56']
        expected_output = [1, 2, 3, 4, '5', '6']
        self.assertEqual(khandy.concat_list(input_data), expected_output)
        self.assertEqual(khandy.concat_list('str'), ['s', 't', 'r'])
        self.assertEqual(khandy.concat_list(['s', 't', 'r']), ['s', 't', 'r'])
        
    def test_type_error(self):
        with self.assertRaises(TypeError):
            khandy.concat_list([1, 2, 3])


class TestIsSeqOf(unittest.TestCase):
    def test_valid_sequences(self):
        self.assertTrue(khandy.is_seq_of([1, 2, 3], int))
        # self.assertTrue(khandy.is_seq_of([1, 2, 3], float))
        self.assertTrue(khandy.is_seq_of(['a', 'b', 'c'], str))
        self.assertTrue(khandy.is_seq_of([[1], [2], [3]], list))

    def test_invalid_sequences(self):
        self.assertFalse(khandy.is_seq_of([1, 'b', 3], int))
        self.assertFalse(khandy.is_seq_of(['a', 2, 'c'], str))
        self.assertFalse(khandy.is_seq_of([[1], 'b', [3]], list))

    def test_empty_sequence(self):
        self.assertTrue(khandy.is_seq_of([], int))
        self.assertTrue(khandy.is_seq_of([], str))
        self.assertTrue(khandy.is_seq_of([], list))

    def test_item_type_as_tuple(self):
        self.assertTrue(khandy.is_seq_of([1, 'b', 3], (int, str)))
        self.assertTrue(khandy.is_seq_of([[1], 'b', [3]], (list, str)))
        
    def test_seq_type_as_tuple(self):
        self.assertTrue(khandy.is_seq_of([1, 2, 3], int, seq_type=list))
        self.assertTrue(khandy.is_seq_of([1, 2, 3], int, seq_type=collections.abc.Sequence))
        self.assertTrue(khandy.is_seq_of([1, 2, 3], int, seq_type=(list, tuple)))
        self.assertTrue(khandy.is_seq_of([1, 2, 3], int, seq_type=(list, collections.abc.Sequence)))
        self.assertFalse(khandy.is_seq_of([1, 2, 3], int, seq_type=tuple))
        with self.assertRaises(TypeError):
            khandy.is_seq_of([1, 2, 3], int, seq_type=[list, tuple])
 
 
if __name__ == '__main__':
    unittest.main()