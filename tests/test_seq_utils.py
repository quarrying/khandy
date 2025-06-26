import collections
import unittest

from khandy.seq_utils import is_seq_of


class TestIsSeqOf(unittest.TestCase):
    def test_valid_sequences(self):
        self.assertTrue(is_seq_of([1, 2, 3], int))
        # self.assertTrue(is_seq_of([1, 2, 3], float))
        self.assertTrue(is_seq_of(['a', 'b', 'c'], str))
        self.assertTrue(is_seq_of([[1], [2], [3]], list))

    def test_invalid_sequences(self):
        self.assertFalse(is_seq_of([1, 'b', 3], int))
        self.assertFalse(is_seq_of(['a', 2, 'c'], str))
        self.assertFalse(is_seq_of([[1], 'b', [3]], list))

    def test_empty_sequence(self):
        self.assertTrue(is_seq_of([], int))
        self.assertTrue(is_seq_of([], str))
        self.assertTrue(is_seq_of([], list))

    def test_item_type_as_tuple(self):
        self.assertTrue(is_seq_of([1, 'b', 3], (int, str)))
        self.assertTrue(is_seq_of([[1], 'b', [3]], (list, str)))
        
    def test_seq_type_as_tuple(self):
        self.assertTrue(is_seq_of([1, 2, 3], int, seq_type=list))
        self.assertTrue(is_seq_of([1, 2, 3], int, seq_type=collections.abc.Sequence))
        self.assertTrue(is_seq_of([1, 2, 3], int, seq_type=(list, tuple)))
        self.assertTrue(is_seq_of([1, 2, 3], int, seq_type=(list, collections.abc.Sequence)))
        self.assertFalse(is_seq_of([1, 2, 3], int, seq_type=tuple))
        with self.assertRaises(TypeError):
            is_seq_of([1, 2, 3], int, seq_type=[list, tuple])
 
if __name__ == '__main__':
    unittest.main()