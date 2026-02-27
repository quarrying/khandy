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


class TestToTuple(unittest.TestCase):
    def test_scalar(self):
        self.assertEqual(khandy.to_ntuple(5, 3), (5, 5, 5))
        self.assertEqual(khandy.to_ntuple('a', 2), ('a', 'a'))

    def test_sequence(self):
        self.assertEqual(khandy.to_ntuple([1, 2, 3], 3), (1, 2, 3))
        self.assertEqual(khandy.to_ntuple((4, 5), 2), (4, 5))
        self.assertEqual(khandy.to_ntuple('xy', 2), ('xy', 'xy'))  # string is not treated as sequence

    def test_invalid_length(self):
        with self.assertRaises(AssertionError):
            khandy.to_ntuple([1, 2], 3)
        with self.assertRaises(AssertionError):
            khandy.to_ntuple((1, 2, 3, 4), 3)

    def test_invalid_n(self):
        with self.assertRaises(ValueError):
            khandy.to_ntuple(1, 0)
        with self.assertRaises(ValueError):
            khandy.to_ntuple(1, -1)
        with self.assertRaises(ValueError):
            khandy.to_ntuple(1, 1.5)

    def test_to_xtuple(self):
        self.assertEqual(khandy.to_1tuple(1), (1,))
        self.assertEqual(khandy.to_2tuple(1), (1, 1))
        self.assertEqual(khandy.to_3tuple(1), (1, 1, 1))
        self.assertEqual(khandy.to_4tuple(1), (1, 1, 1, 1))


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


class TestSplitBy(unittest.TestCase):

    def test_split_by_normal(self):
        iterator = [1, 2, 3, 4, 5]
        condition = lambda index, x: x % 2 == 0  # Split on even numbers
        self.assertEqual(
            khandy.split_by(iterator, condition, mode="first"),
            [[1], [2, 3], [4, 5]],
        )
        self.assertEqual(
            khandy.split_by(iterator, condition, mode="last"),
            [[1, 2], [3, 4], [5]],
        )
        self.assertEqual(
            khandy.split_by(iterator, condition, mode="drop"),
            [[1], [3], [5]],
        )

    def test_split_by_no_matches(self):
        # Test case where no items satisfy the condition
        iterable = [1, 3, 5, 7]
        condition = lambda index, x: x % 2 == 0  # No even numbers

        self.assertEqual(khandy.split_by(iterable, condition, mode="first"), [[1, 3, 5, 7]])
        self.assertEqual(khandy.split_by(iterable, condition, mode="last"), [[1, 3, 5, 7]])
        self.assertEqual(khandy.split_by(iterable, condition, mode="drop"), [[1, 3, 5, 7]])

    def test_split_by_empty_iterable(self):
        # Test case for empty iterable
        iterable = []
        condition = lambda index, x: x > 0
        self.assertEqual(khandy.split_by(iterable, condition, mode="first"), [])
        self.assertEqual(khandy.split_by(iterable, condition, mode="last"), [])
        self.assertEqual(khandy.split_by(iterable, condition, mode="drop"), [[]])

    def test_split_by_single(self):
        # Test case where a single item satisfies the condition
        iterable = ['hello']
        condition = lambda index, x: x == 'hello'
        self.assertEqual(khandy.split_by(iterable, condition, mode="first"), [['hello']])
        self.assertEqual(khandy.split_by(iterable, condition, mode="last"), [['hello']])
        self.assertEqual(khandy.split_by(iterable, condition, mode="drop"), [[], []])

    def test_split_by_two(self):
        # Test case where two items satisfy the condition
        iterable = ['hello', 'hello']
        condition = lambda index, x: x == 'hello'
        self.assertEqual(khandy.split_by(iterable, condition, mode="first"), [['hello'], ['hello']])
        self.assertEqual(khandy.split_by(iterable, condition, mode="last"), [['hello'], ['hello']])
        self.assertEqual(khandy.split_by(iterable, condition, mode="drop"), [[], [], []])

    def test_split_by_invalid_mode(self):
        # Test case for invalid mode
        iterable = [1, 2, 3]
        condition = lambda index, x: x > 1
        with self.assertRaises(ValueError):
            khandy.split_by(iterable, condition, mode='invalid')


if __name__ == '__main__':
    unittest.main()
