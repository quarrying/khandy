import random
import unittest

import khandy


class TestDictUtils(unittest.TestCase):
    def test_get_dict_first_item(self):
        with self.assertRaises(TypeError):
            khandy.get_dict_first_item([])
        with self.assertRaises(ValueError):
            khandy.get_dict_first_item({}, raise_if_empty=True)
        self.assertIsNone(khandy.get_dict_first_item({}))
        self.assertEqual(khandy.get_dict_first_item({'a': 1}), ('a', 1))
        self.assertEqual(khandy.get_dict_first_item(dict([('a', 1), ('b', 2)])), ('a', 1))


class TestSampleMultidict(unittest.TestCase):
    
    def setUp(self):
        """Set up test data for the tests."""
        self.test_multidict = {
            'a': [1, 2, 3, 4],
            'b': [5, 6, 7],
            'c': [8, 9],
            'd': [10, 11, 12, 13, 14]
        }
        
    def test_num_keys_none_all_keys_returned(self):
        """Test when num_keys is None, all keys are returned."""
        result = khandy.sample_multidict(self.test_multidict, num_keys=None)
        
        # Check that all original keys are present
        self.assertEqual(set(result.keys()), set(self.test_multidict.keys()))
        # Check that values are preserved for each key
        for key in result:
            self.assertEqual(result[key], self.test_multidict[key])
    
    def test_num_keys_zero_all_keys_returned(self):
        """Test when num_keys is 0, all keys are returned (special handling)."""
        result = khandy.sample_multidict(self.test_multidict, num_keys=0)
        
        # When num_keys is 0 or negative, all keys should be returned
        self.assertEqual(set(result.keys()), set(self.test_multidict.keys()))
        for key in result:
            self.assertEqual(result[key], self.test_multidict[key])
    
    def test_num_keys_negative_all_keys_returned(self):
        """Test when num_keys is negative, all keys are returned."""
        result = khandy.sample_multidict(self.test_multidict, num_keys=-5)
        
        self.assertEqual(set(result.keys()), set(self.test_multidict.keys()))
        for key in result:
            self.assertEqual(result[key], self.test_multidict[key])
    
    def test_positive_num_keys_exact_amount(self):
        """Test when num_keys is positive, exactly that many keys are returned."""
        num_keys = 2
        result = khandy.sample_multidict(self.test_multidict, num_keys=num_keys)
        
        self.assertEqual(len(result), num_keys)
        # Ensure returned keys are subset of original keys
        self.assertTrue(set(result.keys()).issubset(set(self.test_multidict.keys())))
    
    def test_num_keys_greater_than_available(self):
        """Test when num_keys exceeds available keys, all keys are returned."""
        num_keys = 10  # More than available keys
        result = khandy.sample_multidict(self.test_multidict, num_keys=num_keys)
        
        self.assertEqual(set(result.keys()), set(self.test_multidict.keys()))
        for key in result:
            self.assertEqual(result[key], self.test_multidict[key])
    
    def test_num_per_key_none_all_values_preserved(self):
        """Test when num_per_key is None, all values for selected keys are preserved."""
        result = khandy.sample_multidict(self.test_multidict, num_keys=2, num_per_key=None)
        
        self.assertEqual(len(result), 2)
        for key in result:
            # All values for the selected key should be preserved
            self.assertEqual(result[key], self.test_multidict[key])
    
    def test_num_per_key_negative_all_values_preserved(self):
        """Test when num_per_key is None, all values for selected keys are preserved."""
        result = khandy.sample_multidict(self.test_multidict, num_keys=2, num_per_key=-1)
        
        self.assertEqual(len(result), 2)
        for key in result:
            # All values for the selected key should be preserved
            self.assertEqual(result[key], self.test_multidict[key])
            
    def test_num_per_key_positive_subset_of_values(self):
        """Test when num_per_key is positive, only that many values per key are returned."""
        num_keys = 2
        num_per_key = 2
        result = khandy.sample_multidict(self.test_multidict, num_keys=num_keys, num_per_key=num_per_key)
        
        self.assertEqual(len(result), num_keys)
        for key in result:
            self.assertEqual(len(result[key]), num_per_key)
            # Ensure returned values are subset of original values
            for value in result[key]:
                self.assertIn(value, self.test_multidict[key])
    
    def test_num_per_key_greater_than_available(self):
        """Test when num_per_key exceeds available values, all values are returned for that key."""
        num_keys = 2
        num_per_key = 10  # More than available in any single key
        result = khandy.sample_multidict(self.test_multidict, num_keys=num_keys, num_per_key=num_per_key)
        
        self.assertEqual(len(result), num_keys)
        for key in result:
            # Should return all original values since num_per_key > len(original_values)
            self.assertEqual(result[key], self.test_multidict[key])
    
    def test_empty_multidict(self):
        """Test behavior with empty multidict."""
        empty_multidict = {}
        result = khandy.sample_multidict(empty_multidict, num_keys=2)
        
        self.assertEqual(result, {})
    
    def test_single_key_multidict(self):
        """Test with multidict containing only one key."""
        single_key_multidict = {'a': [1, 2, 3]}
        result = khandy.sample_multidict(single_key_multidict, num_keys=1)
        
        self.assertEqual(len(result), 1)
        self.assertIn('a', result)
        self.assertEqual(result['a'], [1, 2, 3])
    
    def test_deterministic_behavior_with_seed(self):
        """Test that results are reproducible with fixed seed."""
        random.seed(42)
        result1 = khandy.sample_multidict(self.test_multidict, num_keys=2, num_per_key=2)
        
        random.seed(42)
        result2 = khandy.sample_multidict(self.test_multidict, num_keys=2, num_per_key=2)
        
        # Results should be identical with same seed
        self.assertEqual(result1, result2)


class TestFilterMultidictByNumber(unittest.TestCase):
    
    def setUp(self):
        """Set up test data"""
        self.test_dict = {
            'a': [1, 2, 3],      # length 3
            'b': [4, 5],         # length 2
            'c': [6],            # length 1
            'd': [7, 8, 9, 10],  # length 4
            'e': []              # length 0
        }
    
    def test_no_bounds_returns_copy(self):
        """Test when both lower and upper bounds are None"""
        result = khandy.filter_multidict_by_number(self.test_dict)
        expected = self.test_dict.copy()
        self.assertEqual(result, expected)
        # Ensure it's a copy and not the same object
        self.assertIsNot(result, self.test_dict)
    
    def test_only_upper_bound(self):
        """Test with only upper bound specified"""
        result = khandy.filter_multidict_by_number(self.test_dict, upper=2)
        expected = {
            'b': [4, 5],  # length 2
            'c': [6],     # length 1
            'e': []       # length 0
        }
        self.assertEqual(result, expected)
    
    def test_only_lower_bound(self):
        """Test with only lower bound specified"""
        result = khandy.filter_multidict_by_number(self.test_dict, lower=3)
        expected = {
            'a': [1, 2, 3],      # length 3
            'd': [7, 8, 9, 10]   # length 4
        }
        self.assertEqual(result, expected)
    
    def test_both_bounds(self):
        """Test with both lower and upper bounds specified"""
        result = khandy.filter_multidict_by_number(self.test_dict, lower=2, upper=3)
        expected = {
            'a': [1, 2, 3],  # length 3
            'b': [4, 5]      # length 2
        }
        self.assertEqual(result, expected)
    
    def test_empty_dict(self):
        """Test with empty dictionary"""
        result = khandy.filter_multidict_by_number({}, lower=1, upper=5)
        self.assertEqual(result, {})
        
        result = khandy.filter_multidict_by_number({})
        self.assertEqual(result, {})
    
    def test_dict_with_empty_lists(self):
        """Test dictionary containing empty lists"""
        empty_dict = {'empty': [], 'non_empty': [1, 2]}
        
        result = khandy.filter_multidict_by_number(empty_dict, upper=1)
        expected = {'empty': []}
        self.assertEqual(result, expected)
        
        result = khandy.filter_multidict_by_number(empty_dict, lower=1)
        expected = {'non_empty': [1, 2]}
        self.assertEqual(result, expected)
    
    def test_boundary_conditions(self):
        """Test boundary conditions"""
        # When upper bound equals exact length
        result = khandy.filter_multidict_by_number(self.test_dict, upper=3)
        expected = {
            'a': [1, 2, 3],  # length 3 (equals upper bound)
            'b': [4, 5],     # length 2
            'c': [6],        # length 1
            'e': []          # length 0
        }
        self.assertEqual(result, expected)
        
        # When lower bound equals exact length
        result = khandy.filter_multidict_by_number(self.test_dict, lower=2)
        expected = {
            'a': [1, 2, 3],      # length 3
            'b': [4, 5],         # length 2 (equals lower bound)
            'd': [7, 8, 9, 10]   # length 4
        }
        self.assertEqual(result, expected)
    
    def test_value_error_when_lower_greater_than_upper(self):
        """Test that ValueError is raised when lower > upper"""
        with self.assertRaises(ValueError) as context:
            khandy.filter_multidict_by_number(self.test_dict, lower=5, upper=3)
        self.assertIn('lower must not be greater than upper', str(context.exception))
    
    def test_single_element_bounds(self):
        """Test with bounds that result in single elements"""
        result = khandy.filter_multidict_by_number(self.test_dict, lower=1, upper=1)
        expected = {'c': [6]}  # Only 'c' has length exactly 1
        self.assertEqual(result, expected)


class TestRekeyMultidict(unittest.TestCase):
    
    def test_valid_mapping_all_keys_present(self):
        """Test when all keys in multidict_obj exist in key_map with raise_if_key_error=True"""
        multidict_obj = {'a': [1, 2], 'b': [3, 4]}
        key_map = {'a': 'x', 'b': 'y'}
        expected = {'x': [1, 2], 'y': [3, 4]}
        
        result = khandy.rekey_multidict(multidict_obj, key_map, raise_if_key_error=True)
        self.assertEqual(result, expected)
    
    def test_valid_mapping_with_raise_false_missing_keys(self):
        """Test when some keys in multidict_obj don't exist in key_map with raise_if_key_error=False"""
        multidict_obj = {'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}
        key_map = {'a': 'x', 'b': 'y'}  # 'c' is missing from key_map
        expected = {'x': [1, 2], 'y': [3, 4], 'c': [5, 6]}  # 'c' should remain unchanged
        
        result = khandy.rekey_multidict(multidict_obj, key_map, raise_if_key_error=False)
        self.assertEqual(result, expected)
    
    def test_missing_key_raises_error_when_flag_true(self):
        """Test that KeyError is raised when a key is missing and raise_if_key_error=True"""
        multidict_obj = {'a': [1, 2], 'b': [3, 4]}
        key_map = {'a': 'x'}  # 'b' is missing from key_map
        
        with self.assertRaises(KeyError):
            khandy.rekey_multidict(multidict_obj, key_map, raise_if_key_error=True)
    
    def test_empty_multidict_returns_empty(self):
        """Test with empty multidict_obj"""
        multidict_obj = {}
        key_map = {'a': 'x', 'b': 'y'}
        expected = {}
        
        result = khandy.rekey_multidict(multidict_obj, key_map, raise_if_key_error=True)
        self.assertEqual(result, expected)
    
    def test_empty_key_map_with_raise_false(self):
        """Test with empty key_map and raise_if_key_error=False (should return original keys)"""
        multidict_obj = {'a': [1, 2], 'b': [3, 4]}
        key_map = {}  # Empty key_map
        expected = {'a': [1, 2], 'b': [3, 4]}  # Original keys preserved
        
        result = khandy.rekey_multidict(multidict_obj, key_map, raise_if_key_error=False)
        self.assertEqual(result, expected)
    
    def test_empty_key_map_with_raise_true(self):
        """Test with empty key_map and raise_if_key_error=True (should raise KeyError)"""
        multidict_obj = {'a': [1, 2], 'b': [3, 4]}
        key_map = {}  # Empty key_map
        
        with self.assertRaises(KeyError):
            khandy.rekey_multidict(multidict_obj, key_map, raise_if_key_error=True)
    
    def test_complex_data_types_as_values(self):
        """Test with complex data types as values in the lists"""
        multidict_obj = {
            'a': [{'nested': 'dict'}, [1, 2, 3]], 
            'b': ['string', 42]
        }
        key_map = {'a': 'alpha', 'b': 'beta'}
        expected = {
            'alpha': [{'nested': 'dict'}, [1, 2, 3]], 
            'beta': ['string', 42]
        }
        
        result = khandy.rekey_multidict(multidict_obj, key_map, raise_if_key_error=True)
        self.assertEqual(result, expected)
    
    def test_non_string_keys_and_values(self):
        """Test with non-string keys and various value types"""
        multidict_obj = {1: ['a', 'b'], 2: [True, False]}
        key_map = {1: 'one', 2: 'two'}
        expected = {'one': ['a', 'b'], 'two': [True, False]}
        
        result = khandy.rekey_multidict(multidict_obj, key_map, raise_if_key_error=True)
        self.assertEqual(result, expected)
    
    def test_mixed_type_keys(self):
        """Test with mixed type keys in both multidict_obj and key_map"""
        multidict_obj = {1: [10, 20], 'str_key': [30, 40]}
        key_map = {1: 'number_one', 'str_key': 'renamed_str_key'}
        expected = {'number_one': [10, 20], 'renamed_str_key': [30, 40]}
        
        result = khandy.rekey_multidict(multidict_obj, key_map, raise_if_key_error=True)
        self.assertEqual(result, expected)
    
    def test_duplicate_values_preserved(self):
        """Test that duplicate values in lists are preserved after remapping"""
        multidict_obj = {'old_key': [1, 1, 2, 2, 3]}
        key_map = {'old_key': 'new_key'}
        expected = {'new_key': [1, 1, 2, 2, 3]}
        
        result = khandy.rekey_multidict(multidict_obj, key_map, raise_if_key_error=True)
        self.assertEqual(result, expected)
    
    def test_original_object_not_modified(self):
        """Test that the original multidict_obj is not modified during remapping"""
        original_multidict = {'a': [1, 2], 'b': [3, 4]}
        original_copy = {'a': [1, 2], 'b': [3, 4]}  # Make a copy to compare
        key_map = {'a': 'x', 'b': 'y'}
        
        result = khandy.rekey_multidict(original_multidict, key_map, raise_if_key_error=True)
        
        # Original object should remain unchanged
        self.assertEqual(original_multidict, original_copy)
        # Result should have remapped keys
        self.assertEqual(result, {'x': [1, 2], 'y': [3, 4]})
    
    def test_multiple_values_in_list(self):
        """Test with multiple different types of values in the same list"""
        multidict_obj = {'key1': [1, 'string', None, [1, 2], {'a': 1}]}
        key_map = {'key1': 'mapped_key1'}
        expected = {'mapped_key1': [1, 'string', None, [1, 2], {'a': 1}]}
        
        result = khandy.rekey_multidict(multidict_obj, key_map, raise_if_key_error=True)
        self.assertEqual(result, expected)

    def test_multiple_keys_mapping_to_same_new_key(self):
        """Test when multiple old keys map to the same new key (values should be combined)"""
        multidict_obj = {'a': [1, 2], 'b': [3, 4]}
        key_map = {'a': 'x', 'b': 'x'}
        expected = {'x': [1, 2, 3, 4]}  # Values from both 'a' and 'b' should be combined under 'x'
        result = khandy.rekey_multidict(multidict_obj, key_map, raise_if_key_error=True)
        self.assertEqual(result, expected)
        

class TestCreateMultidict(unittest.TestCase):

    def test_basic_grouping(self):
        result = khandy.create_multidict(['a', 'b', 'a'], [1, 2, 3])
        self.assertEqual(result, {'a': [1, 3], 'b': [2]})

    def test_value_container_is_list(self):
        result = khandy.create_multidict(['a', 'a'], [1, 2])
        for value in result.values():
            self.assertIsInstance(value, list)

    def test_duplicate_values_preserved_per_key(self):
        result = khandy.create_multidict(['a', 'a', 'a'], [1, 1, 1])
        self.assertEqual(result, {'a': [1, 1, 1]})

    def test_iterables_supported(self):
        # Any Iterable (tuple, generator, ...) is accepted; not just list.
        result = khandy.create_multidict(
            ('a', 'b', 'a'),
            (x for x in (1, 2, 3)),
        )
        self.assertEqual(result, {'a': [1, 3], 'b': [2]})

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            khandy.create_multidict(['a', 'b'], [1])
        self.assertIn('must have the same length', str(ctx.exception))

    def test_empty_inputs(self):
        self.assertEqual(khandy.create_multidict([], []), {})


class TestCreateMultidictUnique(unittest.TestCase):

    def test_basic_grouping(self):
        result = khandy.create_multidict_unique(['a', 'b', 'a'], [1, 2, 3])
        self.assertEqual(result, {'a': {1, 3}, 'b': {2}})

    def test_value_container_is_set(self):
        result = khandy.create_multidict_unique(['a', 'a'], [1, 2])
        for value in result.values():
            self.assertIsInstance(value, set)

    def test_duplicate_values_deduped_per_key(self):
        result = khandy.create_multidict_unique(['a', 'a', 'a'], [1, 1, 1])
        self.assertEqual(result, {'a': {1}})

    def test_unhashable_values_raise(self):
        with self.assertRaises(TypeError):
            khandy.create_multidict_unique(['a', 'b'], [[1, 2], [3, 4]])

    def test_iterables_supported(self):
        # Any Iterable (tuple, generator, ...) is accepted; not just list.
        result = khandy.create_multidict_unique(
            ('a', 'b', 'a'),
            (x for x in (1, 2, 3)),
        )
        self.assertEqual(result, {'a': {1, 3}, 'b': {2}})

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            khandy.create_multidict_unique(['a', 'b'], [1])
        self.assertIn('must have the same length', str(ctx.exception))

    def test_empty_inputs(self):
        self.assertEqual(khandy.create_multidict_unique([], []), {})


class TestConvertMultidictToRecords(unittest.TestCase):
    
    def test_basic_functionality_value_first(self):
        """Test basic conversion with value_first=True"""
        multidict = {'a': [1, 2], 'b': [3]}
        result = khandy.convert_multidict_to_records(multidict, value_first=True)
        expected = ['1,a', '2,a', '3,b']
        self.assertEqual(result, expected)
    
    def test_basic_functionality_key_first(self):
        """Test basic conversion with value_first=False"""
        multidict = {'a': [1, 2], 'b': [3]}
        result = khandy.convert_multidict_to_records(multidict, value_first=False)
        expected = ['a,1', 'a,2', 'b,3']
        self.assertEqual(result, expected)
    
    def test_empty_multidict(self):
        """Test with empty multidict"""
        multidict = {}
        result = khandy.convert_multidict_to_records(multidict)
        expected = []
        self.assertEqual(result, expected)
    
    def test_multidict_with_empty_lists(self):
        """Test multidict containing empty lists"""
        multidict = {'a': [], 'b': [1, 2]}
        result = khandy.convert_multidict_to_records(multidict)
        expected = ['1,b', '2,b']
        self.assertEqual(result, expected)
    
    def test_all_empty_lists(self):
        """Test multidict where all values are empty lists"""
        multidict = {'a': [], 'b': []}
        result = khandy.convert_multidict_to_records(multidict)
        expected = []
        self.assertEqual(result, expected)
    
    def test_mixed_data_types(self):
        """Test with mixed data types as keys and values"""
        multidict = {1: ['x', 'y'], 'str_key': [99, None]}
        result = khandy.convert_multidict_to_records(multidict, value_first=True)
        expected = ['x,1', 'y,1', '99,str_key', 'None,str_key']
        self.assertEqual(result, expected)
    
    def test_single_value_per_key(self):
        """Test with single value per key (each list has only one element)"""
        multidict = {'p': [100], 'q': [200]}
        result = khandy.convert_multidict_to_records(multidict, value_first=True)
        expected = ['100,p', '200,q']
        self.assertEqual(result, expected)
    
    def test_complex_values_and_keys(self):
        """Test with complex objects as keys and values"""
        class CustomObj:
            def __init__(self, name):
                self.name = name
            def __str__(self):
                return f"CustomObj({self.name})"
        
        obj1 = CustomObj("first")
        obj2 = CustomObj("second")
        multidict = {obj1: [42], 'string_key': [obj2]}
        result = khandy.convert_multidict_to_records(multidict, value_first=True)
        expected = [f'42,{obj1}', f'{obj2},string_key']
        self.assertEqual(result, expected)


class TestSortMultidictByNumber(unittest.TestCase):

    def setUp(self):
        self.test_multidict = {
            'a': [1, 2, 3, 4],          # length 4
            'b': [5, 6, 7],             # length 3
            'c': [8, 9],                # length 2
            'd': [10, 11, 12, 13, 14],  # length 5
            'e': [15],                  # length 1
        }

    def test_reverse_true_descending_default(self):
        """Default reverse=True returns keys ordered by descending length."""
        result = khandy.sort_multidict_by_number(self.test_multidict)
        self.assertEqual(list(result.keys()), ['d', 'a', 'b', 'c', 'e'])

    def test_reverse_false_ascending(self):
        """reverse=False returns keys ordered by ascending length."""
        result = khandy.sort_multidict_by_number(self.test_multidict, reverse=False)
        self.assertEqual(list(result.keys()), ['e', 'c', 'b', 'a', 'd'])

    def test_num_keys_none_keeps_all(self):
        """num_keys_to_keep=None retains all keys (sorted)."""
        result = khandy.sort_multidict_by_number(self.test_multidict, num_keys_to_keep=None)
        self.assertEqual(set(result.keys()), set(self.test_multidict.keys()))
        self.assertEqual(len(result), len(self.test_multidict))

    def test_num_keys_positive_truncates(self):
        """num_keys_to_keep=N keeps exactly N keys (in sort order)."""
        result = khandy.sort_multidict_by_number(self.test_multidict, num_keys_to_keep=2)
        # Default reverse=True: top-2 by length are 'd' (5) then 'a' (4).
        self.assertEqual(list(result.keys()), ['d', 'a'])
        self.assertEqual(len(result), 2)

    def test_num_keys_exceeds_len_keeps_all(self):
        """num_keys_to_keep greater than len is clamped to len."""
        result = khandy.sort_multidict_by_number(self.test_multidict, num_keys_to_keep=100)
        self.assertEqual(set(result.keys()), set(self.test_multidict.keys()))
        self.assertEqual(len(result), len(self.test_multidict))

    def test_num_keys_zero_or_negative_keeps_all(self):
        """num_keys_to_keep=0 or negative is treated as 'keep all' (matches sample_multidict)."""
        for bad_value in (0, -1, -100):
            with self.subTest(num_keys_to_keep=bad_value):
                result = khandy.sort_multidict_by_number(
                    self.test_multidict, num_keys_to_keep=bad_value,
                )
                self.assertEqual(set(result.keys()), set(self.test_multidict.keys()))
                self.assertEqual(len(result), len(self.test_multidict))

    def test_empty_multidict(self):
        """Empty input returns empty dict."""
        result = khandy.sort_multidict_by_number({})
        self.assertEqual(result, {})

        result = khandy.sort_multidict_by_number({}, num_keys_to_keep=3)
        self.assertEqual(result, {})

    def test_lists_are_same_objects(self):
        """Returned lists must be the same objects as in the input (no copies)."""
        result = khandy.sort_multidict_by_number(self.test_multidict, num_keys_to_keep=2)
        for key in result:
            self.assertIs(result[key], self.test_multidict[key])

    def test_tie_break_is_stable(self):
        """Keys with equal-length values keep their original insertion order."""
        # Insertion order: 'a' (2), 'b' (2), 'c' (2) all length 2.
        tied = {'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}
        result = khandy.sort_multidict_by_number(tied, reverse=False)
        self.assertEqual(list(result.keys()), ['a', 'b', 'c'])

        result_desc = khandy.sort_multidict_by_number(tied, reverse=True)
        self.assertEqual(list(result_desc.keys()), ['a', 'b', 'c'])

    def test_non_dict_raises_type_error(self):
        """Non-dict input raises TypeError with a descriptive message."""
        with self.assertRaises(TypeError) as ctx:
            khandy.sort_multidict_by_number([])  # type: ignore[arg-type]
        self.assertIn('multidict_obj should be a dict', str(ctx.exception))

        with self.assertRaises(TypeError):
            khandy.sort_multidict_by_number(None)  # type: ignore[arg-type]

    def test_default_is_descending_docstring_regression(self):
        """Locks in the corrected docstring semantics: default returns descending."""
        # With the old (buggy) docstring saying 'ascending when True', a caller
        # expecting descending would have called with reverse=False. Verify the
        # function still behaves as documented in the new docstring.
        multidict = {'x': [1, 2, 3], 'y': [1], 'z': [1, 2]}
        result = khandy.sort_multidict_by_number(multidict)
        # 'x' has length 3, 'z' has length 2, 'y' has length 1.
        self.assertEqual(list(result.keys()), ['x', 'z', 'y'])


if __name__ == '__main__':
    unittest.main()

