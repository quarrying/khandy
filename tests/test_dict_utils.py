import random
import unittest
from collections import OrderedDict

import khandy


class TestDictUtils(unittest.TestCase):
    def test_get_dict_first_item(self):
        with self.assertRaises(TypeError):
            khandy.get_dict_first_item([])
        with self.assertRaises(ValueError):
            khandy.get_dict_first_item({}, raise_if_empty=True)
        self.assertIsNone(khandy.get_dict_first_item({}))
        self.assertEqual(khandy.get_dict_first_item({'a': 1}), ('a', 1))
        self.assertEqual(khandy.get_dict_first_item(OrderedDict([('a', 1), ('b', 2)])), ('a', 1))


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
    
    def test_assertion_error_when_lower_greater_than_upper(self):
        """Test that assertion error is raised when lower > upper"""
        with self.assertRaises(AssertionError) as context:
            khandy.filter_multidict_by_number(self.test_dict, lower=5, upper=3)
        self.assertIn('lower must not be greater than upper', str(context.exception))
    
    def test_single_element_bounds(self):
        """Test with bounds that result in single elements"""
        result = khandy.filter_multidict_by_number(self.test_dict, lower=1, upper=1)
        expected = {'c': [6]}  # Only 'c' has length exactly 1
        self.assertEqual(result, expected)


class TestRemapMultidictKeys(unittest.TestCase):
    
    def test_valid_mapping_all_keys_present(self):
        """Test when all keys in multidict_obj exist in key_map with raise_if_key_error=True"""
        multidict_obj = {'a': [1, 2], 'b': [3, 4]}
        key_map = {'a': 'x', 'b': 'y'}
        expected = {'x': [1, 2], 'y': [3, 4]}
        
        result = khandy.remap_multidict_keys(multidict_obj, key_map, raise_if_key_error=True)
        self.assertEqual(result, expected)
    
    def test_valid_mapping_with_raise_false_missing_keys(self):
        """Test when some keys in multidict_obj don't exist in key_map with raise_if_key_error=False"""
        multidict_obj = {'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}
        key_map = {'a': 'x', 'b': 'y'}  # 'c' is missing from key_map
        expected = {'x': [1, 2], 'y': [3, 4], 'c': [5, 6]}  # 'c' should remain unchanged
        
        result = khandy.remap_multidict_keys(multidict_obj, key_map, raise_if_key_error=False)
        self.assertEqual(result, expected)
    
    def test_missing_key_raises_error_when_flag_true(self):
        """Test that KeyError is raised when a key is missing and raise_if_key_error=True"""
        multidict_obj = {'a': [1, 2], 'b': [3, 4]}
        key_map = {'a': 'x'}  # 'b' is missing from key_map
        
        with self.assertRaises(KeyError):
            khandy.remap_multidict_keys(multidict_obj, key_map, raise_if_key_error=True)
    
    def test_empty_multidict_returns_empty(self):
        """Test with empty multidict_obj"""
        multidict_obj = {}
        key_map = {'a': 'x', 'b': 'y'}
        expected = {}
        
        result = khandy.remap_multidict_keys(multidict_obj, key_map, raise_if_key_error=True)
        self.assertEqual(result, expected)
    
    def test_empty_key_map_with_raise_false(self):
        """Test with empty key_map and raise_if_key_error=False (should return original keys)"""
        multidict_obj = {'a': [1, 2], 'b': [3, 4]}
        key_map = {}  # Empty key_map
        expected = {'a': [1, 2], 'b': [3, 4]}  # Original keys preserved
        
        result = khandy.remap_multidict_keys(multidict_obj, key_map, raise_if_key_error=False)
        self.assertEqual(result, expected)
    
    def test_empty_key_map_with_raise_true(self):
        """Test with empty key_map and raise_if_key_error=True (should raise KeyError)"""
        multidict_obj = {'a': [1, 2], 'b': [3, 4]}
        key_map = {}  # Empty key_map
        
        with self.assertRaises(KeyError):
            khandy.remap_multidict_keys(multidict_obj, key_map, raise_if_key_error=True)
    
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
        
        result = khandy.remap_multidict_keys(multidict_obj, key_map, raise_if_key_error=True)
        self.assertEqual(result, expected)
    
    def test_non_string_keys_and_values(self):
        """Test with non-string keys and various value types"""
        multidict_obj = {1: ['a', 'b'], 2: [True, False]}
        key_map = {1: 'one', 2: 'two'}
        expected = {'one': ['a', 'b'], 'two': [True, False]}
        
        result = khandy.remap_multidict_keys(multidict_obj, key_map, raise_if_key_error=True)
        self.assertEqual(result, expected)
    
    def test_mixed_type_keys(self):
        """Test with mixed type keys in both multidict_obj and key_map"""
        multidict_obj = {1: [10, 20], 'str_key': [30, 40]}
        key_map = {1: 'number_one', 'str_key': 'renamed_str_key'}
        expected = {'number_one': [10, 20], 'renamed_str_key': [30, 40]}
        
        result = khandy.remap_multidict_keys(multidict_obj, key_map, raise_if_key_error=True)
        self.assertEqual(result, expected)
    
    def test_duplicate_values_preserved(self):
        """Test that duplicate values in lists are preserved after remapping"""
        multidict_obj = {'old_key': [1, 1, 2, 2, 3]}
        key_map = {'old_key': 'new_key'}
        expected = {'new_key': [1, 1, 2, 2, 3]}
        
        result = khandy.remap_multidict_keys(multidict_obj, key_map, raise_if_key_error=True)
        self.assertEqual(result, expected)
    
    def test_original_object_not_modified(self):
        """Test that the original multidict_obj is not modified during remapping"""
        original_multidict = {'a': [1, 2], 'b': [3, 4]}
        original_copy = {'a': [1, 2], 'b': [3, 4]}  # Make a copy to compare
        key_map = {'a': 'x', 'b': 'y'}
        
        result = khandy.remap_multidict_keys(original_multidict, key_map, raise_if_key_error=True)
        
        # Original object should remain unchanged
        self.assertEqual(original_multidict, original_copy)
        # Result should have remapped keys
        self.assertEqual(result, {'x': [1, 2], 'y': [3, 4]})
    
    def test_multiple_values_in_list(self):
        """Test with multiple different types of values in the same list"""
        multidict_obj = {'key1': [1, 'string', None, [1, 2], {'a': 1}]}
        key_map = {'key1': 'mapped_key1'}
        expected = {'mapped_key1': [1, 'string', None, [1, 2], {'a': 1}]}
        
        result = khandy.remap_multidict_keys(multidict_obj, key_map, raise_if_key_error=True)
        self.assertEqual(result, expected)


class TestConvertMultidictToRecords(unittest.TestCase):
    
    def test_basic_functionality_value_first(self):
        """Test basic conversion with value_first=True"""
        multidict = {'a': [1, 2], 'b': [3]}
        key_map = None
        result = khandy.convert_multidict_to_records(multidict, key_map, value_first=True)
        expected = ['1,a', '2,a', '3,b']
        self.assertEqual(result, expected)
    
    def test_basic_functionality_key_first(self):
        """Test basic conversion with value_first=False"""
        multidict = {'a': [1, 2], 'b': [3]}
        key_map = None
        result = khandy.convert_multidict_to_records(multidict, key_map, value_first=False)
        expected = ['a,1', 'a,2', 'b,3']
        self.assertEqual(result, expected)
    
    def test_with_key_mapping(self):
        """Test conversion with key mapping applied"""
        multidict = {'old_a': [1, 2], 'old_b': [3]}
        key_map = {'old_a': 'new_a', 'old_b': 'new_b'}
        result = khandy.convert_multidict_to_records(multidict, key_map, value_first=True)
        expected = ['1,new_a', '2,new_a', '3,new_b']
        self.assertEqual(result, expected)
    
    def test_empty_multidict(self):
        """Test with empty multidict"""
        multidict = {}
        key_map = None
        result = khandy.convert_multidict_to_records(multidict, key_map)
        expected = []
        self.assertEqual(result, expected)
    
    def test_multidict_with_empty_lists(self):
        """Test multidict containing empty lists"""
        multidict = {'a': [], 'b': [1, 2]}
        key_map = None
        result = khandy.convert_multidict_to_records(multidict, key_map)
        expected = ['1,b', '2,b']
        self.assertEqual(result, expected)
    
    def test_all_empty_lists(self):
        """Test multidict where all values are empty lists"""
        multidict = {'a': [], 'b': []}
        key_map = None
        result = khandy.convert_multidict_to_records(multidict, key_map)
        expected = []
        self.assertEqual(result, expected)
    
    def test_mixed_data_types(self):
        """Test with mixed data types as keys and values"""
        multidict = {1: ['x', 'y'], 'str_key': [99, None]}
        key_map = None
        result = khandy.convert_multidict_to_records(multidict, key_map, value_first=True)
        expected = ['x,1', 'y,1', '99,str_key', 'None,str_key']
        self.assertEqual(result, expected)
    
    def test_key_mapping_with_missing_keys_raise_error(self):
        """Test key mapping when some keys are missing and raise_if_key_error=True"""
        multidict = {'a': [1, 2], 'c': [3]}  # 'c' not in key_map
        key_map = {'a': 'mapped_a', 'b': 'mapped_b'}  # 'c' missing from key_map
        with self.assertRaises(KeyError):
            khandy.convert_multidict_to_records(multidict, key_map, raise_if_key_error=True)
    
    def test_key_mapping_with_missing_keys_no_error(self):
        """Test key mapping when some keys are missing and raise_if_key_error=False"""
        multidict = {'a': [1, 2], 'c': [3]}  # 'c' not in key_map
        key_map = {'a': 'mapped_a', 'b': 'mapped_b'}  # 'c' missing from key_map
        result = khandy.convert_multidict_to_records(multidict, key_map, raise_if_key_error=False, value_first=True)
        # 'a' gets mapped, 'c' stays as original
        expected = ['1,mapped_a', '2,mapped_a', '3,c']
        self.assertEqual(result, expected)
    
    def test_none_key_map_same_as_original(self):
        """Test that key_map=None produces same result as identity mapping"""
        multidict = {'x': [10], 'y': [20, 30]}
        result_with_none = khandy.convert_multidict_to_records(multidict, None)
        # Identity mapping should produce same result
        identity_map = {'x': 'x', 'y': 'y'}
        result_with_identity = khandy.convert_multidict_to_records(multidict, identity_map)
        self.assertEqual(result_with_none, result_with_identity)
    
    def test_single_value_per_key(self):
        """Test with single value per key (each list has only one element)"""
        multidict = {'p': [100], 'q': [200]}
        key_map = None
        result = khandy.convert_multidict_to_records(multidict, key_map, value_first=True)
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
        key_map = None
        result = khandy.convert_multidict_to_records(multidict, key_map, value_first=True)
        expected = [f'42,{obj1}', f'{obj2},string_key']
        self.assertEqual(result, expected)

if __name__ == '__main__':
    unittest.main()

