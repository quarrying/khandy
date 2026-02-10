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


if __name__ == '__main__':
    unittest.main()

