from collections import OrderedDict

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
        self.assertEqual(khandy.get_dict_first_item(OrderedDict([('a', 1), ('b', 2)])), ('a', 1))


if __name__ == '__main__':
    unittest.main()

