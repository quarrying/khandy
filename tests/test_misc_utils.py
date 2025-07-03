import unittest
import khandy


class TestIsNumberBetween(unittest.TestCase):
    def test_basic(self):
        # left close right open
        self.assertTrue(khandy.is_number_between(5, 1, 10))
        self.assertTrue(khandy.is_number_between(1, 1, 10))
        self.assertFalse(khandy.is_number_between(10, 1, 10))
        self.assertFalse(khandy.is_number_between(0, 1, 10))
        self.assertFalse(khandy.is_number_between(11, 1, 10))

    def test_open_closed(self):
        # left open right close
        self.assertTrue(khandy.is_number_between(10, 1, 10, lower_close=False, upper_close=True))
        self.assertFalse(khandy.is_number_between(1, 1, 10, lower_close=False, upper_close=True))
        # left open right open
        self.assertFalse(khandy.is_number_between(1, 1, 10, lower_close=False, upper_close=False))
        self.assertFalse(khandy.is_number_between(10, 1, 10, lower_close=False, upper_close=False))
        self.assertTrue(khandy.is_number_between(5, 1, 10, lower_close=False, upper_close=False))

    def test_none_bounds(self):
        # no lower bound
        self.assertTrue(khandy.is_number_between(-100, None, 10))
        self.assertFalse(khandy.is_number_between(11, None, 10))
        # no upper bound
        self.assertTrue(khandy.is_number_between(100, 10, None))
        self.assertFalse(khandy.is_number_between(9, 10, None))
        # no bounds
        self.assertTrue(khandy.is_number_between(123, None, None))

    def test_invalid_interval(self):
        # invalid interval, i.e. empty set
        self.assertFalse(khandy.is_number_between(5, 10, 1))
        self.assertFalse(khandy.is_number_between(5, 10, 10, lower_close=False, upper_close=False))


if __name__ == '__main__':
    unittest.main()

