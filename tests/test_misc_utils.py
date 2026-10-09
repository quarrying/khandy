import os
import tempfile
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


class TestGetGitRepoRoot(unittest.TestCase):
    def test_returns_root_for_repo(self):
        # Test suite is itself run from inside this repository
        root = khandy.get_git_repo_root()
        self.assertTrue(os.path.isabs(root))
        self.assertTrue(os.path.isdir(root))
        self.assertTrue(os.path.exists(os.path.join(root, '.git')))

    def test_default_arg_matches_cwd(self):
        self.assertEqual(
            khandy.get_git_repo_root(),
            khandy.get_git_repo_root('.'),
        )

    def test_nested_path_returns_root(self):
        here = os.path.dirname(os.path.abspath(__file__))
        root_from_here = khandy.get_git_repo_root(here)
        root_from_root = khandy.get_git_repo_root('.')
        self.assertEqual(os.path.normpath(root_from_here),
                         os.path.normpath(root_from_root))

    def test_nonexistent_path_raises(self):
        with self.assertRaises(FileNotFoundError):
            khandy.get_git_repo_root('Z:/definitely/not/here/12345')

    def test_file_not_directory_raises(self):
        this_file = os.path.abspath(__file__)
        with self.assertRaises(NotADirectoryError):
            khandy.get_git_repo_root(this_file)

    def test_not_a_git_repo_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                khandy.get_git_repo_root(tmp)

    def test_returns_normalized_path(self):
        root = khandy.get_git_repo_root('.')
        # Should not contain . or .. segments
        self.assertNotIn('..', root.split(os.sep))


if __name__ == '__main__':
    unittest.main()