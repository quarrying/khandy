import pathlib
import unittest
import khandy


def _compare_get_path_parts(path):
    return len(khandy.get_path_parts(path)) == len(pathlib.Path(path).parts)


class TestFsUtils(unittest.TestCase):
    def test_normalize_extension(self):
        self.assertEqual(khandy.normalize_extension('.jpg'), '.jpg')
        self.assertEqual(khandy.normalize_extension('jpg'), '.jpg')
        self.assertEqual(khandy.normalize_extension('.JPG'), '.jpg')
        self.assertEqual(khandy.normalize_extension('JPG'), '.jpg')
        self.assertEqual(khandy.normalize_extension('.jPg'), '.jpg')
        
        self.assertEqual(khandy.normalize_extension(''), '.')
        self.assertEqual(khandy.normalize_extension('.'), '.')
        
    def test_get_path_parts(self):
        self.assertTrue(_compare_get_path_parts(''))
        self.assertTrue(_compare_get_path_parts('/'))
        self.assertTrue(_compare_get_path_parts('.'))
        self.assertTrue(_compare_get_path_parts('..'))

        self.assertTrue(_compare_get_path_parts('a'))
        self.assertTrue(_compare_get_path_parts('a/'))
        self.assertTrue(_compare_get_path_parts('a/b'))
        self.assertTrue(_compare_get_path_parts('a/b/'))

        self.assertTrue(_compare_get_path_parts('/a'))
        self.assertTrue(_compare_get_path_parts('/a/'))
        self.assertTrue(_compare_get_path_parts('/a/b'))
        self.assertTrue(_compare_get_path_parts('/a/b/'))
        
        self.assertTrue(_compare_get_path_parts('c:'))
        self.assertTrue(_compare_get_path_parts('c:/'))
        self.assertTrue(_compare_get_path_parts('c:/a'))
        self.assertTrue(_compare_get_path_parts('c:/a/'))

        self.assertTrue(_compare_get_path_parts('./a'))
        self.assertTrue(_compare_get_path_parts('./a/'))
        self.assertTrue(_compare_get_path_parts('./a/b'))
        self.assertTrue(_compare_get_path_parts('./a/b/'))

        self.assertTrue(_compare_get_path_parts('../a'))
        self.assertTrue(_compare_get_path_parts('../a/'))
        self.assertTrue(_compare_get_path_parts('../a/b'))
        self.assertTrue(_compare_get_path_parts('../a/b/'))


if __name__ == '__main__':
    unittest.main()
    