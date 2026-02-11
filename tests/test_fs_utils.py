import os
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

    def test_get_path_stem(self):
        self.assertEqual(khandy.get_path_stem('/foo/bar.txt'), 'bar')
        self.assertEqual(khandy.get_path_stem('/foo/bar'), 'bar')
        self.assertEqual(khandy.get_path_stem('/foo/.txt'), '.txt')
        self.assertEqual(khandy.get_path_stem('/foo/'), '')
        self.assertEqual(khandy.get_path_stem('.'), '.')
        self.assertEqual(khandy.get_path_stem('..'), '..')
        
    def test_replace_path_stem(self):
        self.assertEqual(khandy.replace_path_stem('', ''), '')
        self.assertEqual(khandy.replace_path_stem('bar', ''), '')
        self.assertEqual(khandy.replace_path_stem('bar.txt', ''), '.txt')
        self.assertEqual(khandy.replace_path_stem('.txt', ''), '')
        
        self.assertEqual(khandy.replace_path_stem('', 'baz'), 'baz')
        self.assertEqual(khandy.replace_path_stem('bar', 'baz'), 'baz')
        self.assertEqual(khandy.replace_path_stem('bar.txt', 'baz'), 'baz.txt')
        self.assertEqual(khandy.replace_path_stem('.txt', 'baz'), 'baz')
        
        self.assertEqual(khandy.replace_path_stem('/foo/', ''), f'/foo{os.sep}')
        self.assertEqual(khandy.replace_path_stem('/foo/bar', ''), f'/foo{os.sep}')
        self.assertEqual(khandy.replace_path_stem('/foo/bar.txt', ''), f'/foo{os.sep}.txt')
        self.assertEqual(khandy.replace_path_stem('/foo/.txt', ''), f'/foo{os.sep}')
        
        self.assertEqual(khandy.replace_path_stem('/foo/', 'baz'), f'/foo{os.sep}baz')
        self.assertEqual(khandy.replace_path_stem('/foo/bar', 'baz'), f'/foo{os.sep}baz')
        self.assertEqual(khandy.replace_path_stem('/foo/bar.txt', 'baz'), f'/foo{os.sep}baz.txt')
        self.assertEqual(khandy.replace_path_stem('/foo/.txt', 'baz'), f'/foo{os.sep}baz')
    
    def test_replace_path_stem_with_callable(self):
        stem_change_func = lambda x: khandy.get_path_stem(x).replace('-', '_')
        self.assertEqual(khandy.replace_path_stem('a-1.txt', stem_change_func), 'a_1.txt')
        
    def test_get_path_extension(self):
        self.assertEqual(khandy.get_path_extension("/foo/bar.txt"), '.txt')
        self.assertEqual(khandy.get_path_extension("/foo/bar."), '.')
        self.assertEqual(khandy.get_path_extension("/foo/bar"), '')

        self.assertEqual(khandy.get_path_extension("/foo/"), '')
        self.assertEqual(khandy.get_path_extension("/foo/."), '')
        self.assertEqual(khandy.get_path_extension("/foo/.."), '')
        self.assertEqual(khandy.get_path_extension("/foo/.hidden"), '')
        self.assertEqual(khandy.get_path_extension("/foo/..bar"), '')
        
        self.assertEqual(khandy.get_path_extension("/foo/bar.txt/bar.cc"), '.cc')
        self.assertEqual(khandy.get_path_extension("/foo/bar.txt/bar."), '.')
        self.assertEqual(khandy.get_path_extension("/foo/bar.txt/bar"), '')
        
    def test_replace_path_extension(self):
        self.assertEqual(khandy.replace_path_extension('/foo/bar.jpeg', 'jpg'), '/foo/bar.jpg')
        self.assertEqual(khandy.replace_path_extension('/foo/bar.jpeg', '.jpg'), '/foo/bar.jpg')
        self.assertEqual(khandy.replace_path_extension('/foo/bar.jpeg', '.'), '/foo/bar.')
        self.assertEqual(khandy.replace_path_extension('/foo/bar.jpeg', ''), '/foo/bar')
        self.assertEqual(khandy.replace_path_extension('/foo/bar.jpeg', None), '/foo/bar')
        
        self.assertEqual(khandy.replace_path_extension('/foo/bar.', 'jpg'), '/foo/bar.jpg')
        self.assertEqual(khandy.replace_path_extension('/foo/bar.', '.jpg'), '/foo/bar.jpg')
        self.assertEqual(khandy.replace_path_extension('/foo/bar.', '.'), '/foo/bar.')
        self.assertEqual(khandy.replace_path_extension('/foo/bar.', ''), '/foo/bar')
        self.assertEqual(khandy.replace_path_extension('/foo/bar.', None), '/foo/bar')
        
        self.assertEqual(khandy.replace_path_extension('/foo/bar', 'jpg'), '/foo/bar.jpg')
        self.assertEqual(khandy.replace_path_extension('/foo/bar', '.jpg'), '/foo/bar.jpg')
        self.assertEqual(khandy.replace_path_extension('/foo/bar', '.'), '/foo/bar.')
        self.assertEqual(khandy.replace_path_extension('/foo/bar', ''), '/foo/bar')
        self.assertEqual(khandy.replace_path_extension('/foo/bar', None), '/foo/bar')
        
        self.assertEqual(khandy.replace_path_extension('/foo/.', 'jpg'), '/foo/..jpg')
        self.assertEqual(khandy.replace_path_extension('/foo/.', '.jpg'), '/foo/..jpg')
        self.assertEqual(khandy.replace_path_extension('/foo/.', '.'), '/foo/..')
        self.assertEqual(khandy.replace_path_extension('/foo/.', ''), '/foo/.')
        self.assertEqual(khandy.replace_path_extension('/foo/.', None), '/foo/.')
        
        self.assertEqual(khandy.replace_path_extension('/foo/', 'jpg'), '/foo/.jpg')
        self.assertEqual(khandy.replace_path_extension('/foo/', '.jpg'), '/foo/.jpg')
        self.assertEqual(khandy.replace_path_extension('/foo/', '.'), '/foo/.')
        self.assertEqual(khandy.replace_path_extension('/foo/', ''), '/foo/')
        self.assertEqual(khandy.replace_path_extension('/foo/', None), '/foo/')
        
        self.assertEqual(khandy.replace_path_extension('/foo/.jpeg', 'jpg'), '/foo/.jpeg.jpg')
        self.assertEqual(khandy.replace_path_extension('/foo/.jpeg', '.jpg'), '/foo/.jpeg.jpg')
        self.assertEqual(khandy.replace_path_extension('/foo/.jpeg', '.'), '/foo/.jpeg.')
        self.assertEqual(khandy.replace_path_extension('/foo/.jpeg', ''), '/foo/.jpeg')
        self.assertEqual(khandy.replace_path_extension('/foo/.jpeg', None), '/foo/.jpeg')
        
    def test_replace_path_extension_with_callable(self):
        ext_change_func = lambda x: khandy.get_path_extension(x).lower()
        self.assertEqual(khandy.replace_path_extension('FOO.JPG', ext_change_func), 'FOO.jpg')
        
    def test_replace_parent_dir(self):
        self.assertEqual(khandy.replace_path_parent('foo/bar.txt', 'baz'), f'baz{os.sep}bar.txt')
        self.assertEqual(khandy.replace_path_parent('foo/bar.txt', ''), 'bar.txt')
        self.assertEqual(khandy.replace_path_parent('foo/bar.txt', None), 'bar.txt')
        
    
if __name__ == '__main__':
    unittest.main()
