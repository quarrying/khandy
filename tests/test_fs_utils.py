import os
import pathlib
import unittest
from unittest.mock import patch

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
        

original_relpath = os.path.relpath

class TestListItemsInDir(unittest.TestCase):

    @patch('khandy.fs_utils.os.path.isdir')
    @patch('khandy.fs_utils.os.path.exists')
    @patch('khandy.fs_utils.os.listdir')
    @patch('khandy.fs_utils.os.getcwd')
    @patch('khandy.fs_utils.os.path.expanduser')
    def test_list_items_non_recursive_full_path(self, mock_expanduser, mock_getcwd, 
                                                mock_listdir, mock_exists, mock_isdir):
        """Test listing items non-recursively with full paths"""
        mock_getcwd.return_value = '/current/dir'
        mock_expanduser.return_value = '/mocked/path'
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_listdir.return_value = ['file1.txt', 'subdir1', 'file2.jpg']
        
        result = khandy.list_items_in_dir('/some/path', recursive=False, full_path=True)
        
        expected = [
            os.path.join('/mocked/path', 'file1.txt'),
            os.path.join('/mocked/path', 'file2.jpg'),
            os.path.join('/mocked/path', 'subdir1')
        ]
        self.assertEqual(sorted(result), sorted(expected))

    @patch('khandy.fs_utils.os.path.isdir')
    @patch('khandy.fs_utils.os.path.exists')
    @patch('khandy.fs_utils.os.listdir')
    @patch('khandy.fs_utils.os.getcwd')
    @patch('khandy.fs_utils.os.path.expanduser')
    def test_list_items_non_recursive_relative_path(self, mock_expanduser, mock_getcwd, 
                                                    mock_listdir, mock_exists, mock_isdir):
        """Test listing items non-recursively without full paths"""
        mock_getcwd.return_value = '/current/dir'
        mock_expanduser.return_value = '/mocked/path'
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_listdir.return_value = ['file1.txt', 'subdir1', 'file2.jpg']
        
        result = khandy.list_items_in_dir('/some/path', recursive=False, full_path=False)
        
        expected = ['file1.txt', 'file2.jpg', 'subdir1']
        self.assertEqual(sorted(result), sorted(expected))

    @patch('khandy.fs_utils.os.walk')
    @patch('khandy.fs_utils.os.path.isdir')
    @patch('khandy.fs_utils.os.path.exists')
    @patch('khandy.fs_utils.os.getcwd')
    @patch('khandy.fs_utils.os.path.expanduser')
    def test_list_items_recursive_full_path(self, mock_expanduser, mock_getcwd, 
                                            mock_exists, mock_isdir, mock_walk):
        """Test listing items recursively with full paths"""
        mock_getcwd.return_value = '/current/dir'
        mock_expanduser.return_value = '/mocked/path'
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_walk.return_value = iter([
            ('/mocked/path', ['subdir1'], ['file1.txt']),
            (os.path.join('/mocked/path', 'subdir1'), [], ['subfile1.txt'])
        ])
        
        result = khandy.list_items_in_dir('/some/path', recursive=True, full_path=True)
        
        expected = [
            os.path.join('/mocked/path', 'file1.txt'),
            os.path.join('/mocked/path', 'subdir1'),
            os.path.join('/mocked/path', 'subdir1', 'subfile1.txt')
        ]
        self.assertEqual(sorted(result), sorted(expected))

    @patch('khandy.fs_utils.os.walk')
    @patch('khandy.fs_utils.os.path.relpath')
    @patch('khandy.fs_utils.os.path.isdir')
    @patch('khandy.fs_utils.os.path.exists')
    @patch('khandy.fs_utils.os.getcwd')
    @patch('khandy.fs_utils.os.path.expanduser')
    def test_list_items_recursive_relative_path(self, mock_expanduser, mock_getcwd, 
                                                mock_exists, mock_isdir, mock_relpath, mock_walk):
        """Test listing items recursively without full paths"""
        mock_getcwd.return_value = '/current/dir'
        mock_expanduser.return_value = '/mocked/path'
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_relpath.side_effect = lambda x, y: original_relpath(x, y)
        mock_walk.return_value = iter([
            ('/mocked/path', ['subdir1'], ['file1.txt']),
            (os.path.join('/mocked/path', 'subdir1'), [], ['subfile1.txt'])
        ])
        
        result = khandy.list_items_in_dir('/some/path', recursive=True, full_path=False)
        
        expected = [
            'file1.txt',
            'subdir1',
            os.path.join('subdir1', 'subfile1.txt')
        ]
        self.assertEqual(sorted(result), sorted(expected))

    @patch('khandy.fs_utils.os.path.isdir')
    @patch('khandy.fs_utils.os.path.exists')
    @patch('khandy.fs_utils.os.getcwd')
    @patch('khandy.fs_utils.os.path.expanduser')
    def test_path_not_exists(self, mock_expanduser, mock_getcwd, mock_exists, mock_isdir):
        """Test behavior when path doesn't exist"""
        mock_getcwd.return_value = '/current/dir'
        mock_expanduser.return_value = '/mocked/path'
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            khandy.list_items_in_dir('/nonexistent/path')

    @patch('khandy.fs_utils.os.path.isdir')
    @patch('khandy.fs_utils.os.path.exists')
    @patch('khandy.fs_utils.os.getcwd')
    @patch('khandy.fs_utils.os.path.expanduser')
    def test_path_is_not_directory(self, mock_expanduser, mock_getcwd, 
                                   mock_exists, mock_isdir):
        """Test behavior when path is not a directory"""
        mock_getcwd.return_value = '/current/dir'
        mock_expanduser.return_value = '/mocked/path'
        mock_exists.return_value = True
        mock_isdir.return_value = False
        
        with self.assertRaises(NotADirectoryError):
            khandy.list_items_in_dir('/not/a/directory')

    @patch('khandy.fs_utils.os.path.isdir')
    @patch('khandy.fs_utils.os.path.exists')
    @patch('khandy.fs_utils.os.listdir')
    @patch('khandy.fs_utils.os.getcwd')
    @patch('khandy.fs_utils.os.path.expanduser')
    def test_empty_directory(self, mock_expanduser, mock_getcwd, 
                             mock_listdir, mock_exists, mock_isdir):
        """Test behavior with an empty directory"""
        mock_getcwd.return_value = '/current/dir'
        mock_expanduser.return_value = '/mocked/path'
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_listdir.return_value = []
        
        result = khandy.list_items_in_dir('/empty/dir', recursive=False, full_path=False)
        
        self.assertEqual(result, [])

    @patch('khandy.fs_utils.os.path.isdir')
    @patch('khandy.fs_utils.os.path.exists')
    @patch('khandy.fs_utils.os.listdir')
    @patch('khandy.fs_utils.os.getcwd')
    @patch('khandy.fs_utils.os.path.expanduser')
    def test_default_path(self, mock_expanduser, mock_getcwd, 
                          mock_listdir, mock_exists, mock_isdir):
        """Test behavior when no path is provided (uses current working directory)"""
        mock_getcwd.return_value = '/current/dir'
        mock_expanduser.return_value = '/current/dir'
        mock_exists.return_value = True
        mock_isdir.return_value = True
        mock_listdir.return_value = ['file1.txt', 'subdir1']
        
        result = khandy.list_items_in_dir(recursive=False, full_path=False)
        
        expected = ['file1.txt', 'subdir1']
        self.assertEqual(sorted(result), sorted(expected))
        
        
class TestEscapeFilename(unittest.TestCase):
    """Unit tests for fs_utils.khandy.escape_filename"""

    def test_normal_filename(self):
        """Test that a normal filename remains unchanged."""
        self.assertEqual(khandy.escape_filename("normal_file.txt"), "normal_file.txt")
        self.assertEqual(khandy.escape_filename("normal file.txt"), "normal file.txt")

    def test_replace_backslash(self):
        """Test replacement of backslash."""
        filename = "file\\name.txt"
        result = khandy.escape_filename(filename)
        self.assertEqual(result, "file_name.txt")

    def test_replace_forward_slash(self):
        """Test replacement of forward slash."""
        filename = "file/name.txt"
        result = khandy.escape_filename(filename)
        self.assertEqual(result, "file_name.txt")

    def test_replace_asterisk(self):
        """Test replacement of asterisk."""
        filename = "file*name.txt"
        result = khandy.escape_filename(filename)
        self.assertEqual(result, "file_name.txt")

    def test_replace_question_mark(self):
        """Test replacement of question mark."""
        filename = "file?name.txt"
        result = khandy.escape_filename(filename)
        self.assertEqual(result, "file_name.txt")

    def test_replace_colon(self):
        """Test replacement of colon."""
        filename = "file:name.txt"
        result = khandy.escape_filename(filename)
        self.assertEqual(result, "file_name.txt")

    def test_replace_double_quote(self):
        """Test replacement of double quote."""
        filename = 'file"name.txt'
        result = khandy.escape_filename(filename)
        self.assertEqual(result, "file_name.txt")

    def test_replace_less_than(self):
        """Test replacement of less than sign."""
        filename = "file<name.txt"
        result = khandy.escape_filename(filename)
        self.assertEqual(result, "file_name.txt")

    def test_replace_greater_than(self):
        """Test replacement of greater than sign."""
        filename = "file>name.txt"
        result = khandy.escape_filename(filename)
        self.assertEqual(result, "file_name.txt")

    def test_replace_pipe(self):
        """Test replacement of pipe character."""
        filename = "file|name.txt"
        result = khandy.escape_filename(filename)
        self.assertEqual(result, "file_name.txt")

    def test_replace_control_characters(self):
        """Test replacement of control characters (0x00-0x1F)."""
        # Testing a few common control characters like newline, tab, etc.
        filename = "file\x00\x01\x02name.txt"
        result = khandy.escape_filename(filename)
        self.assertEqual(result, "file___name.txt")

    def test_custom_replacement_char(self):
        """Test using a custom replacement character."""
        filename = "file/name.txt"
        result = khandy.escape_filename(filename, new_char="-")
        self.assertEqual(result, "file-name.txt")

    def test_empty_string(self):
        """Test handling of an empty string."""
        filename = ""
        result = khandy.escape_filename(filename)
        self.assertEqual(result, "")

    def test_only_invalid_chars(self):
        """Test filename consisting only of invalid characters."""
        filename = "/*?"
        result = khandy.escape_filename(filename)
        self.assertEqual(result, "___")

    def test_no_extension(self):
        """Test filename without extension."""
        filename = "file/name"
        result = khandy.escape_filename(filename)
        self.assertEqual(result, "file_name")

    def test_multiple_consecutive_invalid_chars(self):
        """Test multiple consecutive invalid characters."""
        filename = "file///name.txt"
        result = khandy.escape_filename(filename)
        self.assertEqual(result, "file___name.txt")


if __name__ == '__main__':
    unittest.main()
