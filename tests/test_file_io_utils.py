import os
import tempfile
import unittest

import khandy


class TestLoadList(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.filename = os.path.join(self.tmp_dir.name, 'test.txt')

    def _write(self, content):
        with open(self.filename, 'w', encoding='utf-8', newline='') as f:
            f.write(content)

    def test_default_read(self):
        # ends with newline
        self._write('a\nb\nc\n')
        self.assertEqual(khandy.load_list(self.filename), ['a', 'b', 'c'])
        # does not end with newline
        self._write('a\nb\nc')
        self.assertEqual(khandy.load_list(self.filename), ['a', 'b', 'c'])

    def test_empty_and_blank_lines(self):
        self._write('')
        self.assertEqual(khandy.load_list(self.filename), [])
        self._write('a\n\nb\n')
        self.assertEqual(khandy.load_list(self.filename), ['a', '', 'b'])

    def test_crlf(self):
        self._write('a\r\nb\r\n')
        self.assertEqual(khandy.load_list(self.filename), ['a', 'b'])

    def test_start_and_stop(self):
        self._write('0\n1\n2\n3\n4\n')
        # start only
        self.assertEqual(khandy.load_list(self.filename, start=2), ['2', '3', '4'])
        # stop is exclusive, i.e. lines[start:stop]
        self.assertEqual(khandy.load_list(self.filename, stop=2), ['0', '1'])
        self.assertEqual(khandy.load_list(self.filename, start=1, stop=3), ['1', '2'])
        # stop beyond file length
        self.assertEqual(khandy.load_list(self.filename, start=3, stop=100), ['3', '4'])

    def test_invalid_start(self):
        self._write('a\nb\n')
        with self.assertRaises(TypeError):
            khandy.load_list(self.filename, start='0')
        with self.assertRaises(ValueError):
            khandy.load_list(self.filename, start=-1)

    def test_invalid_stop(self):
        self._write('a\nb\n')
        with self.assertRaises(TypeError):
            khandy.load_list(self.filename, stop=1.5)
        with self.assertRaises(ValueError):
            khandy.load_list(self.filename, start=1, stop=1)
        with self.assertRaises(ValueError):
            khandy.load_list(self.filename, start=2, stop=1)
        # stop=None is always valid
        self.assertEqual(khandy.load_list(self.filename, stop=None), ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
