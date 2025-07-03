import unittest
import khandy


class TestFsUtils(unittest.TestCase):
    def test_normalize_extension(self):
        self.assertEqual(khandy.normalize_extension('.jpg'), '.jpg')
        self.assertEqual(khandy.normalize_extension('jpg'), '.jpg')
        self.assertEqual(khandy.normalize_extension('.JPG'), '.jpg')
        self.assertEqual(khandy.normalize_extension('JPG'), '.jpg')
        self.assertEqual(khandy.normalize_extension('.jPg'), '.jpg')
        
        self.assertEqual(khandy.normalize_extension(''), '.')
        self.assertEqual(khandy.normalize_extension('.'), '.')
        

if __name__ == '__main__':
    unittest.main()
    