import unittest
from khandy import split_content_with_paren


class TestSplitContentWithParen(unittest.TestCase):
    def test_valid_parentheses(self):
        """Test with valid parentheses."""
        self.assertEqual(split_content_with_paren("abc(def)", "hw"), ("abc", "def"))
        self.assertEqual(split_content_with_paren("ghi（jkl）", "fw"), ("ghi", "jkl"))
        self.assertEqual(split_content_with_paren("abc ( def )", "hw"), ("abc ", " def "))
        self.assertEqual(split_content_with_paren(" ( )", "hw"), (" ", " "))
        
        self.assertEqual(split_content_with_paren("(def)", "hw"), (None, "def"))
        self.assertEqual(split_content_with_paren("abc", "hw"), ("abc", None))
        self.assertEqual(split_content_with_paren("", "hw"), (None, None))
        
    def test_no_content_inside_paren(self):
        """Test when there's no content inside the parentheses."""
        with self.assertRaises(ValueError):
            split_content_with_paren("abc()", "hw")

    def test_only_parentheses(self):
        """Test when the entire string is just parentheses."""
        with self.assertRaises(ValueError):
            split_content_with_paren("()", "hw")

    def test_nested_parentheses(self):
        """Test with nested parentheses (pattern does not support nesting)."""
        with self.assertRaises(ValueError):
            split_content_with_paren("(a(b)c)", "hw")

    def test_extra_content_after_paren(self):
        """Test with content after the closing parenthesis."""
        with self.assertRaises(ValueError):
            split_content_with_paren("abc(def)ghi", "hw")
        with self.assertRaises(ValueError):
            split_content_with_paren("(def)ghi", "hw")
        with self.assertRaises(ValueError):
            split_content_with_paren("(def) ", "hw")
            
    def test_unmatched_parentheses(self):
        """Test with unmatched parentheses."""
        with self.assertRaises(ValueError):
            split_content_with_paren("abc(def", "hw")
        with self.assertRaises(ValueError):
            split_content_with_paren("abc)def", "hw")
 

if __name__ == '__main__':
    unittest.main()
    