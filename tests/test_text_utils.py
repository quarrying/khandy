import re
import unittest
from typing import Optional, Tuple

from khandy import split_content_with_paren
from khandy import split_before_after


class TestSplitContentWithParen(unittest.TestCase):
    def test_valid_parentheses(self):
        """Test with valid parentheses."""
        self.assertEqual(split_content_with_paren("abc(def)", "hw"), ("abc", "def"))
        self.assertEqual(split_content_with_paren("ghi（jkl）", "fw"), ("ghi", "jkl"))
        self.assertEqual(split_content_with_paren("abc ( def )", "hw"), ("abc ", " def "))
        self.assertEqual(split_content_with_paren(" ( )", "hw"), (" ", " "))
        
    def test_no_outside_content(self):
        """Test when there's no outside content."""
        self.assertEqual(split_content_with_paren("(def)", "hw"), (None, "def"))
        self.assertEqual(split_content_with_paren("（jkl）", "fw"), (None, "jkl"))
        
    def test_no_paren(self):
        """Test when there's no parentheses in the string."""
        self.assertEqual(split_content_with_paren("abc", "hw"), ("abc", None))
        self.assertEqual(split_content_with_paren("", "hw"), (None, None))
        
    def test_no_content_inside_paren(self):
        """Test when there's no content inside the parentheses."""
        self.assertEqual(split_content_with_paren("abc()", "hw"), ("abc", ""))
        self.assertEqual(split_content_with_paren("()", "hw"), (None, ""))

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
 

def split_before_after_v2(
    label: str, 
    strip_whitespace: bool = True
) -> Tuple[Optional[str], Optional[str]]:
    if strip_whitespace:
        label = label.strip()
        
    PATTERN = r"(?P<entity>[^$]+)?(?:\$(?P<subset>[^$]+))?"
    pattern = re.compile(PATTERN)
    matched = pattern.fullmatch(label)
    if matched is None:
        raise ValueError(f'parse failure: "{label}"')
    else:
        entity, subset_str = matched.groups()
    if strip_whitespace and entity is not None:
        entity = entity.strip()
    if strip_whitespace and subset_str is not None:
        subset_str = subset_str.strip()
        
    return entity, subset_str


class TestSplitBeforeAfter(unittest.TestCase):
    def test_split_before_after_when_not_stripping(self):
        self.assertEqual(split_before_after('entity', False), ('entity', None))
        self.assertEqual(split_before_after('entity$subset', False), ('entity', 'subset'))
        self.assertEqual(split_before_after('$subset', False), (None, 'subset'))
        self.assertEqual(split_before_after('', False), (None, None))
        self.assertEqual(split_before_after('entity$ ', False), ('entity', ' '))
        self.assertEqual(split_before_after(' $subset', False), (' ', 'subset'))
        self.assertEqual(split_before_after('entity $subset', False), ('entity ', 'subset'))
        
        with self.assertRaises(ValueError):
            self.assertEqual(split_before_after('entity$', False), ('entity', None))
        with self.assertRaises(ValueError):
            self.assertEqual(split_before_after('$', False), (None, None))
        with self.assertRaises(ValueError):
            split_before_after('invalid$format$here', False)

    def test_split_before_after_when_strip(self):
        self.assertEqual(split_before_after('entity', True), ('entity', None))
        self.assertEqual(split_before_after('entity$subset', True), ('entity', 'subset'))
        self.assertEqual(split_before_after('$subset', True), (None, 'subset'))
        self.assertEqual(split_before_after('', True), (None, None))
        self.assertEqual(split_before_after('entity $subset', True), ('entity', 'subset'))
        
        with self.assertRaises(ValueError):
            self.assertEqual(split_before_after('entity$ ', True), ('entity', ' '))
        self.assertEqual(split_before_after(' $subset', True), (None, 'subset'))
        
        with self.assertRaises(ValueError):
            self.assertEqual(split_before_after('entity$', True), ('entity', None))
        with self.assertRaises(ValueError):
            self.assertEqual(split_before_after('$', True), (None, None))
        with self.assertRaises(ValueError):
            split_before_after('invalid$format$here', True)
            
    def test_split_before_after_v2_when_not_stripping(self):
        self.assertEqual(split_before_after_v2('entity', False), ('entity', None))
        self.assertEqual(split_before_after_v2('entity$subset', False), ('entity', 'subset'))
        self.assertEqual(split_before_after_v2('$subset', False), (None, 'subset'))
        self.assertEqual(split_before_after_v2('', False), (None, None))
        self.assertEqual(split_before_after_v2('entity$ ', False), ('entity', ' '))
        self.assertEqual(split_before_after_v2(' $subset', False), (' ', 'subset'))
        self.assertEqual(split_before_after_v2('entity $subset', False), ('entity ', 'subset'))
        
        with self.assertRaises(ValueError):
            self.assertEqual(split_before_after_v2('entity$', False), ('entity', None))
        with self.assertRaises(ValueError):
            self.assertEqual(split_before_after_v2('$', False), (None, None))
        with self.assertRaises(ValueError):
            split_before_after_v2('invalid$format$here', False)

    def test_split_before_after_v2_when_strip(self):
        self.assertEqual(split_before_after_v2('entity', True), ('entity', None))
        self.assertEqual(split_before_after_v2('entity$subset', True), ('entity', 'subset'))
        self.assertEqual(split_before_after_v2('$subset', True), (None, 'subset'))
        self.assertEqual(split_before_after_v2('', True), (None, None))
        self.assertEqual(split_before_after_v2('entity $subset', True), ('entity', 'subset'))

        with self.assertRaises(ValueError):
            self.assertEqual(split_before_after_v2('entity$ ', True), ('entity', ' '))
        self.assertEqual(split_before_after_v2(' $subset', True), (None, 'subset'))

        with self.assertRaises(ValueError):
            self.assertEqual(split_before_after_v2('entity$', True), ('entity', None))
        with self.assertRaises(ValueError):
            self.assertEqual(split_before_after_v2('$', True), (None, None))
        with self.assertRaises(ValueError):
            split_before_after_v2('invalid$format$here', True)
            

if __name__ == '__main__':
    unittest.main()
    