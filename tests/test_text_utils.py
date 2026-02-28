import re
import unittest
from typing import Optional, Tuple

import khandy
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
            

class TestHasNestedOrUnmatchedParen(unittest.TestCase):

    def test_valid_parentheses_hw_half_width(self):
        # Test valid parentheses (no nesting, balanced)
        self.assertFalse(khandy.has_nested_or_unmatched_paren("", "hw"))
        self.assertFalse(khandy.has_nested_or_unmatched_paren("abc", "hw"))
        self.assertFalse(khandy.has_nested_or_unmatched_paren("(abc)", "hw"))
        self.assertFalse(khandy.has_nested_or_unmatched_paren("abc(def)", "hw"))
        self.assertFalse(khandy.has_nested_or_unmatched_paren("(abc)def", "hw"))
        self.assertFalse(khandy.has_nested_or_unmatched_paren("abc(def)ghi", "hw"))

    def test_valid_parentheses_fw_full_width(self):
        # Test valid parentheses (no nesting, balanced) with full-width
        self.assertFalse(khandy.has_nested_or_unmatched_paren("", "fw"))
        self.assertFalse(khandy.has_nested_or_unmatched_paren("abc", "fw"))
        self.assertFalse(khandy.has_nested_or_unmatched_paren("（abc）", "fw"))
        self.assertFalse(khandy.has_nested_or_unmatched_paren("abc（def）", "fw"))
        self.assertFalse(khandy.has_nested_or_unmatched_paren("（abc）def", "fw"))
        self.assertFalse(khandy.has_nested_or_unmatched_paren("abc（def）ghi", "fw"))

    def test_nested_parentheses_half_width(self):
        # Test nested parentheses
        self.assertTrue(khandy.has_nested_or_unmatched_paren("(())", "hw"))
        self.assertTrue(khandy.has_nested_or_unmatched_paren("（（abc））", "fw"))

    def test_unmatched_parentheses_half_width(self):
        # Test unmatched parentheses (left over)
        self.assertTrue(khandy.has_nested_or_unmatched_paren("(", "hw"))
        self.assertTrue(khandy.has_nested_or_unmatched_paren(")", "hw"))
        self.assertTrue(khandy.has_nested_or_unmatched_paren(")abc", "hw"))
        self.assertTrue(khandy.has_nested_or_unmatched_paren("abc(", "hw"))
        
        self.assertTrue(khandy.has_nested_or_unmatched_paren("）abc", "fw"))  # Unmatched right
        self.assertTrue(khandy.has_nested_or_unmatched_paren("abc（", "fw"))  # Unmatched left

    def test_invalid_paren_type(self):
        # Test invalid paren_type
        with self.assertRaises(ValueError):
            khandy.has_nested_or_unmatched_paren("abc", "invalid")
        with self.assertRaises(ValueError):
            khandy.has_nested_or_unmatched_paren("abc", "HW")  # Case sensitive
        with self.assertRaises(ValueError):
            khandy.has_nested_or_unmatched_paren("abc", "FW")  # Case sensitive
        with self.assertRaises(ValueError):
            khandy.has_nested_or_unmatched_paren("abc", "xyz")


def parse_score_str(score_str, str_len=None, lower=0.0, upper=1.0):
    # 当 str_len 不为 None, 检查 score_str 的长度是否等于 str_len
    # 若不等于, 则直接返回 None, 否则进入下一步判断
    if str_len is not None and (len(score_str) != str_len):
        return None
    try:
        score = float(score_str)
        return score if lower <= score <= upper else None
    except:
        return None
    

class TestUpsert(unittest.TestCase):
    def test_upsert_prefix_into_path_stem(self):
        text = f'{0.5:.05f}'
        validator = lambda text: parse_score_str(text) is not None
        self.assertEqual(khandy.upsert_prefix_into_path_stem('', text), '0.50000')
        self.assertEqual(khandy.upsert_prefix_into_path_stem('.jpg', text), '0.50000.jpg')
        self.assertEqual(khandy.upsert_prefix_into_path_stem('_a.jpg', text), '0.50000_a.jpg')
        self.assertEqual(khandy.upsert_prefix_into_path_stem('a.jpg', text), '0.50000_a.jpg')

        self.assertEqual(khandy.upsert_prefix_into_path_stem('', text, validator), '0.50000')
        self.assertEqual(khandy.upsert_prefix_into_path_stem('.jpg', text, validator), '0.50000.jpg')
        self.assertEqual(khandy.upsert_prefix_into_path_stem('_a.jpg', text, validator), '0.50000_a.jpg')
        self.assertEqual(khandy.upsert_prefix_into_path_stem('a.jpg', text, validator), '0.50000_a.jpg')

        self.assertEqual(khandy.upsert_prefix_into_path_stem('0.5.jpg', text, validator), '0.50000.jpg')
        self.assertEqual(khandy.upsert_prefix_into_path_stem('0.5_a.jpg', text, validator), '0.50000_a.jpg')

    def test_upsert_suffix_into_path_stem(self):
        text = f'{0.5:.05f}'
        validator = lambda text: parse_score_str(text) is not None
        self.assertEqual(khandy.upsert_suffix_into_path_stem('', text), '0.50000')
        self.assertEqual(khandy.upsert_suffix_into_path_stem('.jpg', text), '0.50000.jpg')
        self.assertEqual(khandy.upsert_suffix_into_path_stem('a_.jpg', text), 'a_0.50000.jpg')
        self.assertEqual(khandy.upsert_suffix_into_path_stem('a.jpg', text), 'a_0.50000.jpg')
        self.assertEqual(khandy.upsert_suffix_into_path_stem('', text, validator), '0.50000')
        self.assertEqual(khandy.upsert_suffix_into_path_stem('.jpg', text, validator), '0.50000.jpg')
        self.assertEqual(khandy.upsert_suffix_into_path_stem('a_.jpg', text, validator), 'a_0.50000.jpg')
        self.assertEqual(khandy.upsert_suffix_into_path_stem('a.jpg', text, validator), 'a_0.50000.jpg')

        validator = lambda text: text in ['PASS', 'KEEP']
        self.assertEqual(khandy.upsert_suffix_into_path_stem('0.5_KEEP.jpg', 'KEEP', validator), '0.5_KEEP.jpg')
        self.assertEqual(khandy.upsert_suffix_into_path_stem('0.5_a_PASS.jpg', 'PASS', validator), '0.5_a_PASS.jpg')
        self.assertEqual(khandy.upsert_suffix_into_path_stem('0.5_a_PASS.jpg', 'KEEP', validator), '0.5_a_KEEP.jpg')
        self.assertEqual(khandy.upsert_suffix_into_path_stem('0.5_a_FOO.jpg', 'PASS', validator), '0.5_a_FOO_PASS.jpg')


class TestParseRangeString(unittest.TestCase):
    
    def test_basic_ranges(self):
        # Test basic comma-separated numbers
        self.assertEqual(khandy.parse_range_string("1,2,3"), [1, 2, 3])
        # Test basic range
        self.assertEqual(khandy.parse_range_string("1-3"), [1, 2, 3])
        # Test mixed numbers and ranges
        self.assertEqual(khandy.parse_range_string("1,3-5,7"), [1, 3, 4, 5, 7])
        # Test single number
        self.assertEqual(khandy.parse_range_string("5"), [5])
    
    def test_circle_digits(self):
        # Test circle digits conversion
        self.assertEqual(khandy.parse_range_string("①,③-⑤,⑦"), [1, 3, 4, 5, 7])
        # Test mixed regular and circle digits
        self.assertEqual(khandy.parse_range_string("1,③-5,⑦"), [1, 3, 4, 5, 7])
    
    def test_custom_separators(self):
        # Test custom separator
        self.assertEqual(khandy.parse_range_string("1;3-5;7", sep=';'), [1, 3, 4, 5, 7])
        # Test custom range separator
        self.assertEqual(khandy.parse_range_string("1,3~5,7", range_sep='~'), [1, 3, 4, 5, 7])
    
    def test_edge_cases(self):
        # Test empty string
        self.assertEqual(khandy.parse_range_string(""), [])
        # Test duplicate numbers
        self.assertEqual(khandy.parse_range_string("1,1,2,2,3"), [1, 2, 3])
        # Test overlapping ranges
        self.assertEqual(khandy.parse_range_string("1-3,2-4"), [1, 2, 3, 4])
        # Test reverse range (should work)
        self.assertEqual(khandy.parse_range_string("5-1"), [])
    
    def test_invalid_inputs(self):
        # Test invalid range format
        with self.assertRaises(ValueError):
            khandy.parse_range_string("1-2-3")
        # Test invalid range with non-numeric values
        with self.assertRaises(ValueError):
            khandy.parse_range_string("1-a")
        # Test invalid range separator
        with self.assertRaises(ValueError):
            khandy.parse_range_string("1~3")


class TestStrContains(unittest.TestCase):

    def test_single_substring_found(self):
        # Test case where a single substring is found
        self.assertTrue(khandy.str_contains("hello world", "world"))
        self.assertTrue(khandy.str_contains("hello world", "hello"))
        self.assertTrue(khandy.str_contains("hello world", "o w"))

    def test_single_substring_not_found(self):
        # Test case where a single substring is not found
        self.assertFalse(khandy.str_contains("hello world", "python"))
        self.assertFalse(khandy.str_contains("hello world", "xyz"))

    def test_tuple_of_substrings_found(self):
        # Test case where any substring in the tuple is found
        self.assertTrue(khandy.str_contains("hello world", ("world", "python")))
        self.assertTrue(khandy.str_contains("hello world", ("hello", "xyz")))
        self.assertTrue(khandy.str_contains("hello world", ("test", "world")))

    def test_tuple_of_substrings_not_found(self):
        # Test case where none of the substrings in the tuple are found
        self.assertFalse(khandy.str_contains("hello world", ("python", "java")))
        self.assertFalse(khandy.str_contains("hello world", ("xyz", "abc")))

    def test_with_start_and_end_indices(self):
        # Test case with start and end indices
        self.assertTrue(khandy.str_contains("hello world", "world", start=6))
        self.assertFalse(khandy.str_contains("hello world", "hello", start=6))
        self.assertTrue(khandy.str_contains("hello world", "hello", end=5))
        self.assertFalse(khandy.str_contains("hello world", "world", end=5))

    def test_empty_substring(self):
        # Test case with an empty substring (should always return True)
        self.assertTrue(khandy.str_contains("hello world", ""))
        self.assertTrue(khandy.str_contains("", ""))

    def test_empty_tuple(self):
        # Test case with an empty tuple (should return False)
        self.assertFalse(khandy.str_contains("hello world", ()))

    def test_invalid_sub_type(self):
        # Test case where sub is neither a string nor a tuple
        with self.assertRaises(TypeError):
            khandy.str_contains("hello world", 123)

    def test_invalid_tuple_item_type(self):
        # Test case where tuple contains non-string items
        with self.assertRaises(TypeError):
            khandy.str_contains("hello world", ("valid", 123))
        with self.assertRaises(TypeError):
            khandy.str_contains("hello world", (123, "valid"))

    def test_edge_cases(self):
        # Edge cases
        self.assertTrue(khandy.str_contains("a", "a"))  # Single character match
        self.assertFalse(khandy.str_contains("a", "b"))  # Single character mismatch
        self.assertTrue(khandy.str_contains("abc", "abc"))  # Full string match
        self.assertFalse(khandy.str_contains("abc", "abcd"))  # Substring longer than string


class TestStrSplit(unittest.TestCase):
    
    def test_single_string_separator(self):
        """Test splitting with a single string separator"""
        result = khandy.str_split("apple-banana-cherry", "-")
        self.assertEqual(result, ["apple", "banana", "cherry"])
        
    def test_none_separator_default_whitespace(self):
        """Test splitting with None separator (default whitespace behavior)"""
        result = khandy.str_split("  apple   banana  cherry  ")
        self.assertEqual(result, ["apple", "banana", "cherry"])
        
    def test_tuple_separators_basic(self):
        """Test splitting with tuple of separators"""
        result = khandy.str_split("apple-banana~cherry#date", ('-', '~', '#'))
        self.assertEqual(result, ["apple", "banana", "cherry", "date"])
        
    def test_tuple_separators_with_maxsplit_zero(self):
        """Test splitting with tuple separators when maxsplit is 0"""
        result = khandy.str_split("apple-banana~cherry#date", ('-', '~', '#'), maxsplit=0)
        self.assertEqual(result, ["apple-banana~cherry#date"])
        
    def test_tuple_separators_with_maxsplit_positive(self):
        """Test splitting with tuple separators and positive maxsplit"""
        result = khandy.str_split("apple-banana~cherry#date", ('-', '~', '#'), maxsplit=2)
        self.assertEqual(result, ["apple", "banana", "cherry#date"])
        
    def test_tuple_separators_with_maxsplit_negative(self):
        """Test splitting with tuple separators when maxsplit is negative"""
        result = khandy.str_split("apple-banana~cherry#date", ('-', '~', '#'), maxsplit=-1)
        self.assertEqual(result, ["apple", "banana", "cherry", "date"])
        
    def test_empty_string_input(self):
        """Test splitting empty string"""
        result = khandy.str_split("", ('-', '~', '#'))
        self.assertEqual(result, [""])
        
    def test_empty_string_with_single_separator(self):
        """Test splitting empty string with single separator"""
        result = khandy.str_split("", "-")
        self.assertEqual(result, [""])
        
    def test_no_matches_with_tuple_separators(self):
        """Test when no separators are found in the string"""
        result = khandy.str_split("applebanana", ('-', '~', '#'))
        self.assertEqual(result, ["applebanana"])
        
    def test_consecutive_separators(self):
        """Test handling consecutive separators"""
        result = khandy.str_split("apple--banana~~cherry", ('-', '~'))
        self.assertEqual(result, ["apple", "", "banana", "", "cherry"])
        
    def test_special_regex_chars_in_separators(self):
        """Test separators containing special regex characters"""
        result = khandy.str_split("apple.banana^cherry$end", ('.', '^', '$'))
        self.assertEqual(result, ["apple", "banana", "cherry", "end"])
        
    def test_maxsplit_one_with_tuple(self):
        """Test maxsplit=1 with tuple separators"""
        result = khandy.str_split("apple-banana~cherry#date", ('-', '~'), maxsplit=1)
        self.assertEqual(result, ["apple", "banana~cherry#date"])
        
    def test_original_behavior_preserved(self):
        """Test that original string.split behavior is preserved for non-tuple separators"""
        # Test with None
        self.assertEqual(khandy.str_split("a b c"), ["a", "b", "c"])
        # Test with single separator
        self.assertEqual(khandy.str_split("a,b,c", ","), ["a", "b", "c"])
        # Test with maxsplit
        self.assertEqual(khandy.str_split("a,b,c,d", ",", maxsplit=1), ["a", "b,c,d"])
        
    def test_single_char_vs_tuple_with_same_char(self):
        """Compare behavior between single char separator and tuple with same char"""
        result1 = khandy.str_split("a-b-c", "-")
        result2 = khandy.str_split("a-b-c", ("-",))
        self.assertEqual(result1, result2)


class TestNormalizeDigitChars(unittest.TestCase):
    
    def test_empty_string(self):
        """Test with empty string"""
        self.assertEqual(khandy.normalize_digit_chars(""), "")
    
    def test_regular_digits_unchanged(self):
        """Test that regular digits remain unchanged"""
        self.assertEqual(khandy.normalize_digit_chars("1234567890"), "1234567890")
    
    def test_circled_numbers_basic(self):
        """Test circled numbers conversion"""
        input_str = "⓪①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
        expected = "01234567891011121314151617181920"
        self.assertEqual(khandy.normalize_digit_chars(input_str), expected)
        input_str = "➀➁➂➃➄➅➆➇➈➉"
        expected = "12345678910"
        self.assertEqual(khandy.normalize_digit_chars(input_str), expected)
        input_str = "➊➋➌➍➎➏➐➑➒➓"
        expected = "12345678910"
        self.assertEqual(khandy.normalize_digit_chars(input_str), expected)
        input_str = "⓿❶❷❸❹❺❻❼❽❾❿⓫⓬⓭⓮⓯⓰⓱⓲⓳⓴"
        expected = "01234567891011121314151617181920"
        self.assertEqual(khandy.normalize_digit_chars(input_str), expected)
        
    def test_large_circled_numbers(self):
        """Test 21-50 range (circled numbers)"""
        input_str = "㉑㉒㉓㉔㉕㉖㉗㉘㉙㉚㉛㉜㉝㉞㉟㊱㊲㊳㊴㊵㊶㊷㊸㊹㊺㊻㊼㊽㊾㊿"
        expected = "212223242526272829303132333435363738394041424344454647484950"
        self.assertEqual(khandy.normalize_digit_chars(input_str), expected)
    
    def test_double_circled_numbers(self):
        """Test double circled numbers"""
        input_str = "⓵⓶⓷⓸⓹⓺⓻⓼⓽⓾"
        expected = "12345678910"
        self.assertEqual(khandy.normalize_digit_chars(input_str), expected)
    
    def test_parenthesized_numbers(self):
        """Test parenthesized numbers"""
        input_str = "⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽⑾⑿⒀⒁⒂⒃⒄⒅⒆⒇"
        expected = "1234567891011121314151617181920"
        self.assertEqual(khandy.normalize_digit_chars(input_str), expected)
    
    def test_full_width_numbers(self):
        """Test full-width numbers"""
        input_str = "０１２３４５６７８９"
        expected = "0123456789"
        self.assertEqual(khandy.normalize_digit_chars(input_str), expected)
    
    def test_dotted_numbers(self):
        """Test dotted numbers"""
        input_str = "⒈⒉⒊⒋⒌⒍⒎⒏⒐⒑⒒⒓⒔⒕⒖⒗⒘⒙⒚⒛"
        expected = "1234567891011121314151617181920"
        self.assertEqual(khandy.normalize_digit_chars(input_str), expected)
    
    def test_mixed_content(self):
        """Test string with mixed content (numbers, letters, symbols)"""
        input_str = "Room①, Section②, Item③, Code④⑤⑥"
        expected = "Room1, Section2, Item3, Code456"
        self.assertEqual(khandy.normalize_digit_chars(input_str), expected)
    
    def test_non_digit_chars_unchanged(self):
        """Test that non-digit characters remain unchanged"""
        input_str = "Hello World! @#$%^&*()"
        expected = "Hello World! @#$%^&*()"
        self.assertEqual(khandy.normalize_digit_chars(input_str), expected)
    
    def test_mixed_regular_and_special_digits(self):
        """Test mix of regular digits and special digit formats"""
        input_str = "Version1.②.③ build④⑤⑥"
        expected = "Version1.2.3 build456"
        self.assertEqual(khandy.normalize_digit_chars(input_str), expected)
    
    def test_unicode_combination(self):
        """Test various unicode number formats together"""
        input_str = "①➁❸➍５⑸"
        expected = "123455"
        self.assertEqual(khandy.normalize_digit_chars(input_str), expected)


if __name__ == '__main__':
    unittest.main()
    