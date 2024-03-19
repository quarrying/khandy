import re
from typing import Tuple, Optional


CONTENT_WITH_HW_PAREN_PATTERN = r'(?:(?P<out_paren>[^(]+))?'
CONTENT_WITH_HW_PAREN_PATTERN += r'(?:[(](?P<in_paren>[^)]*)[)])?'
CONTENT_WITH_HW_PAREN_PATTERN_OBJ = re.compile(CONTENT_WITH_HW_PAREN_PATTERN)

CONTENT_WITH_FW_PAREN_PATTERN = r'(?:(?P<out_paren>[^（]+))?'
CONTENT_WITH_FW_PAREN_PATTERN += r'(?:（(?P<in_paren>[^）]*)）)?'
CONTENT_WITH_FW_PAREN_PATTERN_OBJ = re.compile(CONTENT_WITH_FW_PAREN_PATTERN)

CONTENT_IN_PAREN_PATTERN = r"\([^)]*\)|（[^）]*）"
CONTENT_IN_PAREN_PATTERN_OBJ = re.compile(CONTENT_IN_PAREN_PATTERN)


def has_nested_or_unmatched_paren(string: str, paren_type: str = 'hw') -> bool:
    """Check if a string contains nested or unmatched parentheses.
  
    Args:
        string (str): The input string to be checked.
        paren_type (str): The type of parentheses to be used. Options are 'fw' (full-width) and 'hw' (half-width). Defaults to 'hw'.
  
    Returns:
        bool: True if the string contains nested or unmatched parentheses, False otherwise.
  
    Raises:
        ValueError: If paren_type is not one of 'fw' or 'hw'.
    """  
    if paren_type == 'fw': # full-width
        left_char, right_char = '（）'
    elif paren_type == 'hw': # half-width
        left_char, right_char = '()'
    else:
        raise ValueError(f'paren_type only support fw and hw, got {paren_type}')
    
    stack = []
    for char in string:
        if char == left_char:
            # If stack is not empty, we have a nested parenthesis
            if len(stack) != 0:  
                return True
            stack.append(char)
        elif char == right_char:
            # If stack is empty, we have an unmatched parenthesis
            if len(stack) == 0:  
                return True
            stack.pop()
    # If stack is not empty, we have unmatched opening parentheses
    return len(stack) != 0


def split_content_with_paren(string: str, paren_type: str = 'hw') -> Tuple[Optional[str], Optional[str]]:
    """Split a string into two parts based on the presence of parentheses of a specified type.
  
    Args:
        string (str): The input string to be split.
        paren_type (str, optional): The type of parentheses to split on. Defaults to 'hw' (half-width parentheses).
            Accepted values are 'hw' for half-width parentheses and 'fw' for full-width parentheses.
  
    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing two optional string elements.
            The first element is the part outside the parentheses (or None if not found).
            The second element is the part inside the parentheses (or None if not found).
  
    Raises:  
        AssertionError: If the `paren_type` is not one of 'hw' or 'fw'.
    """ 
    assert paren_type in ('hw', 'fw'), f"Paren type must be either 'hw' or 'fw', got {paren_type}."
    pattern_obj = {
        'hw': CONTENT_WITH_HW_PAREN_PATTERN_OBJ,
        'fw': CONTENT_WITH_FW_PAREN_PATTERN_OBJ
    }[paren_type]

    if has_nested_or_unmatched_paren(string, 'hw'):
        return None, None
    matched = pattern_obj.fullmatch(string)
    if matched is None:
        return None, None
    outside, inside = matched.groups()
    return outside, inside
    

def strip_content_in_paren(string):
    """
    Notes:
        strip_content_in_paren cannot process nested paren correctly
    """
    return re.sub(CONTENT_IN_PAREN_PATTERN_OBJ, "", string)


def is_chinese_char(uchar: str) -> bool:
    """Whether the input char is a Chinese character.

    Args:
        uchar: input char in unicode

    References:
        `is_chinese_char` in https://github.com/thunlp/OpenNRE/
    """
    codepoint = ord(uchar)
    if ((0x4E00 <= codepoint <= 0x9FFF) or # CJK Unified Ideographs
        (0x3400 <= codepoint <= 0x4DBF) or # CJK Unified Ideographs Extension A
        (0xF900 <= codepoint <= 0xFAFF) or # CJK Compatibility Ideographs
        (0x20000 <= codepoint <= 0x2A6DF) or # CJK Unified Ideographs Extension B
        (0x2A700 <= codepoint <= 0x2B73F) or
        (0x2B740 <= codepoint <= 0x2B81F) or
        (0x2B820 <= codepoint <= 0x2CEAF) or
        (0x2F800 <= codepoint <= 0x2FA1F)): # CJK Compatibility Supplement
        return True
    return False


