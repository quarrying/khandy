import re


def strip_content_in_paren(string):
    """
    Notes:
        strip_content_in_paren cannot process nested paren correctly
    """
    return re.sub(r"\([^)]*\)|（[^）]*）", "", string)


def is_chinese_char(uchar: str) -> bool:
    """Whether the input char is a Chinese character.

    Args:
        uchar: input char in unicode

    References:
        `is_chinese_char` in https://github.com/thunlp/OpenNRE/
    """
    codepoint = ord(uchar)
    if ((codepoint >= 0x4E00 and codepoint <= 0x9FFF) or # CJK Unified Ideographs
        (codepoint >= 0x3400 and codepoint <= 0x4DBF) or # CJK Unified Ideographs Extension A
        (codepoint >= 0xF900 and codepoint <= 0xFAFF) or # CJK Compatibility Ideographs
        (codepoint >= 0x20000 and codepoint <= 0x2A6DF) or # CJK Unified Ideographs Extension B
        (codepoint >= 0x2A700 and codepoint <= 0x2B73F) or
        (codepoint >= 0x2B740 and codepoint <= 0x2B81F) or
        (codepoint >= 0x2B820 and codepoint <= 0x2CEAF) or
        (codepoint >= 0x2F800 and codepoint <= 0x2FA1F)): # CJK Compatibility Supplement
        return True
    return False


