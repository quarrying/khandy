import re


CONTENT_WITH_PAREN_PATTERN_STR_EN = r'(?:(?P<out_paren>[^(]+))?'
CONTENT_WITH_PAREN_PATTERN_STR_EN += r'(?:[(](?P<in_paren>[^)]*)[)])?'
CONTENT_WITH_PAREN_PATTERN_EN = re.compile(CONTENT_WITH_PAREN_PATTERN_STR_EN)

CONTENT_WITH_PAREN_PATTERN_STR_CN = r'(?:(?P<out_paren>[^（]+))?'
CONTENT_WITH_PAREN_PATTERN_STR_CN += r'(?:（(?P<in_paren>[^）]*)）)?'
CONTENT_WITH_PAREN_PATTERN_CN = re.compile(CONTENT_WITH_PAREN_PATTERN_STR_CN)

CONTENT_IN_PAREN_PATTERN_STR = r"\([^)]*\)|（[^）]*）"
CONTENT_IN_PAREN_PATTERN = re.compile(CONTENT_IN_PAREN_PATTERN_STR)


def split_content_with_paren(string):
    matched_en = CONTENT_WITH_PAREN_PATTERN_EN.match(string)
    outside, inside = matched_en.groups() 
    if inside is None:
        matched_cn = CONTENT_WITH_PAREN_PATTERN_CN.match(string)
        outside, inside = matched_cn.groups()
    return outside, inside


def strip_content_in_paren(string):
    """
    Notes:
        strip_content_in_paren cannot process nested paren correctly
    """
    return re.sub(CONTENT_IN_PAREN_PATTERN, "", string)


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


