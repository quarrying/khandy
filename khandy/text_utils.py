import re


CONTENT_WITH_HW_PAREN_PATTERN = r'(?:(?P<out_paren>[^(]+))?'
CONTENT_WITH_HW_PAREN_PATTERN += r'(?:[(](?P<in_paren>[^)]*)[)])?'
CONTENT_WITH_HW_PAREN_PATTERN_OBJ = re.compile(CONTENT_WITH_HW_PAREN_PATTERN)

CONTENT_WITH_FW_PAREN_PATTERN = r'(?:(?P<out_paren>[^（]+))?'
CONTENT_WITH_FW_PAREN_PATTERN += r'(?:（(?P<in_paren>[^）]*)）)?'
CONTENT_WITH_FW_PAREN_PATTERN_OBJ = re.compile(CONTENT_WITH_FW_PAREN_PATTERN)

CONTENT_IN_PAREN_PATTERN = r"\([^)]*\)|（[^）]*）"
CONTENT_IN_PAREN_PATTERN_OBJ = re.compile(CONTENT_IN_PAREN_PATTERN)


def split_content_with_paren(string):
    matched_en = CONTENT_WITH_HW_PAREN_PATTERN_OBJ.fullmatch(string)
    if matched_en is not None:
        outside, inside = matched_en.groups()
        if inside is None:
            matched_cn = CONTENT_WITH_FW_PAREN_PATTERN_OBJ.fullmatch(string)
            if matched_cn is not None:
                outside, inside = matched_cn.groups() 
            else:
                outside, inside = None, None
    else:
        outside, inside = None, None
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


