import re
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

if sys.version_info >= (3, 8):
    from typing import Literal, SupportsIndex
else:
    from typing_extensions import Literal, SupportsIndex

CONTENT_WITH_HW_PAREN_PATTERN = r'(?:(?P<out_paren>[^(]+))?'
CONTENT_WITH_HW_PAREN_PATTERN += r'(?:[(](?P<in_paren>[^)]*)[)])?'
CONTENT_WITH_HW_PAREN_PATTERN_OBJ = re.compile(CONTENT_WITH_HW_PAREN_PATTERN)

CONTENT_WITH_FW_PAREN_PATTERN = r'(?:(?P<out_paren>[^（]+))?'
CONTENT_WITH_FW_PAREN_PATTERN += r'(?:（(?P<in_paren>[^）]*)）)?'
CONTENT_WITH_FW_PAREN_PATTERN_OBJ = re.compile(CONTENT_WITH_FW_PAREN_PATTERN)

CONTENT_IN_PAREN_PATTERN = r"\([^)]*\)|（[^）]*）"
CONTENT_IN_PAREN_PATTERN_OBJ = re.compile(CONTENT_IN_PAREN_PATTERN)


def has_nested_or_unmatched_paren(
    string: str, 
    paren_type: Literal['hw', 'fw'] = 'hw'
) -> bool:
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


def split_content_with_paren(
    string: str, 
    paren_type: Literal['hw', 'fw'] = 'hw'
) -> Tuple[Optional[str], Optional[str]]:
    """Split a string into content outside and inside parentheses.
    
    Args:
        string: Input string to be processed
        paren_type: Type of parentheses to handle, either:
            - 'hw' for half-width parentheses: ()
            - 'fw' for full-width parentheses: （）
            Defaults to 'hw'
            
    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple containing two optional string elements.
            The first element is the part outside the parentheses (or None if not found).
            The second element is the part inside the parentheses (or None if not found).
        
    Raises:
        AssertionError: If invalid paren_type is provided
        ValueError: If string contains:
            - Nested/unmatched parentheses
            - Fails to match the expected pattern
    """
    if paren_type not in ('hw', 'fw'):
        raise AssertionError(f"Paren type must be either 'hw' or 'fw', got {paren_type}.")
    
    pattern_obj = {
        'hw': CONTENT_WITH_HW_PAREN_PATTERN_OBJ,
        'fw': CONTENT_WITH_FW_PAREN_PATTERN_OBJ
    }[paren_type]

    if has_nested_or_unmatched_paren(string, paren_type):
        raise ValueError(f'nested or unmatched paren: "{string}"')
    matched = pattern_obj.fullmatch(string)
    if matched is None:
        raise ValueError(f'parse failure: "{string}"')
    outside, inside = matched.groups()
    return outside, inside


def split_before_after(
    text: str,
    strip_whitespace: bool = True,
    sep: str = '$'
) -> Tuple[Optional[str], Optional[str]]:
    """Split a string into two parts by a separator.

    Args:
        text (str): The input string to split.
        strip_whitespace (bool, optional): Whether to strip whitespace from both parts. Defaults to True.
        sep (str, optional): The separator to split on. Defaults to '$'.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple (before, after). If a part is empty, it will be None.

    Raises:
        ValueError: If the string ends with the separator, or contains more than one separator.
    """
    if strip_whitespace:
        text = text.strip()
        
    if text.endswith(sep):
        raise ValueError(f'cannot end with "{sep}": {text}')
    parts = text.split(sep)
    if len(parts) > 2:
        raise ValueError(f'cannot contain more than one "{sep}": {text}')

    before_part = parts[0]
    after_part = parts[1] if len(parts) == 2 else ''
    
    if strip_whitespace:
        before_part = before_part.strip()
        after_part = after_part.strip()
    if before_part == '':
        before_part = None
    if after_part == '':
        after_part = None
    return before_part, after_part


def strip_content_in_paren(string: str) -> str:
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
        `is_chinese_char` in https://github.com/loveminimal/rime-jk
        https://en.wikipedia.org/wiki/CJK_Unified_Ideographs
    """
    codepoint = ord(uchar)
    cjk_ranges = [
        (0x4E00, 0x9FFF),   # CJK Unified Ideographs
        (0x3400, 0x4DBF),   # CJK Unified Ideographs Extension A
        (0x20000, 0x2A6DF), # CJK Unified Ideographs Extension B
        (0x2A700, 0x2B73F), # CJK Unified Ideographs Extension C
        (0x2B740, 0x2B81F), # CJK Unified Ideographs Extension D
        (0x2B820, 0x2CEAF), # CJK Unified Ideographs Extension E
        (0x2CEB0, 0x2EBEF), # CJK Unified Ideographs Extension F
        (0x30000, 0x3134F), # CJK Unified Ideographs Extension G
        (0x31350, 0x323AF), # CJK Unified Ideographs Extension H
        (0x2EBF0, 0x2EE5F), # CJK Unified Ideographs Extension I
        (0x323B0, 0x33479), # CJK Unified Ideographs Extension J
        (0xF900, 0xFAFF),   # CJK Compatibility Ideographs
        (0x2F800, 0x2FA1F)  # CJK Compatibility Ideographs Supplement
    ]
    return any(start <= codepoint <= end for start, end in cjk_ranges)


def strip_blank_lines(lines: List[str]) -> List[str]:
    """Strip leading and trailing blank lines from a list of lines.

    Args:
        lines (list[str]): A list of lines to process.

    Returns:
        list[str]: A new list containing the lines with leading and trailing blank lines removed.
    """
    start_index, end_index = None, None
    for k, line in enumerate(lines):
        if line.strip() != '':
            start_index = k
            break
    if start_index is None:
        return []
    for k, line in enumerate(lines[::-1]):
        if line.strip() != '':
            end_index = len(lines) - k
            break
    return lines[start_index: end_index]


class MarkdownTableAlignType(Enum):
    DEFAULT = 0
    LEFT = 1
    RIGHT = 2
    BOTH = 3


@dataclass
class MarkdownTable:
    headers: List[str]
    align_types: List[MarkdownTableAlignType]
    rows: List[List[str]]


def _get_cell_align_type(cell: str) -> MarkdownTableAlignType:
    cell = re.sub(r'\s', '', cell)
    cell = re.sub(r'-{2,}', '-', cell)
    if cell == '-':
        return MarkdownTableAlignType.DEFAULT
    elif cell == ':-':
        return MarkdownTableAlignType.LEFT
    elif cell == '-:':
        return MarkdownTableAlignType.RIGHT
    elif cell == ':-:':
        return MarkdownTableAlignType.BOTH
    else:
        raise Exception('parse align type')


def _split_table_line(line: str, length: Optional[int] = None) -> List[str]:
    line = line.strip('|')
    cells = line.split('|')
    cells = [cell.strip() for cell in cells]
    if length is not None and length > len(cells):
        cells += ['' for _ in range(length - len(cells))]
    return cells


def parse_markdown_table(lines: List[str]) -> MarkdownTable:
    """Parse a markdown table from a list of lines.
  
    Args:
        lines (List[str]): A list of lines containing the markdown table.
  
    Returns:
        MarkdownTable: A parsed markdown table object.
  
    Raises:
        Exception: If a blank line is encountered.
        Exception: If the number of lines is too short.
        Exception: If the number of cells in the header line does not match the number of cells in the splitter line.
        Exception: If the number of cells in a row does not match the number of cells in the header line.
    """
    lines = strip_blank_lines(lines)
    for line in lines:
        if line.strip() == '':
            raise Exception('blank line')
    if len(lines) < 2:
        raise Exception('#line too short')

    headers = _split_table_line(lines[0])
    splitters = _split_table_line(lines[1])
    if len(splitters) != len(headers):
        raise Exception('#cell unmatched')
    align_types = [_get_cell_align_type(item) for item in splitters]

    rows = []
    for line in lines[2:]:
        parts = _split_table_line(line, len(headers))
        if len(parts) != len(headers):
            raise Exception('#cell unmatched')
        rows.append(parts)
        
    return MarkdownTable(headers, align_types, rows)


def dumps_markdown_table(table: MarkdownTable, align_header: bool = False) -> List[str]:
    """Convert a khandy.MarkdownTable object to a list of strings representing a Markdown table.
  
    Args:
        table (khandy.MarkdownTable): The Markdown table object to be converted.
        align_header (bool, optional): Whether to align the table header to the specified lengths. Defaults to False.
  
    Returns:
        List[str]: A list of strings representing the Markdown table.
    """
    min_length = 5 
    lengths = [max(min_length, len(header)) for header in table.headers]
    for row in table.rows:
        lengths = [max(len(cell), lengths[k]) for k, cell in enumerate(row)]

    if align_header:
        header_cells = [f'{header}'.ljust(lengths[k]) for k, header in enumerate(table.headers)]
    else:
        header_cells = table.headers

    splitter_cells = []
    for k, align_type in enumerate(table.align_types):
        if align_type == MarkdownTableAlignType.DEFAULT:
            cell = f'{"-" * lengths[k]}'
        elif align_type == MarkdownTableAlignType.LEFT:
            cell = f':{"-" * (lengths[k] - 2)}-'
        elif align_type == MarkdownTableAlignType.RIGHT:
            cell = f'-{"-" * (lengths[k] - 2)}:'
        elif align_type == MarkdownTableAlignType.BOTH:
            cell = f':{"-" * (lengths[k] - 2)}:'
        splitter_cells.append(cell)

    cells_list = [header_cells, splitter_cells]
    for row in table.rows:
        cells_list.append([f'{cell}'.ljust(lengths[k]) for k, cell in enumerate(row)])
    dst_lines = [' | '.join(cells) for cells in cells_list]
    dst_lines = [line.strip() for line in dst_lines]
    return dst_lines


def parse_range_string(
    range_string: str, 
    sep: str = ',', 
    range_sep: str = '-'
) -> List[int]:
    """Parse a string representing a range of numbers and return a sorted list of integers.
    
    Args:
        range_string: A string containing comma-separated numbers or ranges (e.g., "1,3-5,7")
        sep: Separator for different parts in the range string, default is ','
        range_sep: Separator used to denote ranges, default is '-'
        
    Returns:
        A sorted list of unique integers parsed from the input string
        
    Raises:
        ValueError: If a range part is invalid (doesn't contain exactly two parts separated by range_sep)
    """
    circle_digits = '⓪①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳'
    circle_digit_map = {digit: str(k) for k, digit in enumerate(circle_digits)}
    for old_str, new_str in circle_digit_map.items():
        range_string = range_string.replace(old_str, new_str)

    result = []
    for part in range_string.split(sep):
        if range_sep in part:
            subparts = part.split(range_sep)
            if len(subparts) != 2:
                raise ValueError(f"Invalid range part: {part}")
            start, end = map(int, subparts)
            result.extend(range(start, end + 1))
        else:
            if part.strip() != '':
                result.append(int(part))
    result = sorted(set(result))
    return result


def str_contains(
    s: str, 
    sub: Union[str, Tuple[str, ...]], 
    start: Optional[SupportsIndex] = None, 
    end: Optional[SupportsIndex] = None, 
) -> bool:
    """Check if a string contains a substring or any of a tuple of substrings.

    Args:
        s: The string to search.
        sub: The substring or tuple of substrings to search for.
        start: The start index (inclusive) to search from.
        end: The end index (exclusive) to search to.

    Returns:
        True if the string contains the substring or any of the substrings in the tuple, False otherwise.

    Raises:
        TypeError: If the sub argument is not a string or tuple.
        TypeError: If any item in the tuple is not a string.

    Notes:
        This function simulates str.startswith and str.endswith with the ability to check for multiple substrings at once.
    """
    if isinstance(sub, str):
        return s.find(sub, start, end) != -1
    if isinstance(sub, tuple):
        for item in sub:
            if not isinstance(item, str):
                raise TypeError(f"expected str, not {type(item).__name__}")
        return any(s.find(item, start, end) != -1 for item in sub)
    raise TypeError(f"str_contains arg 1 must be str or tuple, not {type(sub).__name__}")
