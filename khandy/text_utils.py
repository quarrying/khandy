import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

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

