import logging
import os
import re
import shutil
import warnings
from datetime import datetime
from typing import Callable, List, Literal, Optional, Union


def get_path_stem(path: str) -> str:
    """
    References:
        `std::filesystem::path::stem` since C++17
    """
    return os.path.splitext(os.path.basename(path))[0]


def replace_path_stem(path: str, new_stem: Union[str, Callable[[str], str]]) -> str:
    dirname, basename = os.path.split(path)
    stem, extension = os.path.splitext(basename)
    if isinstance(new_stem, str):
        return os.path.join(dirname, new_stem + extension)
    elif hasattr(new_stem, '__call__'):
        return os.path.join(dirname, new_stem(stem) + extension)
    else:
        raise TypeError('Unsupported Type!')
        

def get_path_extension(path: str) -> str:
    """
    References:
        `std::filesystem::path::extension` since C++17
        
    Notes:
        Not fully consistent with `std::filesystem::path::extension`
    """
    return os.path.splitext(os.path.basename(path))[1]
    

def replace_path_extension(path: str, new_extension: Optional[str] = None) -> str:
    """Replaces the extension with new_extension or removes it when the default value is used.
    Firstly, if this path has an extension, it is removed. Then, a dot character is appended 
    to the pathname, if new_extension is not empty or does not begin with a dot character.

    References:
        `std::filesystem::path::replace_extension` since C++17
    """
    path_wo_ext = os.path.splitext(path)[0]
    if new_extension == '' or new_extension is None:
        return path_wo_ext
    elif new_extension.startswith('.'):
        return f'{path_wo_ext}{new_extension}'
    else:
        return f'{path_wo_ext}.{new_extension}'
    
    
def normalize_extension(extension: str) -> str:
    if extension.startswith('.'):
        new_extension = extension.lower()
    else:
        new_extension =  '.' + extension.lower()
    return new_extension


def is_path_in_extensions(path: str, extensions: Union[str, List[str]]) -> bool:
    if isinstance(extensions, str):
        extensions = [extensions]
    extensions = [normalize_extension(item) for item in extensions]
    extension = get_path_extension(path)
    return extension.lower() in extensions


def get_path_parts(path: str, sep: Optional[Literal['/', '\\']] = None) -> List[str]:
    """Splits a file path into its constituent parts.  
  
    Args:
        path (str): The file path to be split.  
        sep (Optional[Literal['/', '\\']], optional): The separator to use for splitting the path.  
            Must be None, '/', or '\\'. Defaults to the system-specific path separator (os.path.sep).  
  
    Returns:
        List[str]: A list of strings representing the individual parts of the path.  
  
    Raises:
        ValueError: If sep is not None, '/', or '\\'.

    Notes:
        This implementation is different from pathlib.Path.parts. 
    """
    if sep is not None and sep not in ('/', '\\'):
        raise ValueError("sep must be None, '/', or '\\'")
    sep = sep or os.path.sep
    path = os.path.normpath(path)
    parts = path.split(sep)
    return parts


def normalize_path(path: str, norm_case: bool = True) -> str:
    """
    References:
        https://en.cppreference.com/w/cpp/filesystem/canonical
    """
    # On Unix and Windows, return the argument with an initial 
    # component of ~ or ~user replaced by that user's home directory.
    path = os.path.expanduser(path)
    # Return a normalized absolutized version of the pathname path. 
    # On most platforms, this is equivalent to calling the function 
    # normpath() as follows: normpath(join(os.getcwd(), path)).
    path = os.path.abspath(path)
    if norm_case:
        # Normalize the case of a pathname. On Windows, 
        # convert all characters in the pathname to lowercase, 
        # and also convert forward slashes to backward slashes. 
        # On other operating systems, return the path unchanged.
        path = os.path.normcase(path)
    return path
    

def upsert_prefix_into_path_stem(filename: str, prefix: str, validator: Optional[Callable[[str], bool]] = None, sep: str = '_') -> str:
    """Upsert a prefix into the first part of the file stem in a given file path.

    Args:
        filename (str): The full path of the file.
        prefix (str): The prefix to be inserted or updated.
        validator (Optional[Callable[[str], bool]]): An optional callable that takes a string and returns a boolean.
            If provided, it will be used to determine if the prefix should be inserted or updated.
            If the validator returns False for the first part of the stem, the prefix will be inserted.
            If the validator returns True, the first part of the stem will be replaced with the prefix.
        sep (str, optional): The separator used to split the stem. Defaults to '_'.

    Returns:
        str: The new file path with the updated stem.

    Raises:
        AssertionError: If the validator is not None and is not callable.
    """
    assert validator is None or hasattr(validator, '__call__')
    dirname, basename = os.path.split(filename)
    stem, extension = os.path.splitext(basename)
    if stem.startswith('.') and extension == '':
        stem, extension = extension, stem

    stem_parts = stem.split(sep, maxsplit=1)
    first_part = stem_parts[0]
    if first_part == '':
        stem_parts = stem_parts[1:]
        
    if validator is None:
        stem_parts.insert(0, prefix)
    elif not validator(first_part):
        stem_parts.insert(0, prefix)
    else:
        if len(stem_parts) != 0:
            stem_parts[0] = prefix
    new_basename = sep.join(stem_parts) + extension
    return os.path.join(dirname, new_basename)


def upsert_suffix_into_path_stem(filename: str, suffix: str, validator: Optional[Callable[[str], bool]] = None, sep: str = '_') -> str:
    """Upsert a suffix into the last part of the file stem in a given file path.

    Args:
        filename (str): The full path of the file.
        suffix (str): The suffix to be inserted or updated.
        validator (Optional[Callable[[str], bool]]): An optional callable function that takes a string as input
            and returns a boolean. If provided, it will be used to determine if the suffix should be inserted or updated.
            If the validator returns False for the last part of the stem, the suffix will be appended.
            If the validator returns True, the last part of the stem will be replaced with the suffix.
        sep (str, optional): The separator used to split the stem. Defaults to '_'.

    Returns:
        str: The new file path with the updated stem.

    Raises:
        AssertionError: If the validator is not None and is not callable.
    """
    assert validator is None or hasattr(validator, '__call__')
    dirname, basename = os.path.split(filename)
    stem, extension = os.path.splitext(basename)
    if stem.startswith('.') and extension == '':
        stem, extension = extension, stem
        
    stem_parts = stem.rsplit(sep, maxsplit=1)
    last_part = stem_parts[-1]
    if last_part == '':
        stem_parts = stem_parts[:-1]
    
    if validator is None:
        stem_parts.append(suffix)
    elif not validator(last_part):
        stem_parts.append(suffix)
    else:
        if len(stem_parts) != 0:
            stem_parts[-1] = suffix
    new_basename = sep.join(stem_parts) + extension
    return os.path.join(dirname, new_basename)
    

def makedirs(name, mode=0o755):
    """
    References:
        mmcv.mkdir_or_exist
    """
    warnings.warn('`makedirs` will be deprecated!')
    if name == '':
        return
    name = os.path.expanduser(name)
    os.makedirs(name, mode=mode, exist_ok=True)


def listdirs(paths, path_sep=None, full_path=True):
    """Enhancement on `os.listdir`
    """
    warnings.warn('`listdirs` will be deprecated!')
    assert isinstance(paths, (str, tuple, list))
    if isinstance(paths, str):
        path_sep = path_sep or os.path.pathsep
        paths = paths.split(path_sep)
        
    all_filenames = []
    for path in paths:
        path_ex = os.path.expanduser(path)
        filenames = os.listdir(path_ex)
        if full_path:
            filenames = [os.path.join(path_ex, filename) for filename in filenames]
        all_filenames.extend(filenames)
    return all_filenames


def get_all_filenames(path, extensions=None, is_valid_file=None):
    warnings.warn('`get_all_filenames` will be deprecated, use `list_files_in_dir` with `recursive=True` instead!')
    if (extensions is not None) and (is_valid_file is not None):
        raise ValueError("Both extensions and is_valid_file cannot "
                         "be not None at the same time")
    if is_valid_file is None:
        if extensions is not None:
            def is_valid_file(filename):
                return is_path_in_extensions(filename, extensions)
        else:
            def is_valid_file(filename):
                return True

    all_filenames = []
    path_ex = os.path.expanduser(path)
    for root, _, filenames in sorted(os.walk(path_ex, followlinks=True)):
        for filename in sorted(filenames):
            fullname = os.path.join(root, filename)
            if is_valid_file(fullname):
                all_filenames.append(fullname)
    return all_filenames


def get_top_level_dirs(path, full_path=True):
    warnings.warn('`get_top_level_dirs` will be deprecated, use `list_dirs_in_dir` instead!')
    if path is None:
        path = os.getcwd()
    path_ex = os.path.expanduser(path)
    filenames = os.listdir(path_ex)
    if full_path:
        return [os.path.join(path_ex, item) for item in filenames
                if os.path.isdir(os.path.join(path_ex, item))]
    else:
        return [item for item in filenames
                if os.path.isdir(os.path.join(path_ex, item))]


def get_top_level_files(path, full_path=True):
    warnings.warn('`get_top_level_files` will be deprecated, use `list_files_in_dir` instead!')
    if path is None:
        path = os.getcwd()
    path_ex = os.path.expanduser(path)
    filenames = os.listdir(path_ex)
    if full_path:
        return [os.path.join(path_ex, item) for item in filenames
                if os.path.isfile(os.path.join(path_ex, item))]
    else:
        return [item for item in filenames
                if os.path.isfile(os.path.join(path_ex, item))]
                

def list_items_in_dir(path: Optional[str] = None, recursive: bool = False, full_path: bool = True) -> List[str]:
    """List all entries in directory
    """
    if path is None:
        path = os.getcwd()
    path_ex = os.path.expanduser(path)
    if not os.path.exists(path_ex):
        raise FileNotFoundError(f'{path_ex} is not found!')
    elif not os.path.isdir(path_ex):
        raise NotADirectoryError(f'{path_ex} is not a directory!')
    
    if not recursive:
        names = os.listdir(path_ex)
        if full_path:
            return [os.path.join(path_ex, name) for name in sorted(names)]
        else:
            return sorted(names)
    else:
        all_names = []
        for root, dirnames, filenames in sorted(os.walk(path_ex, followlinks=True)):
            all_names += [os.path.join(root, name) for name in sorted(dirnames)]
            all_names += [os.path.join(root, name) for name in sorted(filenames)]
        return all_names


def list_dirs_in_dir(path: Optional[str] = None, recursive: bool = False, full_path: bool = True) -> List[str]:
    """List all dirs in directory
    """
    if path is None:
        path = os.getcwd()
    path_ex = os.path.expanduser(path)
    if not os.path.exists(path_ex):
        raise FileNotFoundError(f'{path_ex} is not found!')
    elif not os.path.isdir(path_ex):
        raise NotADirectoryError(f'{path_ex} is not a directory!')
    
    if not recursive:
        with os.scandir(path_ex) as it:
            if full_path:
                return sorted([entry.path for entry in it if entry.is_dir()])
            else:
                return sorted([entry.name for entry in it if entry.is_dir()])
    else:
        all_names = []
        for root, dirnames, _ in sorted(os.walk(path_ex, followlinks=True)):
            all_names += [os.path.join(root, name) for name in sorted(dirnames)]
        return all_names


def list_files_in_dir(path: Optional[str] = None, recursive: bool = False, full_path: bool = True) -> List[str]:
    """List all files in directory
    """
    if path is None:
        path = os.getcwd()
    path_ex = os.path.expanduser(path)
    if not os.path.exists(path_ex):
        raise FileNotFoundError(f'{path_ex} is not found!')
    elif not os.path.isdir(path_ex):
        raise NotADirectoryError(f'{path_ex} is not a directory!')
    
    if not recursive:
        with os.scandir(path_ex) as it:
            if full_path:
                return sorted([entry.path for entry in it if entry.is_file()])
            else:
                return sorted([entry.name for entry in it if entry.is_file()])
    else:
        all_names = []
        for root, _, filenames in sorted(os.walk(path_ex, followlinks=True)):
            all_names += [os.path.join(root, name) for name in sorted(filenames)]
        return all_names
        

def get_folder_size(path: str) -> int:
    if path is None:
        path = os.getcwd()
    path_ex = os.path.expanduser(path)
    if not os.path.exists(path_ex):
        raise FileNotFoundError(f'{path_ex} is not found!')
    elif not os.path.isdir(path_ex):
        raise NotADirectoryError(f'{path_ex} is not a directory!')
    
    total_size = 0
    for root, _, filenames in os.walk(path_ex):
        for name in filenames:
            total_size += os.path.getsize(os.path.join(root, name))
    return total_size

    
def escape_filename(filename: str, new_char: str = '_') -> str:
    assert isinstance(new_char, str)
    control_chars = ''.join((map(chr, range(0x00, 0x20))))
    pattern = r'[\\/*?:"<>|{}]'.format(control_chars)
    return re.sub(pattern, new_char, filename)


def replace_invalid_filename_char(filename, new_char='_'):
    warnings.warn('`replace_invalid_filename_char` will be deprecated, use `escape_filename` instead!')
    return escape_filename(filename, new_char)


def copy_file(src: str, dst_dir: str, action_if_exist: Optional[Literal['ignore', 'rename']] = 'rename') -> str:
    """
    Args:
        src: source file path
        dst_dir: dest dir
        action_if_exist: 
            None: same as shutil.copy
            ignore: when dest file exists, don't copy and return None
            rename: when dest file exists, copy after rename
            
    Returns:
        dest filename
    """
    dst = os.path.join(dst_dir, os.path.basename(src))

    if action_if_exist is None:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, dst)
    elif action_if_exist.lower() == 'ignore':
        if os.path.exists(dst):
            warnings.warn(f'{dst} already exists, do not copy!')
            return dst
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, dst)
    elif action_if_exist.lower() == 'rename':
        suffix = 2
        stem, extension = os.path.splitext(os.path.basename(src))
        while os.path.exists(dst):
            dst = os.path.join(dst_dir, f'{stem} ({suffix}){extension}')
            suffix += 1
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, dst)
    else:
        raise ValueError('Invalid action_if_exist, got {}.'.format(action_if_exist))
        
    return dst
    
    
def move_file(src: str, dst_dir: str, action_if_exist: Optional[Literal['ignore', 'rename']] = 'rename') -> str:
    """
    Args:
        src: source file path
        dst_dir: dest dir
        action_if_exist: 
            None: same as shutil.move
            ignore: when dest file exists, don't move and return None
            rename: when dest file exists, move after rename
            
    Returns:
        dest filename
    """
    dst = os.path.join(dst_dir, os.path.basename(src))

    if action_if_exist is None:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, dst)
    elif action_if_exist.lower() == 'ignore':
        if os.path.exists(dst):
            warnings.warn(f'{dst} already exists, do not move!')
            return dst
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, dst)
    elif action_if_exist.lower() == 'rename':
        suffix = 2
        stem, extension = os.path.splitext(os.path.basename(src))
        while os.path.exists(dst):
            dst = os.path.join(dst_dir, f'{stem} ({suffix}){extension}')
            suffix += 1
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, dst)
    else:
        raise ValueError('Invalid action_if_exist, got {}.'.format(action_if_exist))
        
    return dst
    
    
def rename_file(src: str, dst: str, action_if_exist: Optional[Literal['ignore', 'rename']] = 'rename') -> str:
    """
    Args:
        src: source file path
        dst: dest file path
        action_if_exist: 
            None: same as os.rename
            ignore: when dest file exists, don't rename and return None
            rename: when dest file exists, rename it
            
    Returns:
        dest filename
    """
    if dst == src:
        return dst
    dst_dir = os.path.dirname(os.path.abspath(dst))
    
    if action_if_exist is None:
        os.makedirs(dst_dir, exist_ok=True)
        os.rename(src, dst)
    elif action_if_exist.lower() == 'ignore':
        if os.path.exists(dst):
            warnings.warn(f'{dst} already exists, do not rename!')
            return dst
        os.makedirs(dst_dir, exist_ok=True)
        os.rename(src, dst)
    elif action_if_exist.lower() == 'rename':
        suffix = 2
        stem, extension = os.path.splitext(os.path.basename(dst))
        while os.path.exists(dst):
            dst = os.path.join(dst_dir, f'{stem} ({suffix}){extension}')
            suffix += 1
        os.makedirs(dst_dir, exist_ok=True)
        os.rename(src, dst)
    else:
        raise ValueError('Invalid action_if_exist, got {}.'.format(action_if_exist))
        
    return dst
    
    
def _get_default_logger(logger_name):
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    already_have = False
    for handler in logger.handlers[:]:
        if type(handler) == logging.StreamHandler:
            already_have = True
            break
        
    if not already_have:
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(console)
    return logger


def copy_dir(src_dir: str, dst_dir: str, 
             action_if_file_exist: Optional[Literal['ignore', 'rename']] = 'rename', 
             logger: Optional[logging.Logger] = None):
    """Recursively copies a directory and its contents from the source directory to the destination directory.  
  
    Args:  
        src_dir (str): Path to the source directory to be copied.  
        dst_dir (str): Path to the destination directory where the source directory will be copied.  
        action_if_file_exist (str, optional): Action to be taken if a file already exists in the destination directory.
            Defaults to 'rename'. Other options refers to `khandy.copy_file`.
        logger (logging.Logger, optional): A custom logger to use for logging messages. 
            If not provided, a default logger is used.
  
    Returns:
        None
  
    Raises:  
        KeyboardInterrupt: If the operation is interrupted by the user.  
        Exception: If any other exception occurs during the copying process, the exception will be caught and logged.  
    """
    if logger is None:
        logger = _get_default_logger(__name__)
    try:
        logger.info(f'{datetime.now()} MAKE {src_dir}')
        for root, dirs, files in os.walk(src_dir):
            dst_root = os.path.normpath(os.path.join(dst_dir, os.path.relpath(root, src_dir)))
            for file in files:
                copy_file(os.path.join(root, file), dst_root, action_if_file_exist)
                logger.info(f'{datetime.now()} COPY {os.path.join(root, file)} -> {dst_root}')
            for dir in dirs:
                os.makedirs(os.path.join(dst_root, dir), exist_ok=True)
                logger.info(f'{datetime.now()} MAKE {os.path.join(root, dir)}')
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        logger.error(f'{datetime.now()} ERROR: {e}')


def move_dir(src_dir: str, dst_dir: str, 
             action_if_file_exist: Optional[Literal['ignore', 'rename']] = 'rename', 
             rmtree_after_move: bool = False, logger: Optional[logging.Logger] = None):
    """Recursively moves a directory and its contents from the source directory to the destination directory.  
  
    Args:  
        src_dir (str): Path to the source directory to be moved.  
        dst_dir (str): Path to the destination directory where the source directory will be moved.  
        action_if_file_exist (str, optional): Action to be taken if a file already exists in the destination directory.  
            Defaults to 'rename'. Other options refers to `khandy.move_file`.
        rmtree_after_move (bool, optional): Whether remove src dir after move.
        logger (logging.Logger, optional): A custom logger to use for logging messages. 
            If not provided, a default logger is used. 
  
    Returns:  
        None
  
    Raises:  
        KeyboardInterrupt: If the operation is interrupted by the user.  
        Exception: If any other exception occurs during the moving process, the exception will be caught and logged.  
    """
    if logger is None:
        logger = _get_default_logger(__name__)
    try:
        for root, dirs, files in os.walk(src_dir):
            dst_root = os.path.normpath(os.path.join(dst_dir, os.path.relpath(root, src_dir)))
            for file in files:
                move_file(os.path.join(root, file), dst_root, action_if_file_exist)
                logger.info(f'{datetime.now()} MOVE {os.path.join(root, file)} -> {dst_root}')
            for dir in dirs:
                os.makedirs(os.path.join(dst_root, dir), exist_ok=True)
        if rmtree_after_move:
            shutil.rmtree(src_dir)
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        logger.error(f'{datetime.now()} ERROR: {e}')
        

def rename_files(old_names: List[str], new_names: List[str], 
                 action_if_file_exist: Optional[Literal['ignore', 'rename']] = 'rename', 
                 logger: Optional[logging.Logger] = None):
    """Renames a list of files based on provided old and new file name pairs.  
    
    Args:
        old_names (List[str]): A list of old file names to be renamed.  
        new_names (List[str]): A list of new file names corresponding to the old file names.  
        action_if_file_exist (str, optional): Specifies the action to be taken if a new file already exists.  
            Defaults to 'rename'. Other options refers to `khandy.rename_file`.
        logger (logging.Logger, optional): A custom logger for recording log messages. 
            If not provided, a default logger is used. 
  
    Returns:
        None
  
    Raises:
        KeyboardInterrupt: If the operation is interrupted by the user.  
        Exception: If any other exception occurs during the renaming process, the exception will be caught and logged.  
    """
    if logger is None:
        logger = _get_default_logger(__name__)
    try:
        for k, (old_name, new_name) in enumerate(zip(old_names, new_names)):
            if new_name == old_name:
                logger.info(f'{datetime.now()} [{k+1}/{len(old_names)}] KEEP SAME: {old_name}')
                continue
            new_name = rename_file(old_name, new_name, action_if_file_exist)
            logger.info(f'{datetime.now()} [{k+1}/{len(old_names)}] RENAME {old_name} -> {new_name}')
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        logger.error(f'{datetime.now()} ERROR: {e}')
