import os
import re
import shutil


def get_path_stem(path):
    """
    References:
        `std::filesystem::path::stem` since C++17
    """
    return os.path.splitext(os.path.basename(path))[0]


def replace_path_stem(path, new_stem):
    dirname, basename = os.path.split(path)
    stem, extension = os.path.splitext(basename)
    if isinstance(new_stem, str):
        return os.path.join(dirname, new_stem + extension)
    elif hasattr(new_stem, '__call__'):
        return os.path.join(dirname, new_stem(stem) + extension)
    else:
        raise ValueError('Unsupported Type!')
        

def get_path_extension(path):
    """
    References:
        `std::filesystem::path::extension` since C++17
        
    Notes:
        Not fully consistent with `std::filesystem::path::extension`
    """
    return os.path.splitext(os.path.basename(path))[1]
    

def replace_path_extension(path, new_extension=None):
    """Replaces the extension with new_extension or removes it when the default value is used.
    Firstly, if this path has an extension, it is removed. Then, a dot character is appended 
    to the pathname, if new_extension is not empty or does not begin with a dot character.

    References:
        `std::filesystem::path::replace_extension` since C++17
    """
    filename_wo_ext = os.path.splitext(path)[0]
    if new_extension == '' or new_extension is None:
        return filename_wo_ext
    elif new_extension.startswith('.'):
        return ''.join([filename_wo_ext, new_extension]) 
    else:
        return '.'.join([filename_wo_ext, new_extension])


def normalize_extension(extension):
    if extension.startswith('.'):
        new_extension = extension.lower()
    else:
        new_extension =  '.' + extension.lower()
    return new_extension


def is_path_in_extensions(path, extensions):
    if isinstance(extensions, str):
        extensions = [extensions]
    extensions = [normalize_extension(item) for item in extensions]
    extension = get_path_extension(path)
    return extension.lower() in extensions


def normalize_path(path, norm_case=True):
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
    

def makedirs(name, mode=0o755):
    """
    References:
        mmcv.mkdir_or_exist
    """
    if name == '':
        return
    name = os.path.expanduser(name)
    os.makedirs(name, mode=mode, exist_ok=True)


def listdirs(paths, path_sep=None, full_path=True):
    """Enhancement on `os.listdir`
    """
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
                

def get_folder_size(dirname):
    if not os.path.exists(dirname):
        raise ValueError("Incorrect path: {}".format(dirname))
    total_size = 0
    for root, _, filenames in os.walk(dirname):
        for name in filenames:
            total_size += os.path.getsize(os.path.join(root, name))
    return total_size

    
def replace_invalid_filename_char(filename, new_char='_'):
    assert isinstance(new_char, str)
    control_chars = ''.join((map(chr, range(0x00, 0x20))))
    pattern = r'[\\/*?:"<>|{}]'.format(control_chars)
    return re.sub(pattern, new_char, filename)


def copy_file(src, dst_dir, action_if_exist='rename'):
    """
    Args:
        src: source file path
        dst_dir: dest dir
        action_if_exist: 
            None: same as shutil.copy
            rename: when dest file exists, rename it
            
    Returns:
        dest filename
    """
    src_basename = os.path.basename(src)
    src_stem, src_extension = os.path.splitext(src_basename)
    dst = os.path.join(dst_dir, src_basename)
    
    if action_if_exist is None:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, dst_dir)
    elif action_if_exist.lower() == 'rename':
        suffix = 2
        while os.path.exists(dst):
            dst_basename = '{} ({}){}'.format(src_stem, suffix, src_extension)
            dst = os.path.join(dst_dir, dst_basename)
            suffix += 1
        else:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy(src, dst)
    else:
        raise ValueError('Invalid action_if_exist, got {}.'.format(action_if_exist))
        
    return dst
    
    
def move_file(src, dst_dir, action_if_exist='rename'):
    """
    Args:
        src: source file path
        dst_dir: dest dir
        action_if_exist: 
            None: same as shutil.move
            rename: when dest file exists, rename it
            
    Returns:
        dest filename
    """
    src_basename = os.path.basename(src)
    src_stem, src_extension = os.path.splitext(src_basename)
    dst = os.path.join(dst_dir, src_basename)
    
    if action_if_exist is None:
        os.makedirs(dst_dir, exist_ok=True)
        shutil.move(src, dst_dir)
    elif action_if_exist.lower() == 'rename':
        suffix = 2
        while os.path.exists(dst):
            dst_basename = '{} ({}){}'.format(src_stem, suffix, src_extension)
            dst = os.path.join(dst_dir, dst_basename)
            suffix += 1
        else:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, dst)
    else:
        raise ValueError('Invalid action_if_exist, got {}.'.format(action_if_exist))
        
    return dst
    
    
def rename_file(src, dst, action_if_exist='rename'):
    """
    Args:
        src: source file path
        dst: dest file path
        action_if_exist: 
            None: same as os.rename
            rename: when dest file exists, rename it
            
    Returns:
        dest filename
    """
    if dst == src:
        return dst
        
    if action_if_exist is None:
        os.rename(src, dst)
    elif action_if_exist.lower() == 'rename':
        dirname, basename = os.path.split(dst)
        stem, extension = os.path.splitext(basename)
        suffix = 2
        while os.path.exists(dst):
            new_basename = '{} ({}){}'.format(stem, suffix, extension)
            dst = os.path.join(dirname, new_basename)
            suffix += 1
        os.makedirs(dirname, exist_ok=True)
        os.rename(src, dst)
    else:
        raise ValueError('Invalid action_if_exist, got {}.'.format(action_if_exist))
        
    return dst
    
    