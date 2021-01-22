import os
import shutil


def get_path_stem(path):
    """
    References:
        `std::filesystem::path::stem` since C++17
    """
    return os.path.splitext(os.path.basename(path))[0]


def replace_path_stem(path, new_stem):
    dirname, basename = os.path.split(path)
    stem, extname = os.path.splitext(basename)
    if isinstance(new_stem, str):
        return os.path.join(dirname, new_stem + extname)
    elif hasattr(new_stem, '__call__'):
        return os.path.join(dirname, new_stem(stem) + extname)
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
    

def replace_path_extension(path, new_extname=None):
    """Replaces the extension with new_extname or removes it when the default value is used.
    Firstly, if this path has an extension, it is removed. Then, a dot character is appended 
    to the pathname, if new_extname is not empty or does not begin with a dot character.

    References:
        `std::filesystem::path::replace_extension` since C++17
    """
    filename_wo_ext = os.path.splitext(path)[0]
    if new_extname == '' or new_extname is None:
        return filename_wo_ext
    elif new_extname.startswith('.'):
        return ''.join([filename_wo_ext, new_extname]) 
    else:
        return '.'.join([filename_wo_ext, new_extname])


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


def get_all_filenames(dirname, extensions=None, is_valid_file=None):
    if (extensions is not None) and (is_valid_file is not None):
        raise ValueError("Both extensions and is_valid_file cannot "
                         "be not None at the same time")
    if is_valid_file is None:
        if extensions is not None:
            def is_valid_file(filename):
                return filename.lower().endswith(extensions)
        else:
            def is_valid_file(filename):
                return True

    all_filenames = []
    dirname = os.path.expanduser(dirname)
    for root, _, filenames in sorted(os.walk(dirname, followlinks=True)):
        for filename in sorted(filenames):
            path = os.path.join(root, filename)
            if is_valid_file(path):
                all_filenames.append(path)
    return all_filenames


def copy_file(src, dst_dir, action_if_exist=None):
    """
    Args:
        src: source file path
        dst_dir: dest dir
        action_if_exist: 
            None: when dest file exists, no operation
            overwritten: when dest file exists, overwritten
            rename: when dest file exists, rename it
            
    Returns:
        dest file basename
    """
    src_basename = os.path.basename(src)
    dst_fullname = os.path.join(dst_dir, src_basename)
    
    if action_if_exist is None:
        if not os.path.exists(dst_fullname):
            makedirs(dst_dir)
            shutil.copy(src, dst_dir)
    elif action_if_exist.lower() == 'overwritten':
        makedirs(dst_dir)
        # shutil.copy
        # If dst is a directory, a file with the same basename as src is 
        # created (or overwritten) in the directory specified. 
        shutil.copy(src, dst_dir)
    elif action_if_exist.lower() == 'rename':
        src_stem, src_extname = os.path.splitext(src_basename)
        suffix = 2
        while os.path.exists(dst_fullname):
            dst_basename = '{} ({}){}'.format(src_stem, suffix, src_extname)
            dst_fullname = os.path.join(dst_dir, dst_basename)
            suffix += 1
        else:
            makedirs(dst_dir)
            shutil.copy(src, dst_fullname)
    else:
        raise ValueError('Invalid action_if_exist, got {}.'.format(action_if_exist))
        
    return os.path.basename(dst_fullname)
    
    
def move_file(src, dst_dir, action_if_exist=None):
    """
    Args:
        src: source file path
        dst_dir: dest dir
        action_if_exist: 
            None: when dest file exists, no operation
            overwritten: when dest file exists, overwritten
            rename: when dest file exists, rename it
            
    Returns:
        dest file basename
    """
    src_basename = os.path.basename(src)
    dst_fullname = os.path.join(dst_dir, src_basename)
    
    if action_if_exist is None:
        if not os.path.exists(dst_fullname):
            makedirs(dst_dir)
            shutil.move(src, dst_dir)
    elif action_if_exist.lower() == 'overwritten':
        if os.path.exists(dst_fullname):
            os.remove(dst_fullname)
        makedirs(dst_dir)
        # shutil.move
        # If the destination already exists but is not a directory, 
        # it may be overwritten depending on os.rename() semantics.
        shutil.move(src, dst_dir)
    elif action_if_exist.lower() == 'rename':
        src_stem, src_extname = os.path.splitext(src_basename)
        suffix = 2
        while os.path.exists(dst_fullname):
            dst_basename = '{} ({}){}'.format(src_stem, suffix, src_extname)
            dst_fullname = os.path.join(dst_dir, dst_basename)
            suffix += 1
        else:
            makedirs(dst_dir)
            shutil.move(src, dst_fullname)
    else:
        raise ValueError('Invalid action_if_exist, got {}.'.format(action_if_exist))
        
    return os.path.basename(dst_fullname)
    

