import hashlib


def calc_hash(content, hash_func=None):
    hash_func = hash_func or hashlib.md5()
    if isinstance(hash_func, str):
        hash_func = hashlib.new(hash_func)
    hash_func.update(content)
    return hash_func.hexdigest()


def calc_file_hash(filename, hash_func=None, chunk_size=1024 * 1024):
    hash_func = hash_func or hashlib.md5()
    if isinstance(hash_func, str):
        hash_func = hashlib.new(hash_func)
    
    with open(filename, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hash_func.update(chunk)
    return hash_func.hexdigest()

    