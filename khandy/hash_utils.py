import hashlib
import sys

from typing import Union, BinaryIO, Protocol, runtime_checkable, overload, Literal
from pathlib import Path

if sys.version_info >= (3, 12):
    from collections.abc import Buffer
else:
    from typing_extensions import Buffer


_DEFAULT_CHUNK_SIZE = 1024 * 1024

@runtime_checkable
class HashAlgo(Protocol):
    """Structural typing for any object exposing hashlib's update / hexdigest / digest interface."""
    def update(self, data: Buffer) -> None: ...
    def hexdigest(self) -> str: ...
    def digest(self) -> bytes: ...


def _resolve_hash_object(hash_object: Union[str, HashAlgo, None]) -> HashAlgo:
    """Resolve hash_object to a hashlib-style object.

    Accepts None (defaults to MD5), an algorithm name (e.g. "sha256"), or
    any object that quacks like a hashlib hash object.
    """
    if hash_object is None:
        return hashlib.md5()
    if isinstance(hash_object, str):
        return hashlib.new(hash_object)
    if hasattr(hash_object, "update") and hasattr(hash_object, "hexdigest"):
        return hash_object
    raise TypeError(f"Invalid hash_object type, got {type(hash_object)}.")


def _update_hash(fileobj: BinaryIO, h: HashAlgo, chunk_size: int) -> None:
    """Stream data from fileobj into hash object h, using the fastest
    path available: zero-copy buffer for BytesIO, then readinto into a
    memoryview, then chunked read as a fallback.
    """
    if hasattr(fileobj, "getbuffer"):
        # consistent with hashlib.file_digest
        # io.BytesIO object, use zero-copy buffer
        h.update(fileobj.getbuffer())
        return

    if hasattr(fileobj, "readinto"):
        buf = bytearray(chunk_size)
        mv = memoryview(buf)
        while n := fileobj.readinto(mv):
            h.update(mv[:n])
        return

    while chunk := fileobj.read(chunk_size):
        h.update(chunk)


@overload
def calc_hash(
    content: Buffer,
    hash_object: Union[str, HashAlgo, None] = None,
    return_str: Literal[True] = True
) -> str: ...


@overload
def calc_hash(
    content: Buffer,
    hash_object: Union[str, HashAlgo, None] = None,
    return_str: Literal[False] = False
) -> HashAlgo: ...


def calc_hash(
    content: Buffer,
    hash_object: Union[str, HashAlgo, None] = None,
    return_str: bool = True
) -> Union[str, HashAlgo]:
    """Compute the hash of an in-memory bytes-like object.

    Args:
        content: bytes-like data to hash.
        hash_object: algorithm name (e.g. "md5", "sha256") or an existing
            hashlib-style object. Defaults to MD5.
        return_str: if True return the hex digest string; if False return
            the hash object so the caller can keep feeding it.

    Returns:
        The hex digest string, or the hash object when return_str=False.
    """
    h = _resolve_hash_object(hash_object)
    h.update(content)
    return h.hexdigest() if return_str else h


@overload
def calc_file_hash(
    file: Union[str, Path, BinaryIO],
    hash_object: Union[str, HashAlgo, None] = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    return_str: Literal[True] = True
) -> str: ...


@overload
def calc_file_hash(
    file: Union[str, Path, BinaryIO],
    hash_object: Union[str, HashAlgo, None] = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    return_str: Literal[False] = False
) -> HashAlgo: ...


def calc_file_hash(
    file: Union[str, Path, BinaryIO],
    hash_object: Union[str, HashAlgo, None] = None,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    return_str: bool = True
) -> Union[str, HashAlgo]:
    """Compute the hash of a file or already-opened binary file-like object.

    Args:
        file: filesystem path (str or pathlib.Path), or an already-opened
            binary file object.
        hash_object: algorithm name or existing hash object.
        chunk_size: read buffer size when streaming from disk. Defaults to 1 MiB.
        return_str: see calc_hash.

    Returns:
        The hex digest string, or the hash object when return_str=False.
    """
    h = _resolve_hash_object(hash_object)
    if isinstance(file, (str, Path)):
        with open(file, "rb") as f:
            _update_hash(f, h, chunk_size)
    else:
        _update_hash(file, h, chunk_size)
    return h.hexdigest() if return_str else h