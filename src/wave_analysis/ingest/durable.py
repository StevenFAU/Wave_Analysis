"""Writes that survive a power cut.

Writing to a temporary file and renaming it into place makes a write *atomic*
(no reader sees half a file) but not *durable*. ext4 commits the rename with
its journal, while the file's data may still be in memory. After a power cut
the file can then come back empty, or with the right size and only NUL bytes.
This happened to the archive on 2026-10-18 and 2026-11-03
(``docs/operations/data_collection.md``, incidents). The cure is to
``fsync`` the data before the rename, and the directory after it.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import IO, Any


def fsync_dir(path: str | Path) -> None:
    """Flush a directory's entries (a rename or a new file in it) to disk."""
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:  # pragma: no cover - platforms that cannot open directories
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - e.g. some network file systems
        pass
    finally:
        os.close(fd)


def fsync_file(fh: IO[Any]) -> None:
    """Flush Python's buffer and the kernel's cache of an open file to disk."""
    fh.flush()
    os.fsync(fh.fileno())


@contextlib.contextmanager
def atomic_open(path: str | Path, mode: str = "w", **kwargs: Any) -> Iterator[IO[Any]]:
    """Open a temporary file next to ``path``; on success sync it and rename it into place.

    ``mode`` is ``"w"`` or ``"wb"`` (plus any ``open`` keyword such as
    ``newline``). If the block raises, the temporary file is removed and
    ``path`` is left as it was.
    """
    if mode not in ("w", "wb"):
        raise ValueError(f"atomic_open writes whole files; mode {mode!r} is not supported")
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=dest.parent, prefix=".partial-")
    tmp = Path(name)
    try:
        if "b" not in mode:
            kwargs.setdefault("encoding", "utf-8")
        with os.fdopen(fd, mode, **kwargs) as fh:
            yield fh
            fsync_file(fh)
        os.replace(tmp, dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    fsync_dir(dest.parent)
