"""Random access into corpus files without holding them in memory.

Lazy encoding bounded the cost of tensors, but parsed documents were still
materialised in full. At roughly ten kilobytes each, the eight-language training
split would need about 57 GB of them, which no ordinary machine has.

The index records where each record begins and nothing else. Eight bytes per
record replaces ten kilobytes, so the same split costs tens of megabytes, and
the record itself is read from disk only when the trainer asks for it.

Building the index requires reading every file once. That pass is also where the
relation inventory and the corpus statistics are collected, so the cost is paid
once rather than once per purpose.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator


@dataclass(slots=True)
class RecordLocation:
    """Where one record lives.

    Slots matter here rather than being a stylistic choice: the index holds one
    of these per record, and at several million records the per-object dictionary
    a normal class carries would cost more than the offsets it stores.

    Attributes:
        path: File containing the record.
        offset: Byte offset of the record's first byte.
    """

    path: Path
    offset: int


class RecordIndex:
    """Random access to records located by byte offset.

    File handles are kept open between reads. Reopening a multi-gigabyte file
    for every training example would dominate the epoch.

    Args:
        locations: Where each record begins, in the order they should be served.
        reader: Reads one record from an open handle positioned at its start.
    """

    def __init__(
        self,
        locations: list[RecordLocation],
        reader: Callable[[object], dict | None],
    ) -> None:
        self._locations = locations
        self._reader = reader
        self._handles: dict[Path, object] = {}

    def __len__(self) -> int:
        """Number of records the index can serve."""
        return len(self._locations)

    def read(self, index: int) -> dict | None:
        """Read one record.

        Args:
            index: Position in the index.

        Returns:
            The decoded record, or ``None`` when it cannot be decoded.
        """
        location = self._locations[index]
        handle = self._handles.get(location.path)
        if handle is None:
            handle = location.path.open("rb")
            self._handles[location.path] = handle
        handle.seek(location.offset)
        return self._reader(handle)

    def close(self) -> None:
        """Release every open file handle."""
        for handle in self._handles.values():
            handle.close()
        self._handles.clear()

    def __del__(self) -> None:
        """Release handles when the index is discarded."""
        try:
            self.close()
        except Exception:
            pass


def read_json_line(handle) -> dict | None:
    """Read one JSON Lines record from a positioned handle.

    Args:
        handle: Binary file handle positioned at the start of a line.

    Returns:
        The decoded record, or ``None`` when the line is not valid JSON.
    """
    line = handle.readline()
    if not line:
        return None
    try:
        return json.loads(line.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def scan_json_lines(path: Path, limit: int | None = None) -> Iterator[tuple[int, dict]]:
    """Yield the offset and content of every record in a JSON Lines file.

    Args:
        path: File to scan.
        limit: Optional cap on the number of records yielded.

    Yields:
        Byte offset and decoded record.
    """
    emitted = 0
    with path.open("rb") as handle:
        while True:
            if limit is not None and emitted >= limit:
                return
            offset = handle.tell()
            line = handle.readline()
            if not line:
                return
            if not line.strip():
                continue
            try:
                record = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            emitted += 1
            yield offset, record

