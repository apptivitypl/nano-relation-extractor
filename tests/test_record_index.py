"""Tests for random access into corpus files.

The index exists so that a corpus larger than memory can be trained on. Two
things must hold for that to work: a record read through the index must be the
same record streaming would have produced, and a limit must be applied while
scanning rather than afterwards. Applying it afterwards still reads every byte,
which on a thirty gigabyte corpus is the whole problem.
"""

from __future__ import annotations

import json
from pathlib import Path

from nano_re.data.record_index import (
    RecordIndex,
    RecordLocation,
    read_json_line,
    scan_json_lines,
)


def _write_jsonl(directory: Path, records: list[dict]) -> Path:
    """Write records as JSON Lines.

    Args:
        directory: Directory to write into.
        records: Records to serialise.

    Returns:
        The written file.
    """
    path = directory / "corpus.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return path


def test_scan_returns_every_record_with_its_offset(tmp_path: Path) -> None:
    """Scanning yields each record once, in file order."""
    records = [{"id": index} for index in range(5)]
    path = _write_jsonl(tmp_path, records)
    scanned = list(scan_json_lines(path))
    assert [record["id"] for _, record in scanned] == [0, 1, 2, 3, 4]
    assert scanned[0][0] == 0
    assert all(later[0] > earlier[0] for earlier, later in zip(scanned, scanned[1:]))


def test_scan_stops_at_the_limit(tmp_path: Path) -> None:
    """A limit truncates the scan rather than the result."""
    path = _write_jsonl(tmp_path, [{"id": index} for index in range(100)])
    assert len(list(scan_json_lines(path, limit=3))) == 3


def test_index_reads_the_same_records_the_scan_saw(tmp_path: Path) -> None:
    """Random access agrees with sequential reading."""
    records = [{"id": index, "text": f"document {index}"} for index in range(20)]
    path = _write_jsonl(tmp_path, records)
    locations = [
        RecordLocation(path=path, offset=offset)
        for offset, _ in scan_json_lines(path)
    ]
    index = RecordIndex(locations, read_json_line)

    assert len(index) == 20
    for position in (0, 7, 19):
        assert index.read(position) == records[position]
    index.close()


def test_index_reads_are_repeatable(tmp_path: Path) -> None:
    """Reading the same position twice returns the same record."""
    path = _write_jsonl(tmp_path, [{"id": index} for index in range(10)])
    locations = [
        RecordLocation(path=path, offset=offset)
        for offset, _ in scan_json_lines(path)
    ]
    index = RecordIndex(locations, read_json_line)
    assert index.read(4) == index.read(4)
    assert index.read(9)["id"] == 9
    assert index.read(0)["id"] == 0
    index.close()


def test_scan_skips_malformed_lines(tmp_path: Path) -> None:
    """A corrupt line does not abort a pass over a generated corpus."""
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"id": 1}\nnot json\n{"id": 2}\n', encoding="utf-8")
    assert [record["id"] for _, record in scan_json_lines(path)] == [1, 2]


def test_scan_skips_blank_lines(tmp_path: Path) -> None:
    """Blank lines are not records."""
    path = tmp_path / "corpus.jsonl"
    path.write_text('{"id": 1}\n\n{"id": 2}\n', encoding="utf-8")
    assert len(list(scan_json_lines(path))) == 2


def test_reading_past_the_end_returns_nothing(tmp_path: Path) -> None:
    """A location at the end of the file yields no record."""
    path = _write_jsonl(tmp_path, [{"id": 1}])
    index = RecordIndex(
        [RecordLocation(path=path, offset=path.stat().st_size)], read_json_line
    )
    assert index.read(0) is None
    index.close()


def test_locations_use_slots() -> None:
    """The per-record object carries no instance dictionary.

    At several million records the dictionary would cost more than the offset it
    holds, which would defeat the point of indexing.
    """
    location = RecordLocation(path=Path("x"), offset=0)
    assert not hasattr(location, "__dict__")
