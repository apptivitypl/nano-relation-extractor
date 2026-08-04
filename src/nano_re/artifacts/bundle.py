"""Assembly and verification of the finished local model bundle.

The bundle is the deliverable: a single directory that contains everything
needed to run the model without this package, plus the reports that document how
it was produced. This module inventories that directory, checks nothing expected
is missing, and writes the manifest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import PackagingConfig

EXPECTED_ARTIFACTS: tuple[str, ...] = (
    "config.json",
    "model.safetensors",
    "model_int8.onnx",
    "tokenizer.json",
    "label_schema.json",
    "MODEL_CARD.md",
)

MANIFEST_FILENAME = "MANIFEST.json"


@dataclass(frozen=True)
class ArtifactEntry:
    """One file in the bundle.

    Attributes:
        name: Path of the file relative to the bundle directory.
        size_bytes: Size of the file on disk.
    """

    name: str
    size_bytes: int

    @property
    def size_mb(self) -> float:
        """Size of the file in megabytes."""
        return self.size_bytes / 1e6

    def to_dict(self) -> dict[str, object]:
        """Return a JSON compatible representation of the entry."""
        return {"name": self.name, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class BundleReport:
    """Inventory of an assembled bundle.

    Attributes:
        directory: Location of the bundle.
        entries: Every file in the bundle.
        missing: Expected artifacts that were absent.
        removed: Files deleted during assembly, such as the float32 graph when
            it was not requested.
    """

    directory: Path
    entries: tuple[ArtifactEntry, ...]
    missing: tuple[str, ...]
    removed: tuple[str, ...] = ()

    @property
    def total_bytes(self) -> int:
        """Combined size of every file in the bundle."""
        return sum(entry.size_bytes for entry in self.entries)

    @property
    def is_complete(self) -> bool:
        """Whether every expected artifact is present."""
        return not self.missing

    def to_dict(self) -> dict[str, object]:
        """Return a JSON compatible representation of the report."""
        return {
            "directory": str(self.directory),
            "entries": [entry.to_dict() for entry in self.entries],
            "missing": list(self.missing),
            "removed": list(self.removed),
            "total_bytes": self.total_bytes,
            "is_complete": self.is_complete,
        }

    def render(self) -> str:
        """Return a human readable inventory of the bundle.

        Returns:
            A plain text table of file names and sizes.
        """
        lines = [f"Bundle: {self.directory}", "Files:"]
        for entry in self.entries:
            lines.append(f"  {entry.name:<28} {entry.size_mb:>10.2f} MB")
        lines.append(f"  {'TOTAL':<28} {self.total_bytes / 1e6:>10.2f} MB")
        for name in self.removed:
            lines.append(f"Removed: {name}")
        if self.missing:
            lines.append(f"Missing expected artifacts: {', '.join(self.missing)}")
        return "\n".join(lines)


class BundleAssembler:
    """Finalises the artifact directory into a self-contained bundle.

    Args:
        config: Packaging settings controlling what the bundle retains.
    """

    def __init__(self, config: PackagingConfig) -> None:
        self._config = config

    def assemble(self, directory: Path, fp32_graph: Path | None = None) -> BundleReport:
        """Prune, inventory and verify the bundle, then write the manifest.

        Args:
            directory: Directory holding the artifacts.
            fp32_graph: Location of the float32 graph, removed when the
                configuration does not ask to keep it.

        Returns:
            The bundle report.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        if not directory.is_dir():
            raise FileNotFoundError(
                f"{directory} does not exist. Run the earlier stages first."
            )

        removed: list[str] = []
        if (
            not self._config.keep_fp32_graph
            and fp32_graph is not None
            and fp32_graph.exists()
        ):
            name = fp32_graph.name
            fp32_graph.unlink()
            removed.append(name)

        manifest_path = directory / MANIFEST_FILENAME
        manifest_path.unlink(missing_ok=True)
        entries = self._inventory(directory)
        present = {entry.name for entry in entries}
        missing = tuple(name for name in EXPECTED_ARTIFACTS if name not in present)

        report = BundleReport(
            directory=directory,
            entries=entries,
            missing=missing,
            removed=tuple(removed),
        )
        manifest_path.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        return report

    def _inventory(self, directory: Path) -> tuple[ArtifactEntry, ...]:
        """List every file in a directory with its size.

        Args:
            directory: Directory to inventory.

        Returns:
            The entries, sorted by relative path.
        """
        return tuple(
            ArtifactEntry(
                name=str(path.relative_to(directory)), size_bytes=path.stat().st_size
            )
            for path in sorted(directory.rglob("*"))
            if path.is_file()
        )
