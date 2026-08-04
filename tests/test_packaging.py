"""Tests that the repository ships what it claims to ship.

A source file excluded by a build or version control rule fails nowhere locally:
the working copy has it, the tests pass, and the omission only surfaces when
somebody clones the repository and the import fails. These checks run against
the rules themselves.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "src" / "nano_re"


def _source_files() -> list[Path]:
    """Return every Python file in the package.

    Returns:
        Paths relative to the repository root.
    """
    return sorted(path.relative_to(ROOT) for path in PACKAGE.rglob("*.py"))


def test_no_source_file_is_excluded_from_version_control() -> None:
    """Every module in the package is committable.

    This exists because an unanchored ``artifacts/`` rule once matched
    ``src/nano_re/artifacts/`` as well as the output directory it was meant for,
    so the package was absent from every clone while working locally.
    """
    sources = _source_files()
    assert sources, "no source files found"

    result = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        input="\n".join(str(path) for path in sources),
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    ignored = [line for line in result.stdout.splitlines() if line.strip()]
    assert not ignored, f"excluded from version control: {ignored}"


def test_every_package_directory_has_an_init() -> None:
    """A directory of modules without an init is not an importable package."""
    for directory in PACKAGE.rglob("*"):
        if not directory.is_dir() or directory.name == "__pycache__":
            continue
        if any(directory.glob("*.py")):
            assert (directory / "__init__.py").exists(), f"{directory} has no __init__"


def test_the_output_directory_is_still_excluded() -> None:
    """Fixing the rule must not start committing trained weights."""
    result = subprocess.run(
        ["git", "check-ignore", "artifacts/model.safetensors"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 0, "the artifacts output directory is not ignored"
