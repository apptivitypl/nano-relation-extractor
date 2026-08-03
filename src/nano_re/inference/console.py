"""Interactive and batch console for extraction.

Keeping the console separate from the extractor means the extraction logic never
learns about terminals, prompts or JSON formatting, and the console never learns
about tensors.
"""

from __future__ import annotations

import json
import sys

from .extractor import RelationExtractor
from .results import ExtractionResult

BANNER = (
    "Paste or type text, then press Enter on an empty line to extract.\n"
    "Press Ctrl-D to quit."
)


class ExtractionConsole:
    """Drives extraction from a terminal, a pipe, a file or a literal string.

    Args:
        extractor: Configured extractor.
        as_json: Whether to emit JSON instead of a human readable report.
        stream: Output stream. Defaults to standard output.
    """

    def __init__(
        self, extractor: RelationExtractor, as_json: bool = False, stream=None
    ) -> None:
        self._extractor = extractor
        self._as_json = as_json
        self._stream = stream or sys.stdout

    def run_once(self, text: str) -> ExtractionResult:
        """Extract from a single text and write the report.

        Args:
            text: Raw input text.

        Returns:
            The extraction result.
        """
        result = self._extractor.extract(text)
        self._emit(result)
        return result

    def run_interactive(self) -> None:
        """Read blocks of text from the terminal until the user quits.

        A blank line submits the block, which lets multi-line text be pasted
        whole rather than being extracted line by line.
        """
        print(f"Backend: {self._extractor.backend_name}", file=self._stream)
        print(BANNER, file=self._stream)
        while True:
            block = self._read_block()
            if block is None:
                print("", file=self._stream)
                return
            if not block.strip():
                continue
            self._emit(self._extractor.extract(block))

    def _read_block(self) -> str | None:
        """Read lines until a blank line or end of input.

        Returns:
            The collected text, or ``None`` when the user ended the session.
        """
        lines: list[str] = []
        while True:
            try:
                line = input("\n> " if not lines else "  ")
            except EOFError:
                return "\n".join(lines) if lines else None
            except KeyboardInterrupt:
                return None
            if not line.strip():
                if lines:
                    return "\n".join(lines)
                continue
            lines.append(line)

    def _emit(self, result: ExtractionResult) -> None:
        """Write one result in the configured format.

        Args:
            result: Extraction to report.
        """
        if self._as_json:
            print(json.dumps(result.to_dict(), indent=2), file=self._stream)
        else:
            print("", file=self._stream)
            print(result.render(), file=self._stream)
