"""Progress reporting for long running stages.

Training a corpus of this size runs for hours. A stage that prints nothing until
it finishes is indistinguishable from a stage that has hung, so every loop that
can take minutes reports where it is, how fast it is going and when it expects
to finish.

Output goes to standard error through tqdm when one is attached to a terminal,
and degrades to periodic lines when it is not, so a run piped to a log file
stays readable instead of filling with carriage returns.
"""

from __future__ import annotations

import sys
import time
from typing import Iterable, Iterator


class ProgressTracker:
    """Reports position and rate through a long loop.

    Args:
        description: Short label shown beside the bar.
        total: Expected number of steps, or ``None`` when unknown.
        stream: Where to write. Defaults to standard error.
        interval: Seconds between lines when not attached to a terminal.
    """

    def __init__(
        self,
        description: str,
        total: int | None = None,
        stream=None,
        interval: float = 30.0,
    ) -> None:
        self._description = description
        self._total = total
        self._stream = stream or sys.stderr
        self._interval = interval
        self._bar = None
        self._count = 0
        self._started = 0.0
        self._last_line = 0.0
        self._postfix: dict[str, str] = {}

    def __enter__(self) -> "ProgressTracker":
        """Start the bar and the clock."""
        self._started = time.perf_counter()
        self._last_line = self._started
        if self._is_interactive():
            try:
                from tqdm.auto import tqdm

                self._bar = tqdm(
                    total=self._total,
                    desc=self._description,
                    unit="doc",
                    file=self._stream,
                    leave=False,
                    dynamic_ncols=True,
                )
            except ImportError:
                self._bar = None
        return self

    def __exit__(self, *exception) -> None:
        """Close the bar and print a closing summary."""
        if self._bar is not None:
            self._bar.close()
        elapsed = time.perf_counter() - self._started
        rate = self._count / elapsed if elapsed > 0 else 0.0
        print(
            f"{self._description}: {self._count} in {_duration(elapsed)} "
            f"({rate:.1f}/s)",
            file=self._stream,
            flush=True,
        )

    def advance(self, step: int = 1, **postfix: object) -> None:
        """Record progress and refresh the report.

        Args:
            step: How many units completed since the last call.
            **postfix: Named values shown alongside the position, such as the
                running loss.
        """
        self._count += step
        if postfix:
            self._postfix = {
                key: (f"{value:.4f}" if isinstance(value, float) else str(value))
                for key, value in postfix.items()
            }
        if self._bar is not None:
            self._bar.update(step)
            if self._postfix:
                self._bar.set_postfix(self._postfix, refresh=False)
            return
        self._maybe_print_line()

    def _maybe_print_line(self) -> None:
        """Print a periodic line when no terminal bar is available."""
        now = time.perf_counter()
        if now - self._last_line < self._interval:
            return
        self._last_line = now
        elapsed = now - self._started
        rate = self._count / elapsed if elapsed > 0 else 0.0
        parts = [f"{self._description}: {self._count}"]
        if self._total:
            share = self._count / self._total
            remaining = (self._total - self._count) / rate if rate > 0 else 0.0
            parts.append(f"/{self._total} ({share:.0%})")
            parts.append(f"eta {_duration(remaining)}")
        parts.append(f"{rate:.1f}/s")
        parts.extend(f"{key}={value}" for key, value in self._postfix.items())
        print(" ".join(parts), file=self._stream, flush=True)

    def _is_interactive(self) -> bool:
        """Whether the output stream is a terminal."""
        return bool(getattr(self._stream, "isatty", lambda: False)())


def track(
    iterable: Iterable,
    description: str,
    total: int | None = None,
    stream=None,
) -> Iterator:
    """Iterate while reporting progress.

    Args:
        iterable: What to iterate.
        description: Short label shown beside the bar.
        total: Expected number of steps, or ``None`` when unknown.
        stream: Where to write. Defaults to standard error.

    Yields:
        The items of ``iterable``.
    """
    if total is None:
        total = len(iterable) if hasattr(iterable, "__len__") else None
    with ProgressTracker(description, total=total, stream=stream) as tracker:
        for item in iterable:
            yield item
            tracker.advance()


def _duration(seconds: float) -> str:
    """Format a duration in the largest sensible unit.

    Args:
        seconds: Elapsed or remaining seconds.

    Returns:
        A short human readable duration.
    """
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"
