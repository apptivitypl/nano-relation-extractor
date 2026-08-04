"""Hardware selection and the tuning that depends on it.

Every hardware dependent decision is concentrated here. The trainer asks for a
device, an autocast context and a batch size without knowing which backend it
received.

The defaults are measured rather than assumed, and the measurements disagree
with the usual advice. Mixed precision helps on CUDA and hurts on Apple Silicon:
on an M4 Pro a training step took 341 ms in float32 and 369 ms under float16
autocast, because the cast operations cost more than the cheaper arithmetic
saves on a unified memory architecture. Autocast is therefore enabled on CUDA
only.
"""

from __future__ import annotations

import contextlib
import os
import random
from dataclasses import dataclass
from typing import ContextManager

import numpy as np
import torch


@dataclass(frozen=True)
class DeviceTuning:
    """Settings a backend performs best with.

    Attributes:
        batch_size: Documents per batch. Larger batches amortise kernel launch
            overhead; on an M4 Pro, moving from four to eight documents cut the
            time per document from 84 ms to 75 ms. On CUDA the figure is chosen
            from the card's memory instead, since that binds first.
        pin_memory: Whether the loader should pin host memory, which speeds the
            copy to a discrete GPU and does nothing on unified memory.
        num_workers: Loader worker processes. Records are parsed in Python on
            demand, so a discrete GPU benefits from parsing ahead of the device.
        autocast_dtype: Reduced precision to compute in, or ``None`` for float32.
        effective_batch_size: Documents per optimiser step. When memory forces a
            smaller batch, gradient accumulation makes up the difference, so the
            optimisation sees the same step size on any card.
    """

    batch_size: int
    pin_memory: bool
    num_workers: int
    autocast_dtype: torch.dtype | None
    effective_batch_size: int = 32


class DeviceManager:
    """Resolves the compute device and the policy that suits it.

    Args:
        preference: Optional device string forcing a specific backend. When
            ``None`` the best available backend is selected automatically.
        use_amp: Whether to allow mixed precision where it is known to help.
    """

    def __init__(self, preference: str | None = None, use_amp: bool = True) -> None:
        self._device = torch.device(preference or self._detect_backend())
        self._tuning = self._resolve_tuning(use_amp)
        if self._device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    @property
    def device(self) -> torch.device:
        """Device that model and batches are placed on."""
        return self._device

    @property
    def tuning(self) -> DeviceTuning:
        """Settings this backend performs best with."""
        return self._tuning

    @property
    def amp_enabled(self) -> bool:
        """Whether autocast is active."""
        return self._tuning.autocast_dtype is not None

    def autocast(self) -> ContextManager[None]:
        """Return the autocast context for the resolved backend.

        Returns:
            A mixed precision context where that helps, otherwise a null context.
        """
        if self._tuning.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.amp.autocast(
            device_type=self._device.type, dtype=self._tuning.autocast_dtype
        )

    def grad_scaler(self) -> torch.amp.GradScaler:
        """Return a gradient scaler matching the precision policy.

        Scaling exists to keep float16 gradients from underflowing. bfloat16 has
        the exponent range of float32 and needs none, so the scaler is enabled
        only for float16.

        Returns:
            A scaler, disabled unless float16 autocast is in use.
        """
        return torch.amp.GradScaler(
            self._device.type,
            enabled=self._tuning.autocast_dtype == torch.float16,
        )

    def resolve_batch_size(self, configured: int) -> int:
        """Choose a batch size, honouring an explicit setting.

        Args:
            configured: Batch size from configuration. Zero or less requests the
                measured default for this backend.

        Returns:
            The batch size to use.
        """
        return configured if configured > 0 else self._tuning.batch_size

    def resolve_accumulation(self, configured: int, batch_size: int) -> int:
        """Choose how many batches to accumulate before stepping.

        A card that can only hold eight documents at a time should still take
        the same optimisation step as one that holds thirty-two, otherwise the
        learning rate means something different on every machine. Accumulation
        makes up whatever the batch size could not.

        Args:
            configured: Accumulation from configuration. Zero or less asks for
                the value that reaches the target effective batch.
            batch_size: Documents actually held per forward pass.

        Returns:
            Batches to accumulate per optimiser step, at least one.
        """
        if configured > 0:
            return configured
        target = self._tuning.effective_batch_size
        return max(1, round(target / max(1, batch_size)))

    def fit_batch_size(self, probe, batch_size: int, minimum: int = 1) -> int:
        """Shrink the batch until one real step fits in memory.

        An out of memory failure an hour into a run is the worst way to discover
        a batch was too large, and no static table can predict it: it depends on
        sequence length, entity count and whether context pooling is on. Trying
        one step of each candidate size costs seconds and removes the guess.

        Args:
            probe: Callable taking a batch size and performing one full training
                step, raising on exhaustion.
            batch_size: First size to try.
            minimum: Smallest size worth attempting.

        Returns:
            The largest size that completed a step.

        Raises:
            RuntimeError: If even the minimum size cannot run.
        """
        candidate = max(minimum, batch_size)
        while candidate >= minimum:
            try:
                probe(candidate)
                return candidate
            except (torch.OutOfMemoryError, RuntimeError) as error:
                if not _is_memory_error(error):
                    raise
                self.empty_cache()
                if candidate == minimum:
                    break
                candidate = max(minimum, candidate // 2)
        raise RuntimeError(
            f"Could not run a training step even at batch size {minimum}. "
            "Reduce NANO_RE_MAX_SEQUENCE_LENGTH or set "
            "NANO_RE_LOCALIZED_CONTEXT=false."
        )

    def empty_cache(self) -> None:
        """Release cached device memory, where the backend has a cache."""
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device.type == "mps":
            torch.mps.empty_cache()

    def seed_everything(self, seed: int) -> None:
        """Seed Python, NumPy and PyTorch generators.

        Args:
            seed: Seed applied to every generator.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def describe(self) -> str:
        """Return a human readable description of the active backend."""
        precision = (
            str(self._tuning.autocast_dtype).replace("torch.", "")
            if self._tuning.autocast_dtype is not None
            else "float32"
        )
        name = self._device.type
        if name == "cuda" and torch.cuda.is_available():
            name = f"cuda ({torch.cuda.get_device_name(0)})"
        return f"{name}, {precision}, {self._tuning.num_workers} loader workers"

    def _resolve_tuning(self, use_amp: bool) -> DeviceTuning:
        """Pick the settings that suit the resolved backend.

        Args:
            use_amp: Whether mixed precision is permitted at all.

        Returns:
            The tuning for this backend.
        """
        if self._device.type == "cuda":
            dtype = None
            if use_amp:
                dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )
            return DeviceTuning(
                batch_size=self._cuda_batch_size(),
                pin_memory=True,
                num_workers=min(8, max(2, (os.cpu_count() or 4) // 2)),
                autocast_dtype=dtype,
            )
        if self._device.type == "mps":
            return DeviceTuning(
                batch_size=8,
                pin_memory=False,
                num_workers=0,
                autocast_dtype=None,
            )
        return DeviceTuning(
            batch_size=4, pin_memory=False, num_workers=0, autocast_dtype=None
        )

    @staticmethod
    def _cuda_batch_size() -> int:
        """Choose a batch size the card can hold.

        Memory, not throughput, is what bounds the batch here. Requesting the
        encoder's attention for context pooling materialises one map per layer,
        which for this architecture is 22 maps of ``batch x heads x S x S``: at
        512 tokens and a batch of 16 that alone is over two gigabytes, before
        activations, gradients and optimiser state. Sizing from the card's
        reported memory avoids an out of memory failure an hour into a run.

        Returns:
            A batch size appropriate to the visible device.
        """
        try:
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            return 8
        if total >= 40:
            return 32
        if total >= 22:
            return 24
        if total >= 14:
            return 16
        if total >= 10:
            return 12
        return 8

    @staticmethod
    def _detect_backend() -> str:
        """Return the best available backend name.

        Returns:
            ``cuda``, ``mps`` or ``cpu``.
        """
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def _is_memory_error(error: Exception) -> bool:
    """Test whether an exception reports exhausted device memory.

    Backends spell this differently, and CUDA raises a dedicated type while
    Apple Silicon raises a generic runtime error, so the message is inspected as
    well as the type.

    Args:
        error: The raised exception.

    Returns:
        ``True`` when the failure was running out of memory.
    """
    if isinstance(error, torch.OutOfMemoryError):
        return True
    text = str(error).lower()
    return "out of memory" in text or "insufficient memory" in text
