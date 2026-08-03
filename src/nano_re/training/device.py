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
            time per document from 84 ms to 75 ms.
        pin_memory: Whether the loader should pin host memory, which speeds the
            copy to a discrete GPU and does nothing on unified memory.
        num_workers: Loader worker processes. Records are parsed in Python on
            demand, so a discrete GPU benefits from parsing ahead of the device.
        autocast_dtype: Reduced precision to compute in, or ``None`` for float32.
    """

    batch_size: int
    pin_memory: bool
    num_workers: int
    autocast_dtype: torch.dtype | None


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
        return (
            f"{name}, {precision}, batch {self._tuning.batch_size}, "
            f"{self._tuning.num_workers} loader workers"
        )

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
                batch_size=16,
                pin_memory=True,
                num_workers=4,
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
