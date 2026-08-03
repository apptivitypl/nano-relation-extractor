"""Hardware selection and mixed precision policy.

Every hardware dependent decision is concentrated here. The trainer asks for an
autocast context and a gradient scaler without knowing whether it is running on
CUDA, Apple Silicon or CPU.
"""

from __future__ import annotations

import contextlib
import random
from typing import ContextManager

import numpy as np
import torch


class DeviceManager:
    """Resolves the compute device and the matching mixed precision policy.

    Automatic mixed precision is enabled only on CUDA. Half precision autocast is
    not a portable win on Apple Silicon or CPU, and enabling a gradient scaler
    without CUDA silently degrades to a no-op, so both are gated explicitly.

    Args:
        preference: Optional device string forcing a specific backend. When
            ``None`` the best available backend is selected automatically.
        use_amp: Whether to request mixed precision when the backend supports it.
    """

    def __init__(self, preference: str | None = None, use_amp: bool = True) -> None:
        self._device = torch.device(preference or self._detect_backend())
        self._use_amp = use_amp and self._device.type == "cuda"

    @property
    def device(self) -> torch.device:
        """Device that model and batches are placed on."""
        return self._device

    @property
    def amp_enabled(self) -> bool:
        """Whether autocast and gradient scaling are active."""
        return self._use_amp

    def autocast(self) -> ContextManager[None]:
        """Return the autocast context for the resolved backend.

        Returns:
            A mixed precision context on CUDA, otherwise a null context.
        """
        if not self._use_amp:
            return contextlib.nullcontext()
        return torch.amp.autocast(device_type=self._device.type, dtype=torch.float16)

    def grad_scaler(self) -> torch.amp.GradScaler:
        """Return a gradient scaler matching the mixed precision policy.

        Returns:
            A scaler that is disabled whenever autocast is inactive.
        """
        return torch.amp.GradScaler(self._device.type, enabled=self._use_amp)

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
        precision = "mixed float16" if self._use_amp else "float32"
        return f"{self._device.type} ({precision})"

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
