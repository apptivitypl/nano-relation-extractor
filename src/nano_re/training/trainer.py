"""The multi-task optimisation loop.

The trainer receives every collaborator it needs and constructs none of them. It
is responsible for exactly one thing: advancing the model through epochs while
respecting the mixed precision, accumulation and clipping policy.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from ..config import TrainingConfig
from ..models import NanoREModel
from .device import DeviceManager
from .evaluator import MultiTaskEvaluator
from .losses import MultiTaskLoss
from .metrics import EvaluationResult
from .progress import ProgressTracker


@dataclass(frozen=True)
class EpochReport:
    """Summary of one training epoch.

    Attributes:
        epoch: One-based epoch number.
        train_loss: Mean weighted loss over the epoch.
        train_ner_loss: Mean unweighted token classification loss.
        train_relation_loss: Mean unweighted relation loss.
        evaluation: Scores measured on the evaluation split after the epoch.
        seconds: Wall clock duration of the epoch including evaluation.
    """

    epoch: int
    train_loss: float
    train_ner_loss: float
    train_relation_loss: float
    evaluation: EvaluationResult
    seconds: float

    def to_dict(self) -> dict[str, object]:
        """Return a JSON compatible representation of the epoch."""
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "train_ner_loss": self.train_ner_loss,
            "train_relation_loss": self.train_relation_loss,
            "evaluation": self.evaluation.to_dict(),
            "seconds": self.seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "EpochReport":
        """Rebuild an epoch summary from a dictionary.

        Args:
            payload: Dictionary previously produced by :meth:`to_dict`.

        Returns:
            The reconstructed summary.
        """
        return cls(
            epoch=int(payload["epoch"]),
            train_loss=float(payload["train_loss"]),
            train_ner_loss=float(payload["train_ner_loss"]),
            train_relation_loss=float(payload["train_relation_loss"]),
            evaluation=EvaluationResult.from_dict(payload["evaluation"]),
            seconds=float(payload["seconds"]),
        )


@dataclass
class TrainingReport:
    """Everything the model card needs to describe a training run.

    Attributes:
        backbone_name: Encoder identifier.
        device: Human readable description of the compute backend.
        relation_objective: Name of the relation loss strategy used.
        epochs: Per-epoch summaries in chronological order.
        best_epoch: One-based index of the checkpoint that was kept.
        best_evaluation: Scores of the retained checkpoint.
        total_seconds: Wall clock duration of the whole run.
        hyperparameters: Flattened training configuration.
    """

    backbone_name: str
    device: str
    relation_objective: str
    epochs: list[EpochReport] = field(default_factory=list)
    best_epoch: int = 0
    best_evaluation: EvaluationResult | None = None
    total_seconds: float = 0.0
    hyperparameters: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON compatible representation of the run."""
        return {
            "backbone_name": self.backbone_name,
            "device": self.device,
            "relation_objective": self.relation_objective,
            "epochs": [epoch.to_dict() for epoch in self.epochs],
            "best_epoch": self.best_epoch,
            "best_evaluation": (
                self.best_evaluation.to_dict() if self.best_evaluation else None
            ),
            "total_seconds": self.total_seconds,
            "hyperparameters": self.hyperparameters,
        }

    def save(self, path: Path) -> Path:
        """Write the report to disk as JSON.

        Args:
            path: Destination file path.

        Returns:
            The path that was written.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "TrainingReport":
        """Rebuild a training report from a dictionary.

        Args:
            payload: Dictionary previously produced by :meth:`to_dict`.

        Returns:
            The reconstructed report.
        """
        best = payload.get("best_evaluation")
        return cls(
            backbone_name=str(payload["backbone_name"]),
            device=str(payload["device"]),
            relation_objective=str(payload["relation_objective"]),
            epochs=[EpochReport.from_dict(item) for item in payload["epochs"]],
            best_epoch=int(payload["best_epoch"]),
            best_evaluation=None if best is None else EvaluationResult.from_dict(best),
            total_seconds=float(payload["total_seconds"]),
            hyperparameters=dict(payload["hyperparameters"]),
        )

    @classmethod
    def load(cls, path: Path) -> "TrainingReport":
        """Read a previously written report.

        Args:
            path: Source file path.

        Returns:
            The reconstructed report.
        """
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


class MultiTaskTrainer:
    """Trains a :class:`NanoREModel` on both tasks jointly.

    Args:
        model: Model to optimise.
        criterion: Weighted multi-task loss.
        evaluator: Scorer invoked after each epoch.
        device_manager: Resolved device and mixed precision policy.
        config: Optimisation settings.
        on_epoch_end: Optional callback receiving each :class:`EpochReport`.
    """

    def __init__(
        self,
        model: NanoREModel,
        criterion: MultiTaskLoss,
        evaluator: MultiTaskEvaluator,
        device_manager: DeviceManager,
        config: TrainingConfig,
        on_epoch_end=None,
    ) -> None:
        self._model = model
        self._criterion = criterion
        self._evaluator = evaluator
        self._device_manager = device_manager
        self._config = config
        self._on_epoch_end = on_epoch_end

    def train(
        self, train_loader: DataLoader, eval_loader: DataLoader
    ) -> TrainingReport:
        """Run the full optimisation schedule.

        The model is left holding the parameters of the best scoring epoch, so
        the caller can persist it directly without reloading a checkpoint.

        Args:
            train_loader: Loader over the training split.
            eval_loader: Loader over the evaluation split.

        Returns:
            A report describing every epoch and the retained checkpoint.
        """
        device = self._device_manager.device
        self._device_manager.seed_everything(self._config.seed)
        self._model.to(device)

        optimizer = torch.optim.AdamW(
            self._model.parameter_groups(
                weight_decay=self._config.weight_decay,
                encoder_lr=self._config.learning_rate,
                head_lr=self._config.head_learning_rate,
            )
        )
        steps_per_epoch = max(
            1, len(train_loader) // self._config.gradient_accumulation_steps
        )
        total_steps = steps_per_epoch * self._config.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * self._config.warmup_ratio),
            num_training_steps=total_steps,
        )
        scaler = self._device_manager.grad_scaler()

        report = TrainingReport(
            backbone_name=self._model.architecture.backbone_name,
            device=self._device_manager.describe(),
            relation_objective=self._criterion.relation_objective.name,
            hyperparameters=_serialisable_config(self._config),
        )
        best_score = float("-inf")
        best_state: dict[str, torch.Tensor] | None = None
        run_start = time.perf_counter()

        for epoch in range(1, self._config.epochs + 1):
            epoch_start = time.perf_counter()
            losses = self._run_epoch(
                train_loader, optimizer, scheduler, scaler, epoch
            )
            evaluation = self._evaluator.evaluate(self._model, eval_loader)
            epoch_report = EpochReport(
                epoch=epoch,
                train_loss=losses[0],
                train_ner_loss=losses[1],
                train_relation_loss=losses[2],
                evaluation=evaluation,
                seconds=time.perf_counter() - epoch_start,
            )
            report.epochs.append(epoch_report)

            if evaluation.combined_f1 > best_score:
                best_score = evaluation.combined_f1
                report.best_epoch = epoch
                report.best_evaluation = evaluation
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self._model.state_dict().items()
                }

            if self._on_epoch_end is not None:
                self._on_epoch_end(epoch_report)

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.to(torch.device("cpu"))
        report.total_seconds = time.perf_counter() - run_start
        return report

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: torch.amp.GradScaler,
        epoch: int,
    ) -> tuple[float, float, float]:
        """Run a single training epoch.

        Args:
            loader: Loader over the training split.
            optimizer: Optimiser to step.
            scheduler: Learning rate schedule advanced with the optimiser.
            scaler: Gradient scaler matching the mixed precision policy.
            epoch: One-based epoch number, shown in the progress report.

        Returns:
            Mean total, NER and relation losses over the epoch.
        """
        self._model.train()
        accumulation = max(1, self._config.gradient_accumulation_steps)
        totals = [0.0, 0.0, 0.0, 0.0]
        pending = 0
        optimizer.zero_grad(set_to_none=True)
        tracker = ProgressTracker(
            f"epoch {epoch}/{self._config.epochs}", total=len(loader)
        )

        with tracker:
            self._epoch_body(
                loader, optimizer, scheduler, scaler, accumulation, totals, tracker
            )
        num_batches = int(totals[3])
        divisor = max(1, num_batches)
        return (totals[0] / divisor, totals[1] / divisor, totals[2] / divisor)

    def _epoch_body(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        scaler: torch.amp.GradScaler,
        accumulation: int,
        totals: list[float],
        tracker: ProgressTracker,
    ) -> None:
        """Iterate one epoch, accumulating losses and reporting progress.

        Args:
            loader: Loader over the training split.
            optimizer: Optimiser to step.
            scheduler: Learning rate schedule advanced with the optimiser.
            scaler: Gradient scaler matching the mixed precision policy.
            accumulation: Micro-batches per optimiser step.
            totals: Running loss sums and batch count, updated in place.
            tracker: Progress reporter advanced once per batch.
        """
        device = self._device_manager.device
        pending = 0
        for step, batch in enumerate(loader, start=1):
            if batch is not None:
                batch = batch.to(device)
                with self._device_manager.autocast():
                    outputs = self._model(**batch.model_inputs())
                    losses = self._criterion(
                        ner_logits=outputs.ner_logits,
                        ner_labels=batch.ner_labels,
                        relation_logits=outputs.relation_logits,
                        relation_labels=batch.relation_labels,
                        pair_mask=batch.pair_mask,
                    )
                scaler.scale(losses.total / accumulation).backward()

                totals[0] += float(losses.total.item())
                totals[1] += float(losses.ner.item())
                totals[2] += float(losses.relation.item())
                totals[3] += 1.0
                pending += 1

            seen = max(1.0, totals[3])
            tracker.advance(
                loss=totals[0] / seen,
                ner=totals[1] / seen,
                re=totals[2] / seen,
            )

            if pending and (step % accumulation == 0 or step == len(loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), self._config.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                pending = 0


def _serialisable_config(config: TrainingConfig) -> dict[str, object]:
    """Convert a training configuration into JSON compatible values.

    Args:
        config: Configuration to convert.

    Returns:
        A dictionary with paths rendered as strings.
    """
    payload = asdict(config)
    payload["output_dir"] = str(payload["output_dir"])
    payload["init_from"] = (
        str(payload["init_from"]) if payload["init_from"] is not None else None
    )
    return payload
