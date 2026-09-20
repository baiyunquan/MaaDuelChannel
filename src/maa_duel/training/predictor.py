from __future__ import annotations

import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from maa_duel.dataset import PredictorSample
from maa_duel.schema import Winner
from maa_duel.store import read_jsonl
from maa_duel.training.model import DuelTransformer, ModelConfig


@dataclass
class TensorBatch:
    left_ids: torch.Tensor
    left_positions: torch.Tensor
    left_mask: torch.Tensor
    right_ids: torch.Tensor
    right_positions: torch.Tensor
    right_mask: torch.Tensor
    labels: torch.Tensor

    def to(self, device: torch.device) -> TensorBatch:
        return TensorBatch(**{name: value.to(device) for name, value in vars(self).items()})


class PredictorDataset(Dataset):
    def __init__(self, samples: list[PredictorSample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> PredictorSample:
        return self.samples[index]


def collate_samples(samples: list[PredictorSample]) -> TensorBatch:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(samples)
    max_left = max(1, max(len(sample.left_units) for sample in samples))
    max_right = max(1, max(len(sample.right_units) for sample in samples))
    left_ids = torch.zeros((batch_size, max_left), dtype=torch.long)
    right_ids = torch.zeros((batch_size, max_right), dtype=torch.long)
    left_positions = torch.zeros((batch_size, max_left, 2), dtype=torch.float32)
    right_positions = torch.zeros((batch_size, max_right, 2), dtype=torch.float32)
    left_mask = torch.zeros((batch_size, max_left), dtype=torch.bool)
    right_mask = torch.zeros((batch_size, max_right), dtype=torch.bool)
    labels = torch.zeros(batch_size, dtype=torch.float32)

    for batch_index, sample in enumerate(samples):
        for unit_index, unit in enumerate(sample.left_units):
            left_ids[batch_index, unit_index] = unit.enemy_id
            left_positions[batch_index, unit_index] = torch.tensor((unit.x, unit.y))
            left_mask[batch_index, unit_index] = True
        for unit_index, unit in enumerate(sample.right_units):
            right_ids[batch_index, unit_index] = unit.enemy_id
            right_positions[batch_index, unit_index] = torch.tensor((1.0 - unit.x, unit.y))
            right_mask[batch_index, unit_index] = True
        labels[batch_index] = 1.0 if sample.winner is Winner.LEFT else 0.0

    return TensorBatch(
        left_ids=left_ids,
        left_positions=left_positions,
        left_mask=left_mask,
        right_ids=right_ids,
        right_positions=right_positions,
        right_mask=right_mask,
        labels=labels,
    )


def _git_commit() -> str:
    repository = Path(__file__).resolve().parents[3]
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _checkpoint(
    path: Path,
    model: DuelTransformer,
    config: ModelConfig,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    dataset_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(config),
            "epoch": epoch,
            "dataset_sha256": dataset_sha256,
            "git_commit": _git_commit(),
        },
        path,
    )


def train_predictor_model(
    workspace: Path,
    *,
    all_samples: bool,
    epochs: int = 100,
    batch_size: int = 64,
    embedding_dim: int = 128,
    heads: int = 4,
    layers: int = 3,
    dropout: float = 0.1,
    learning_rate: float = 3e-4,
    device: str | None = None,
    seed: int = 20260920,
) -> dict[str, object]:
    if not all_samples:
        raise ValueError("predictor training requires --all; no validation or test split is created")
    manifest_dir = workspace / "manifests"
    samples = read_jsonl(manifest_dir / "predictor.jsonl", PredictorSample)
    if not samples:
        raise ValueError("predictor dataset is empty; run build-dataset after extraction and review")
    metadata = json.loads((manifest_dir / "predictor.meta.json").read_text(encoding="utf-8"))
    if metadata.get("accepted_samples") != len(samples) or metadata.get("written_samples") != len(samples):
        raise ValueError(
            f"accepted/written/training count mismatch: "
            f"{metadata.get('accepted_samples')}/{metadata.get('written_samples')}/{len(samples)}"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    maximum_enemy_id = max(unit.enemy_id for sample in samples for unit in (*sample.left_units, *sample.right_units))
    config = ModelConfig(
        num_enemy_ids=maximum_enemy_id + 1,
        embedding_dim=embedding_dim,
        heads=heads,
        layers=layers,
        dropout=dropout,
    )
    model = DuelTransformer(config).to(selected_device)
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        PredictorDataset(samples),
        batch_size=min(batch_size, len(samples)),
        shuffle=True,
        collate_fn=collate_samples,
        generator=loader_generator,
    )
    left_wins = sum(sample.winner is Winner.LEFT for sample in samples)
    right_wins = len(samples) - left_wins
    positive_weight = right_wins / left_wins if left_wins and right_wins else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=selected_device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model_dir = workspace / "models" / "predictor"
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for batch in loader:
            batch = batch.to(selected_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                batch.left_ids,
                batch.left_positions,
                batch.left_mask,
                batch.right_ids,
                batch.right_positions,
                batch.right_mask,
            )
            loss = criterion(logits, batch.labels)
            loss.backward()
            optimizer.step()
            batch_size_actual = batch.labels.shape[0]
            total_loss += float(loss.detach()) * batch_size_actual
            correct += int(((logits.detach() >= 0) == (batch.labels >= 0.5)).sum())
            seen += batch_size_actual
        epoch_loss = total_loss / seen
        epoch_accuracy = correct / seen
        history.append({"epoch": epoch, "train_loss": epoch_loss, "train_accuracy": epoch_accuracy})
        _checkpoint(model_dir / "last.pt", model, config, optimizer, epoch, metadata["dataset_sha256"])
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            _checkpoint(
                model_dir / "best-train-loss.pt",
                model,
                config,
                optimizer,
                epoch,
                metadata["dataset_sha256"],
            )

    report: dict[str, object] = {
        "training_samples": len(samples),
        "validation_samples": 0,
        "test_samples": 0,
        "dataset_sha256": metadata["dataset_sha256"],
        "device": str(selected_device),
        "model_config": asdict(config),
        "class_balance": {"left": left_wins, "right": right_wins},
        "metrics_scope": "training-only",
        "history": history,
        "git_commit": _git_commit(),
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "training-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report
