from __future__ import annotations

import json
import random
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from maa_duel.contracts import ModelVersion, sha256_file, write_contract
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

    def to(self, device: torch.device, *, non_blocking: bool = False) -> TensorBatch:
        return TensorBatch(**{name: value.to(device, non_blocking=non_blocking) for name, value in vars(self).items()})


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
    workers: int = 4,
    amp: bool = True,
    amp_dtype: str = "float16",
    compile_model: bool = False,
    pin_memory: bool = True,
    tf32: bool = True,
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
    if amp_dtype not in {"float16", "bfloat16"}:
        raise ValueError("amp_dtype must be float16 or bfloat16")
    cuda_enabled = selected_device.type == "cuda"
    amp_enabled = amp and cuda_enabled
    autocast_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
    torch.set_float32_matmul_precision("high")
    if cuda_enabled:
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
    maximum_enemy_id = max(unit.enemy_id for sample in samples for unit in (*sample.left_units, *sample.right_units))
    config = ModelConfig(
        num_enemy_ids=maximum_enemy_id + 1,
        embedding_dim=embedding_dim,
        heads=heads,
        layers=layers,
        dropout=dropout,
    )
    model = DuelTransformer(config).to(selected_device)
    training_model = torch.compile(model) if compile_model else model
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        PredictorDataset(samples),
        batch_size=min(batch_size, len(samples)),
        shuffle=True,
        collate_fn=collate_samples,
        generator=loader_generator,
        num_workers=workers,
        pin_memory=pin_memory and cuda_enabled,
        persistent_workers=workers > 0,
    )
    left_wins = sum(sample.winner is Winner.LEFT for sample in samples)
    right_wins = len(samples) - left_wins
    positive_weight = right_wins / left_wins if left_wins and right_wins else 1.0
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=selected_device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = torch.amp.GradScaler(
        selected_device.type,
        enabled=amp_enabled and autocast_dtype is torch.float16,
    )
    model_dir = workspace / "models" / "predictor"
    history: list[dict[str, float | int]] = []
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for batch in loader:
            batch = batch.to(selected_device, non_blocking=pin_memory and cuda_enabled)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=selected_device.type,
                dtype=autocast_dtype,
                enabled=amp_enabled,
            ):
                logits = training_model(
                    batch.left_ids,
                    batch.left_positions,
                    batch.left_mask,
                    batch.right_ids,
                    batch.right_positions,
                    batch.right_mask,
                )
                loss = criterion(logits, batch.labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
        "precision": amp_dtype if amp_enabled else "float32",
        "gpu": {
            "amp": amp_enabled,
            "workers": workers,
            "pin_memory": pin_memory and cuda_enabled,
            "compile": compile_model,
            "tf32": tf32 and cuda_enabled,
        },
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
    checkpoint = model_dir / "best-train-loss.pt"
    model_version = f"predictor-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    version_dir = model_dir / "versions" / model_version
    version_dir.mkdir(parents=True, exist_ok=False)
    version_checkpoint = version_dir / "best-train-loss.pt"
    shutil.copy2(checkpoint, version_checkpoint)
    shutil.copy2(model_dir / "training-report.json", version_dir / "training-report.json")
    contract = ModelVersion(
        model_version=model_version,
        task="predictor",
        dataset_version=metadata.get("dataset_version"),
        base_model="DuelTransformer",
        checkpoint=version_checkpoint.relative_to(workspace).as_posix(),
        checkpoint_sha256=sha256_file(version_checkpoint),
        device=str(selected_device),
        precision=("fp16" if amp_dtype == "float16" else "bf16") if amp_enabled else "fp32",
        training_args={
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "workers": workers,
            "amp": amp_enabled,
            "amp_dtype": amp_dtype,
            "compile": compile_model,
            "pin_memory": pin_memory,
            "tf32": tf32,
            "seed": seed,
        },
        git_commit=_git_commit(),
    )
    write_contract(version_dir / "model.json", contract)
    write_contract(model_dir / "model.json", contract)
    return report
