from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def write_jsonl(path: Path, rows: Iterable[BaseModel]) -> None:
    """Atomically replace a JSONL manifest with UTF-8 records."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(row.model_dump_json(exclude_none=True))
            stream.write("\n")
    os.replace(temporary, path)


def read_jsonl[ModelT: BaseModel](path: Path, model: type[ModelT]) -> list[ModelT]:
    if not path.exists():
        return []
    rows: list[ModelT] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                rows.append(model.model_validate_json(line))
            except (ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid JSONL record at {path}:{line_number}") from exc
    return rows
