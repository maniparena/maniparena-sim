"""Shared helpers for dataset handlers."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from opencvpr.utils.math_utils import to_numpy_keep_shape as to_numpy


def safe_name(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text or "")
    return cleaned.strip("_") or "dataset"


def resolve_root_path(path_str: str) -> Path:
    path = Path(os.path.expanduser(path_str))
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    return value


def write_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_compatible(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        for row in rows:
            file_obj.write(json.dumps(json_compatible(row), ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def nested_get(data: dict[str, Any], *keys: str) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def require_pyarrow():
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("Direct LeRobot export requires `pyarrow`.") from exc
    return pa, pq


def feature_stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr[:, None]
    arr_float = arr.astype(np.float64)
    return {
        "min": arr.min(axis=0).tolist(),
        "max": arr.max(axis=0).tolist(),
        "mean": arr_float.mean(axis=0).tolist(),
        "std": arr_float.std(axis=0).tolist(),
        "count": [int(arr.shape[0])],
    }
