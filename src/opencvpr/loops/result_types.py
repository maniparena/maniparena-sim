"""Result objects returned by collection and validation loops."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CollectionResult:
    env_name: str
    prompt: str
    hdf5_path: Path | None = None
    exported_paths: list[str] = field(default_factory=list)
    num_episodes: int = 0
    num_frames: int = 0
    success_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.num_episodes, 1)


@dataclass
class ValidationResult:
    env_name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    output_dir: str | None = None
