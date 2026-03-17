"""Thin metrics adapter around Arena-native metrics."""

from __future__ import annotations

from typing import Any


class MetricProvider:
    """Default metric provider used by validation."""

    def compute(self, env: Any) -> dict:
        from isaaclab_arena.metrics.metrics import compute_metrics

        return compute_metrics(env)
