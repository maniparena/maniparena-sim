"""Validation runner and typed eval config."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from isaacsim.core.utils import stage as stage_utils

from maniparena_sim.loops.result_types import ValidationResult
from maniparena_sim.terms.metrics.provider import MetricProvider


class PolicyLike(Protocol):
    def get_actions(self, env: Any, obs: dict[str, Any]) -> torch.Tensor: ...

    def reset(self, env_ids: torch.Tensor) -> None: ...


@dataclass
class EvalConfig:
    task_name: str
    num_episodes: int = 10
    max_steps: int = 800
    working_path: str = "~/maniparena_output/eval/"
    policy_type: str = ""
    policy_config: dict[str, Any] = field(default_factory=dict)


def validate_policy(policy: PolicyLike, environment, config: EvalConfig) -> ValidationResult:
    benchmark = None
    try:
        from isaacsim.core.utils.extensions import enable_extension

        enable_extension("isaacsim.benchmark.services")
        from isaacsim.benchmark.services import BaseIsaacBenchmark

        benchmark = BaseIsaacBenchmark(
            benchmark_name="bimanual_eval",
            workflow_metadata={"metadata": [{"name": "task", "data": environment.name}, {"name": "num_episodes", "data": config.num_episodes}]},
            backend_type="JSONFileMetrics",
        )
    except Exception:
        benchmark = None

    output_dir = os.path.expanduser(config.working_path)
    os.makedirs(output_dir, exist_ok=True)
    stage_utils.create_new_stage()
    gym_env = environment.build_gym_environment(mode="evaluate", output_dir=output_dir, output_file_name="eval")
    if gym_env is None:
        return ValidationResult(env_name=environment.name, metrics={}, output_dir=output_dir)
    if benchmark is not None:
        benchmark.set_phase("evaluation")
    max_steps_per_ep = int(gym_env.max_episode_length) if hasattr(gym_env, "max_episode_length") else config.max_steps
    max_global_steps = config.num_episodes * max_steps_per_ep * 2
    episode_count = 0
    obs, _ = gym_env.reset()
    try:
        with torch.inference_mode():
            for _ in range(max_global_steps):
                actions = policy.get_actions(gym_env, obs)
                obs, _, terminated, truncated, _ = gym_env.step(actions)
                done = terminated | truncated
                if done.any():
                    env_ids = done.nonzero(as_tuple=False).squeeze(-1)
                    policy.reset(env_ids)
                    episode_count += int(done.sum())
                if episode_count >= config.num_episodes:
                    break
    finally:
        metrics: dict[str, Any] = {}
        try:
            metrics = MetricProvider().compute(gym_env)
        except Exception:
            metrics = {"num_episodes": episode_count}
        try:
            gym_env.close()
        except Exception:
            pass
    return ValidationResult(env_name=environment.name, metrics=metrics, output_dir=output_dir)
