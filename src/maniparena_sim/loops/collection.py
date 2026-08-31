"""Collection runners and typed configs."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from isaacsim.core.utils import stage as stage_utils

from maniparena_sim.planners.quanta_x1_vr_teleop import VuerTeleopPlanner
from maniparena_sim.planners.keyboard_teleop import KeyboardTeleopPlanner
from maniparena_sim.planners.master_slave_teleop import MasterSlaveTeleopPlanner
from maniparena_sim.planners.planner_base import PlannerBase
from maniparena_sim.loops.result_types import CollectionResult
from maniparena_sim.loops.teleop_collection import run_teleop_collection_loop
from maniparena_sim.terms.recorders.dataset_handlers.bimanual_lerobot.helpers import (
    create_bimanual_lerobot_handler_type,
    resolve_bimanual_lerobot_dataset_layout,
)


@dataclass
class CollectConfig:
    task_name: str
    control_mode: str
    num_demos: int = 10
    step_hz: int = 20
    max_steps: int = 400
    working_path: str = "~/maniparena_output/recordings/"
    format: str = "bimanual_lerobot"
    teleop_config: dict[str, Any] = field(default_factory=dict)


TELEOP_PLANNER_TYPES = {
    "keyboard": KeyboardTeleopPlanner,
    "vuer": VuerTeleopPlanner,
    "vr": VuerTeleopPlanner,
    "master_slave": MasterSlaveTeleopPlanner,
}


def create_teleop_planner(config: CollectConfig) -> PlannerBase:
    if config.control_mode not in TELEOP_PLANNER_TYPES:
        raise ValueError(f"Unsupported control_mode '{config.control_mode}'")
    planner = TELEOP_PLANNER_TYPES[config.control_mode]()
    planner.settings = planner.Settings(
        step_hz=config.step_hz,
        max_steps=config.max_steps,
        **{k: v for k, v in config.teleop_config.items() if k in getattr(planner.Settings, "__dataclass_fields__", {})},
    )
    return planner


def _resolve_dataset_export(environment, env_name: str, prompt: str, planner: PlannerBase, config: CollectConfig):
    if config.format == "hdf5":
        return None, {}, []
    if config.format != "bimanual_lerobot":
        return None, {}, []
    dataset_fps = environment.dataset_fps or float(config.step_hz)
    layout = resolve_bimanual_lerobot_dataset_layout(
        save_path=config.working_path,
        env_name=env_name,
        device_name=planner.device_name,
    )
    handler_type = create_bimanual_lerobot_handler_type(
        save_path=config.working_path,
        env_name=env_name,
        task_name=prompt or env_name,
        fps=dataset_fps,
        device_name=planner.device_name,
    )
    return handler_type, layout, [layout["joint_root"], layout["ee_root"]]


def run_collection_loop(
    environment,
    planner: PlannerBase,
    num_demos: int,
    output_dir: str,
    output_file_name: str,
    dataset_file_handler_class_type: type | None = None,
):
    stage_utils.create_new_stage()
    gym_env = environment.build_gym_environment(
        output_dir=output_dir,
        output_file_name=output_file_name,
        enable_recording=True,
        dataset_file_handler_class_type=dataset_file_handler_class_type,
    )
    if gym_env is None:
        return 0, 0, 0
    n = gym_env.num_envs
    post_done_grace = 5
    num_episodes = 0
    total_frames = 0
    success_count = 0
    failed_count = 0
    max_total_failed = 6000 * n
    try:
        planner.initial_plan(gym_env)
        obs = gym_env.observation_manager.compute()
        while success_count < num_demos:
            actions = planner.get_actions(gym_env, obs)
            obs, _, terminated, truncated, _ = gym_env.step(actions)
            total_frames += n
            reset_mask = terminated | truncated
            if not reset_mask.any():
                planner.handle_planner_done(gym_env, post_done_grace)
                if failed_count >= max_total_failed:
                    break
                continue
            success_dones = gym_env.termination_manager.get_term("success")
            auto_reset_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)
            env_id_to_success = {int(i): bool(success_dones[i]) for i in auto_reset_ids.tolist()}
            for idx in auto_reset_ids.tolist():
                num_episodes += 1
                if bool(success_dones[idx]):
                    success_count += 1
                else:
                    failed_count += 1
            planner.on_reset(gym_env, obs, auto_reset_ids, env_id_to_success)
            planner.handle_planner_done(gym_env, post_done_grace, skip_ids=set(auto_reset_ids.tolist()))
            if failed_count >= max_total_failed:
                break
    finally:
        gym_env.close()
        environment.is_running = False
    return num_episodes, total_frames, success_count


def collect(environment, config: CollectConfig) -> CollectionResult:
    env_name = environment.name
    prompt = environment.get_prompt()
    planner = create_teleop_planner(config)
    if not planner.setup(environment, environment.task):
        raise RuntimeError("Planner setup failed")
    handler_type, layout, exported_paths = _resolve_dataset_export(environment, env_name, prompt, planner, config)
    if handler_type is None:
        working_path = Path(os.path.expanduser(config.working_path)) / env_name
        working_path.mkdir(parents=True, exist_ok=True)
        hdf5_path = working_path / "demo.hdf5"
    else:
        working_path = Path(layout["base_root"]).expanduser()
        working_path.mkdir(parents=True, exist_ok=True)
        hdf5_path = None

    if isinstance(planner, (KeyboardTeleopPlanner, VuerTeleopPlanner, MasterSlaveTeleopPlanner)):
        num_episodes, num_frames, success_count = run_teleop_collection_loop(
            environment=environment,
            planner=planner,
            env_name=env_name,
            output_dir=str(working_path),
            dataset_file_handler_class_type=handler_type,
        )
    else:
        num_episodes, num_frames, success_count = run_collection_loop(
            environment=environment,
            planner=planner,
            num_demos=config.num_demos,
            output_dir=str(working_path),
            output_file_name="demo",
            dataset_file_handler_class_type=handler_type,
        )

    return CollectionResult(
        env_name=env_name,
        prompt=prompt,
        hdf5_path=hdf5_path,
        exported_paths=exported_paths,
        num_episodes=num_episodes,
        num_frames=num_frames,
        success_count=success_count,
        metadata={
            "task_name": config.task_name,
            "control_mode": config.control_mode,
            "step_hz": config.step_hz,
            "format": config.format,
        },
    )
