"""Unified teleop collection loop for all planner types."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch

from opencvpr.environment.builder import CollectEnv
from opencvpr.loops.result_types import CollectionResult


def run_planner_collection(
    ctx: CollectEnv,
    planner,
    simulation_app,
) -> CollectionResult:
    """Run teleop collection with any TeleopPlanner.

    Works with KeyboardTeleopPlanner, VRTeleopPlanner,
    MasterSlaveTeleopPlanner, or any TeleopPlanner subclass.
    """
    gym_env = ctx.gym_env
    rm = gym_env.recorder_manager

    planner_env = SimpleNamespace(
        device=gym_env.device,
        embodiment=ctx.embodiment,
        action_space=gym_env.action_space,
    )
    if hasattr(gym_env, 'single_action_space'):
        planner_env.single_action_space = (
            gym_env.single_action_space
        )
    planner.setup(planner_env, ctx.task)

    success_count = 0
    episode_count = 0
    total_frames = 0

    def _clear_task_state():
        for attr in ('_buttons_contact_history_state',):
            if hasattr(gym_env, attr):
                getattr(gym_env, attr).zero_()

    def _next_hdf5_name():
        idx = 0
        while True:
            name = f'{ctx.prefix}_episode{idx}'
            path = os.path.join(
                ctx.output_dir, f'{name}.hdf5',
            )
            if not os.path.exists(path):
                return name
            idx += 1

    def _export(mark_success: bool):
        nonlocal success_count, episode_count
        episode_count += 1
        if mark_success:
            success_count += 1
        rm.record_pre_reset(
            [0], force_export_or_skip=False,
        )
        rm.set_success_to_episodes(
            [0],
            torch.tensor(
                [mark_success], dtype=torch.bool,
                device=gym_env.device,
            ),
        )
        if ctx.is_direct_lerobot:
            rm.export_episodes([0])
            return
        ep_name = _next_hdf5_name()
        rm.cfg.dataset_filename = ep_name
        fh = rm.cfg.dataset_file_handler_class_type()
        fh.create(
            os.path.join(ctx.output_dir, ep_name),
            env_name=ctx.env_name,
        )
        rm._dataset_file_handler = fh
        rm.export_episodes([0])
        fh.close()
        rm._dataset_file_handler = None

    def _reset_env():
        rm.reset([0])
        gym_env.sim.reset()
        gym_env.reset()
        _clear_task_state()
        planner.reset()

    gym_env.sim.reset()
    gym_env.reset()
    _clear_task_state()
    planner.prepare_episode(gym_env, {})
    act_dim = gym_env.action_manager.action.shape[-1]

    while simulation_app.is_running():
        if planner.done_signal:
            if planner.success_signal:
                _export(mark_success=True)
                _reset_env()
                print(
                    f'[episode {episode_count}] saved '
                    f'SUCCESS (total={success_count})'
                )
            else:
                _reset_env()
                print('[episode] reset/skipped')
            planner.prepare_episode(gym_env, {})
            continue

        action = planner.get_actions(gym_env, {})
        if action.shape[-1] != act_dim:
            buf = torch.zeros(
                1, act_dim, device=gym_env.device,
            )
            n = min(action.shape[-1], act_dim)
            buf[:, :n] = action[:, :n]
            action = buf

        obs, _, terminated, truncated, _ = gym_env.step(
            action,
        )
        total_frames += 1

        if (terminated | truncated).any().item():
            _clear_task_state()
            planner.reset()
            planner.prepare_episode(gym_env, obs)
            continue

    planner.cleanup()
    return CollectionResult(
        env_name=ctx.env_name,
        prompt=ctx.task.get_prompt(),
        exported_paths=ctx.exported_paths,
        num_episodes=episode_count,
        num_frames=total_frames,
        success_count=success_count,
    )
