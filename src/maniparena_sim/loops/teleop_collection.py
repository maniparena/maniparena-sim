"""Unified teleop collection loop for all planner types."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from maniparena_sim.environment.builder import CollectEnv
from maniparena_sim.loops.dataset_export import export_episode
from maniparena_sim.loops.result_types import CollectionResult


def _recording_layout_from_ctx(ctx: CollectEnv):
    from maniparena_sim.loops.dataset_export import RecordingLayout

    return RecordingLayout(
        env_name=ctx.env_name,
        output_dir=ctx.output_dir,
        prefix=ctx.prefix,
        bootstrap_name=f'{ctx.prefix}_bootstrap',
        exported_paths=list(ctx.exported_paths),
        is_direct_lerobot=ctx.is_direct_lerobot,
    )


def run_planner_collection(
    ctx: CollectEnv,
    planner,
    simulation_app,
) -> CollectionResult:
    """Run teleop collection with any TeleopPlanner.

    Works with KeyboardTeleopPlanner, VuerTeleopPlanner,
    MasterSlaveTeleopPlanner, or any TeleopPlanner subclass.
    """
    gym_env = ctx.gym_env
    layout = _recording_layout_from_ctx(ctx)

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

    def _export(mark_success: bool):
        nonlocal success_count, episode_count
        episode_count += 1
        if mark_success:
            success_count += 1
        path = export_episode(gym_env, layout, 0, mark_success)
        if path:
            print(f'[record] saved episode to {path}')

    def _reset_env():
        gym_env.recorder_manager.reset([0])
        # Prefer env.reset() only: sim.reset() after Fabric is up triggers
        # "Fabric Kinematics already initialized" and can skip root pose writes.
        gym_env.reset()
        _clear_task_state()
        planner.reset()

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
