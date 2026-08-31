#!/usr/bin/env python3
"""Closed-loop policy evaluation for Bimanual and QUANTA_X1.

Usage:
    # Bimanual (16D EE)
    python scripts/eval.py \
        --robot bimanual --task sort_blocks \
        --config configs/eval/robot.yaml --viz kit

    # QUANTA_X1 Wall-X whole-body (21D) + fixed third-person perspective video
    python scripts/eval.py \
        --robot quanta_x1 --task put_bottle_on_woodshelf \
        --config configs/eval/quanta_x1_put_bottle.yaml \
        --video
"""

from __future__ import annotations

import argparse
import datetime
import os

import yaml
from isaaclab.app import AppLauncher


def load_yaml(path: str) -> dict:
    """Load a YAML config file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate remote policy on maniparena robots.',
    )
    parser.add_argument(
        '--video',
        action='store_true',
        default=False,
        help='Record Isaac Lab perspective video clips via env.render().',
    )
    parser.add_argument(
        '--video_length',
        type=int,
        default=None,
        help='Steps per mp4 clip when video_mode=step (default: max_steps).',
    )
    parser.add_argument(
        '--video_interval',
        type=int,
        default=None,
        help='Start a step clip every N steps when video_mode=step.',
    )
    parser.add_argument(
        '--log_dir',
        default=None,
        help='Base log directory for --video output (default: ~/maniparena_output/logs/eval).',
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument(
        '--robot', default='bimanual',
        choices=['bimanual', 'quanta_x1'],
    )
    parser.add_argument(
        '--task', required=True,
        choices=[
            'sort_blocks', 'fruits_to_basket',
            'buttons_contact', 'put_bottle_on_woodshelf',
        ],
    )
    parser.add_argument('--config', required=True)
    parser.add_argument(
        '--server-address', default=None,
        help='Override policy_config.model_address (host only, or host:port).',
    )
    parser.add_argument(
        '--server-port', type=int, default=None,
        help='Override policy_config.model_port.',
    )
    return parser.parse_args()


def _apply_server_overrides(pc: dict, args) -> dict:
    """Apply CLI server address/port onto policy_config."""
    out = dict(pc or {})
    address = getattr(args, 'server_address', None)
    port = getattr(args, 'server_port', None)
    if address:
        host = str(address).strip()
        if ':' in host and port is None:
            host, port_s = host.rsplit(':', 1)
            out['model_address'] = host
            out['model_port'] = int(port_s)
        else:
            out['model_address'] = host.split(':')[0]
    if port is not None:
        out['model_port'] = int(port)
    return out


def _env_core(gym_env):
    """Return the underlying Isaac Lab env when RecordVideo wraps it."""
    return getattr(gym_env, 'unwrapped', gym_env)


def _resolve_log_dir(args, payload, robot: str, task: str) -> str:
    base = args.log_dir or payload.get('log_dir') or '~/maniparena_output/logs/eval'
    base = os.path.expanduser(str(base))
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    return os.path.join(base, f'{robot}_{task}', timestamp)


def _video_record_wrapper(gym_env):
    """Return the RecordVideo wrapper if ``gym_env`` is wrapped."""
    from gymnasium.wrappers import RecordVideo

    env = gym_env
    while env is not None:
        if isinstance(env, RecordVideo):
            return env
        env = getattr(env, 'env', None)
    return None


def _start_episode_video(gym_env, episode_index: int) -> None:
    wrapper = _video_record_wrapper(gym_env)
    if wrapper is None:
        return
    name = f'{wrapper.name_prefix}-eval-episode-{episode_index:04d}'
    wrapper.start_recording(name)
    wrapper._capture_frame()


def _finish_episode_video(gym_env, episode_index: int) -> None:
    wrapper = _video_record_wrapper(gym_env)
    if wrapper is None or not wrapper.recording:
        return
    name = wrapper._video_name
    wrapper.stop_recording()
    if name:
        path = os.path.join(wrapper.video_folder, f'{name}.mp4')
        print(f'[video] saved episode {episode_index} -> {path}')


def _reset_eval_env(gym_env, core, env_ids=None):
    """Reset while bypassing RecordVideo for partial ``env_ids`` resets."""
    if env_ids is None:
        return gym_env.reset()
    if _video_record_wrapper(gym_env) is not None:
        return core.reset(env_ids=env_ids)
    return gym_env.reset(env_ids=env_ids)


def _wrap_record_video(gym_env, log_dir: str, args, payload: dict):
    if not args.video:
        return gym_env, None

    import gymnasium as gym

    video_mode = str(payload.get('video_mode', 'episode')).lower()
    max_steps = int(payload.get('max_steps', 800))
    video_folder = os.path.join(log_dir, 'videos', 'eval')
    os.makedirs(video_folder, exist_ok=True)

    if video_mode == 'step':
        video_length = args.video_length or int(payload.get('video_length', max_steps))
        video_interval = args.video_interval or int(
            payload.get('video_interval', max_steps),
        )
        print(f'[video] perspective clips -> {video_folder}')
        print(f'[video] mode=step length={video_length} interval={video_interval}')
        wrapped = gym.wrappers.RecordVideo(
            gym_env,
            video_folder=video_folder,
            step_trigger=lambda step: step % video_interval == 0,
            video_length=video_length,
            disable_logger=True,
        )
        return wrapped, video_folder

    print(f'[video] perspective clips -> {video_folder}')
    print('[video] mode=episode (flush mp4 when each eval episode finishes)')
    wrapped = gym.wrappers.RecordVideo(
        gym_env,
        video_folder=video_folder,
        episode_trigger=lambda _ep: False,
        step_trigger=lambda _step: False,
        video_length=0,
        disable_logger=True,
    )
    return wrapped, video_folder


def _export_finished_episodes(
    gym_env,
    ctx,
    env_ids,
    *,
    has_success_term: bool,
) -> list[str]:
    """Persist finished episodes using the same path as teleop collection."""
    from maniparena_sim.loops.dataset_export import export_episode

    if ctx.recording is None:
        return []

    saved: list[str] = []
    for env_id in env_ids.tolist():
        env_id = int(env_id)
        mark_success = False
        if has_success_term:
            mark_success = bool(
                gym_env.termination_manager.get_term('success')[env_id].item()
            )
        path = export_episode(
            gym_env, ctx.recording, env_id, mark_success,
        )
        if path:
            saved.append(path)
            print(f'[record] episode saved to {path}')
    return saved


def main() -> int:
    """Run closed-loop policy evaluation."""
    args = parse_args()
    payload = load_yaml(args.config)
    args.enable_cameras = bool(
        payload.get('enable_cameras', True)
    ) or bool(args.video)

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch

    from maniparena_sim.environment.registry import (
        bootstrap_arena_registry,
    )

    bootstrap_arena_registry()
    pc = _apply_server_overrides(payload.get('policy_config', {}), args)
    print(
        f"[eval] policy server "
        f"{pc.get('model_address', 'localhost')}:{pc.get('model_port', 8000)}"
    )

    render_mode = 'rgb_array' if args.video else None
    log_dir = _resolve_log_dir(args, payload, args.robot, args.task) if args.video else None

    if args.robot == 'quanta_x1':
        from maniparena_sim.environment.builder import (
            QUANTA_X1_SUPPORTED_TASKS,
            build_quanta_x1_eval_gym_env,
        )
        from maniparena_sim.policy.quanta_x1_wallx_policy import (
            QuantaX1WallxPolicy,
            QuantaX1WallxPolicyConfig,
        )

        if args.task not in QUANTA_X1_SUPPORTED_TASKS:
            raise SystemExit(
                f'quanta_x1 unsupported task: {args.task}; '
                f'supported={list(QUANTA_X1_SUPPORTED_TASKS)}'
            )
        ctx = build_quanta_x1_eval_gym_env(
            args.task, payload,
            headless=bool(getattr(args, 'headless', False)),
            device=getattr(args, 'device', 'cuda:0'),
            render_mode=render_mode,
        )
        policy = QuantaX1WallxPolicy(
            QuantaX1WallxPolicyConfig(
                model_address=str(pc.get('model_address', 'localhost')),
                model_port=int(pc.get('model_port', 8000)),
                instruction=str(
                    pc.get(
                        'instruction',
                        'pick the bottle from the table and place it on the wooden shelf',
                    ),
                ),
                action_horizon=int(pc.get('action_horizon', 32)),
                action_chunk_length=int(pc.get('action_chunk_length', 32)),
                interpolation_multiplier=int(
                    pc.get('interpolation_multiplier', 2),
                ),
                ee_pose_normalize=bool(pc.get('ee_pose_normalize', False)),
                pos_gain=float(pc.get('pos_gain', 1.0)),
                rot_gain=float(pc.get('rot_gain', 1.0)),
                wheel_radius=float(pc.get('wheel_radius', 0.084)),
                wheel_track_width=float(pc.get('wheel_track_width', 0.458)),
            ),
        )
        policy_name = 'QuantaX1WallxPolicy'
    else:
        from maniparena_sim.environment.builder import build_eval_gym_env
        from maniparena_sim.policy.robot_policy import (
            RobotClosedloopPolicy,
            RobotPolicyConfig,
        )

        ctx = build_eval_gym_env(
            args.task, payload,
            headless=bool(getattr(args, 'headless', False)),
            device=getattr(args, 'device', 'cuda:0'),
            render_mode=render_mode,
        )
        policy = RobotClosedloopPolicy(
            RobotPolicyConfig(
                model_address=str(pc.get('model_address', 'localhost')),
                model_port=int(pc.get('model_port', 8000)),
                instruction=str(pc.get('instruction', 'pick up the object')),
                action_horizon=int(pc.get('action_horizon', 32)),
                action_chunk_length=int(pc.get('action_chunk_length', 32)),
                interpolation_multiplier=int(
                    pc.get('interpolation_multiplier', 2),
                ),
                ee_pose_normalize=bool(pc.get('ee_pose_normalize', True)),
                pos_gain=float(pc.get('pos_gain', 1.0)),
                rot_gain=float(pc.get('rot_gain', 1.0)),
            ),
        )
        policy_name = 'RobotClosedloopPolicy'

    gym_env, video_folder = _wrap_record_video(ctx.gym_env, log_dir, args, payload)
    core = _env_core(gym_env)
    num_episodes = int(payload.get('num_episodes', 10))
    max_steps = int(payload.get('max_steps', 800))

    print('=' * 60)
    print(f'  Robot:      {args.robot}')
    print(f'  Task:       {args.task}')
    print(f'  Policy:     {policy_name}')
    print(f'  Server:     {policy.cfg.model_address}'
          f':{policy.cfg.model_port}')
    print(f'  Episodes:   {num_episodes}')
    print(f'  Max steps:  {max_steps} (per episode)')
    print(f'  Output:     {ctx.output_dir}')
    if video_folder:
        print(f'  Video:      {video_folder}')
    if ctx.recording is not None:
        print(f'  Recording:  {ctx.recording.fmt} (teleop-compatible export)')
    print('=' * 60)

    obs, _ = gym_env.reset()
    if args.video and str(payload.get('video_mode', 'episode')).lower() != 'step':
        _start_episode_video(gym_env, 1)
    episode_count = 0
    step_count = 0
    episode_successes: list[bool] = []
    saved_paths: list[str] = []
    max_global = num_episodes * max_steps + int(core.num_envs)
    has_success_term = (
        hasattr(core, 'termination_manager')
        and 'success' in core.termination_manager.active_terms
    )
    ep_steps = torch.zeros(
        core.num_envs, dtype=torch.long, device=core.device,
    )
    progress_every = max(1, min(50, int(max_steps) // 4))

    with torch.inference_mode():
        for _ in range(max_global):
            if not simulation_app.is_running():
                break
            actions = policy.get_actions(core, obs)
            obs, _, terminated, truncated, _ = gym_env.step(
                actions,
            )
            step_count += 1
            ep_steps += 1
            done = terminated | truncated
            timed_out = ep_steps >= int(max_steps)
            finished = done | timed_out

            if int(ep_steps[0].item()) % progress_every == 0:
                print(
                    f'[episode] {episode_count + 1}/{num_episodes} '
                    f'step {int(ep_steps[0].item())}/{max_steps}'
                )

            if not finished.any():
                continue

            env_ids = finished.nonzero(
                as_tuple=False,
            ).squeeze(-1)
            if env_ids.ndim == 0:
                env_ids = env_ids.unsqueeze(0)

            if has_success_term:
                succ = core.termination_manager.get_term(
                    'success',
                )[env_ids]
                episode_successes.extend(
                    bool(x) for x in succ.tolist()
                )
            else:
                episode_successes.extend(
                    [False] * int(env_ids.numel())
                )

            saved_paths.extend(
                _export_finished_episodes(
                    core,
                    ctx,
                    env_ids,
                    has_success_term=has_success_term,
                )
            )

            finished_episode_no = episode_count + 1
            episode_video = (
                args.video
                and str(payload.get('video_mode', 'episode')).lower() != 'step'
            )
            if episode_video:
                _finish_episode_video(gym_env, finished_episode_no)

            # Natural done already auto-reset inside step(); only force-reset timeouts.
            force_ids = (~done & timed_out).nonzero(
                as_tuple=False,
            ).squeeze(-1)
            if force_ids.ndim == 0:
                force_ids = force_ids.unsqueeze(0)
            if force_ids.numel() > 0:
                print(
                    f'[episode] timeout after {max_steps} steps -> reset '
                    f'(env_ids={force_ids.tolist()})'
                )
                obs, _ = _reset_eval_env(gym_env, core, env_ids=force_ids)

            if episode_video and finished_episode_no < num_episodes:
                _start_episode_video(gym_env, finished_episode_no + 1)

            reason = 'timeout' if bool(timed_out.any()) and not bool(done.any()) else (
                'success' if has_success_term and bool(
                    core.termination_manager.get_term('success')[env_ids].any()
                ) else 'done'
            )
            print(
                f'[episode] finished #{episode_count + 1} '
                f'reason={reason} ep_steps={ep_steps[env_ids].tolist()}'
            )

            policy.reset(env_ids)
            ep_steps[env_ids] = 0
            episode_count += int(finished.sum())
            if episode_count >= num_episodes:
                break

    if len(episode_successes) > num_episodes:
        episode_successes = episode_successes[:num_episodes]
        episode_count = num_episodes

    num_success = sum(episode_successes)
    num_recorded = len(episode_successes)
    success_rate = (
        num_success / num_recorded if num_recorded else 0.0
    )

    print(
        f'\neval finished: {ctx.env_name} '
        f'episodes={episode_count} steps={step_count}'
    )
    print('=' * 60)
    print('  Success summary')
    print(f'  Episodes:     {num_recorded}')
    print(f'  Successes:    {num_success}')
    print(f'  Failures:     {num_recorded - num_success}')
    print(f'  Success rate: {success_rate:.1%} ({num_success}/{num_recorded})')
    if episode_successes:
        flags = ' '.join(
            'S' if ok else 'F' for ok in episode_successes
        )
        print(f'  Per-episode:  [{flags}]')
    if saved_paths:
        print('  Saved datasets:')
        for path in saved_paths:
            print(f'    {path}')
    elif ctx.recording is not None:
        print('  Saved datasets: (none — no episode finished)')
    if ctx.recording is not None and ctx.recording.exported_paths:
        print('  LeRobot roots:')
        for path in ctx.recording.exported_paths:
            print(f'    {path}')

    try:
        if hasattr(core, 'compute_metrics'):
            metrics = core.compute_metrics()
            entries = getattr(metrics, 'metric_data_entries', {}) or {}
            if entries:
                print('  Arena metrics:')
                for name, entry in entries.items():
                    print(f'    {name}: {entry.metric_value}')
    except Exception as exc:
        print(f'  Arena metrics: unavailable ({exc})')
    print('=' * 60)

    if args.robot == 'quanta_x1' and ctx.recording is not None:
        from maniparena_sim.terms.recorders.streaming.file_session import (
            drain_recorder_async_exports,
        )
        drain_recorder_async_exports(core)

    policy.cleanup()
    gym_env.close()
    simulation_app.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
