#!/usr/bin/env python3
"""Closed-loop policy evaluation for Bimanual and EX001.

Usage:
    # Bimanual (16D EE)
    python scripts/eval.py \
        --robot bimanual --task sort_blocks \
        --config configs/eval/robot.yaml --viz kit

    # EX001 Wall-X whole-body (21D)
    python scripts/eval.py \
        --robot ex001 --task put_bottle_on_woodshelf \
        --config configs/eval/ex001_put_bottle.yaml --viz kit
"""

from __future__ import annotations

import argparse

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
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument(
        '--robot', default='bimanual',
        choices=['bimanual', 'ex001'],
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
            # Allow --server-address host:port when --server-port omitted.
            host, port_s = host.rsplit(':', 1)
            out['model_address'] = host
            out['model_port'] = int(port_s)
        else:
            out['model_address'] = host.split(':')[0]
    if port is not None:
        out['model_port'] = int(port)
    return out


def main() -> int:
    """Run closed-loop policy evaluation."""
    args = parse_args()
    payload = load_yaml(args.config)
    args.enable_cameras = bool(
        payload.get('enable_cameras', True)
    )

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

    if args.robot == 'ex001':
        from maniparena_sim.environment.builder import (
            EX001_SUPPORTED_TASKS,
            build_ex001_eval_gym_env,
        )
        from maniparena_sim.policy.ex001_wallx_policy import (
            Ex001WallxPolicy,
            Ex001WallxPolicyConfig,
        )

        if args.task not in EX001_SUPPORTED_TASKS:
            raise SystemExit(
                f'ex001 unsupported task: {args.task}; '
                f'supported={list(EX001_SUPPORTED_TASKS)}'
            )
        ctx = build_ex001_eval_gym_env(
            args.task, payload,
            headless=bool(getattr(args, 'headless', False)),
            device=getattr(args, 'device', 'cuda:0'),
        )
        policy = Ex001WallxPolicy(
            Ex001WallxPolicyConfig(
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
        policy_name = 'Ex001WallxPolicy'
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
        )
        policy = RobotClosedloopPolicy(
            RobotPolicyConfig(
                model_address=str(pc.get('model_address', 'localhost')),
                model_port=int(pc.get('model_port', 8000)),  # CLI may override
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

    gym_env = ctx.gym_env
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
    print('=' * 60)

    obs, _ = gym_env.reset()
    episode_count = 0
    step_count = 0
    episode_successes: list[bool] = []
    # Hard cap: each env runs at most max_steps, then we force-reset and count 1 episode.
    max_global = num_episodes * max_steps + int(gym_env.num_envs)
    has_success_term = (
        hasattr(gym_env, 'termination_manager')
        and 'success' in gym_env.termination_manager.active_terms
    )
    ep_steps = torch.zeros(
        gym_env.num_envs, dtype=torch.long, device=gym_env.device,
    )

    progress_every = max(1, min(50, int(max_steps) // 4))

    with torch.inference_mode():
        for _ in range(max_global):
            if not simulation_app.is_running():
                break
            actions = policy.get_actions(gym_env, obs)
            obs, _, terminated, truncated, _ = gym_env.step(
                actions,
            )
            step_count += 1
            ep_steps += 1
            done = terminated | truncated
            timed_out = ep_steps >= int(max_steps)
            finished = done | timed_out

            # Progress so it's obvious the per-episode counter is advancing.
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
                succ = gym_env.termination_manager.get_term(
                    'success',
                )[env_ids]
                episode_successes.extend(
                    bool(x) for x in succ.tolist()
                )
            else:
                episode_successes.extend(
                    [False] * int(env_ids.numel())
                )

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
                obs, _ = gym_env.reset(env_ids=force_ids)

            reason = 'timeout' if bool(timed_out.any()) and not bool(done.any()) else (
                'success' if has_success_term and bool(
                    gym_env.termination_manager.get_term('success')[env_ids].any()
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

    try:
        if hasattr(gym_env, 'compute_metrics'):
            metrics = gym_env.compute_metrics()
            entries = getattr(metrics, 'metric_data_entries', {}) or {}
            if entries:
                print('  Arena metrics:')
                for name, entry in entries.items():
                    print(f'    {name}: {entry.metric_value}')
    except Exception as exc:
        print(f'  Arena metrics: unavailable ({exc})')
    print('=' * 60)

    policy.cleanup()
    gym_env.close()
    simulation_app.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
