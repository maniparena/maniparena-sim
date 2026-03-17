#!/usr/bin/env python3
"""Robot closed-loop policy evaluation for Bimanual.

Usage:
    python scripts/eval.py \
        --task sort_blocks \
        --config configs/eval/robot.yaml
"""

from __future__ import annotations

import argparse

import yaml
from isaaclab.app import AppLauncher


def load_yaml(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate Robot policy on Bimanual.',
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument(
        '--task', required=True,
        choices=[
            'sort_blocks', 'fruits_to_basket',
            'buttons_contact',
        ],
    )
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = load_yaml(args.config)
    args.enable_cameras = bool(
        payload.get('enable_cameras', True)
    )

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch

    from maniparena_sim.environment.builder import build_eval_gym_env
    from maniparena_sim.environment.registry import (
        bootstrap_arena_registry,
    )
    from maniparena_sim.policy.robot_policy import (
        RobotClosedloopPolicy,
        RobotPolicyConfig,
    )

    bootstrap_arena_registry()

    ctx = build_eval_gym_env(
        args.task, payload,
        headless=bool(getattr(args, 'headless', False)),
        device=getattr(args, 'device', 'cuda:0'),
    )
    gym_env = ctx.gym_env

    pc = payload.get('policy_config', {})
    policy = RobotClosedloopPolicy(
        RobotPolicyConfig(
            model_address=str(
                pc.get('model_address', 'localhost'),
            ),
            model_port=int(pc.get('model_port', 8000)),
            instruction=str(
                pc.get('instruction', 'pick up the object'),
            ),
            action_horizon=int(
                pc.get('action_horizon', 16),
            ),
            action_chunk_length=int(
                pc.get('action_chunk_length', 4),
            ),
            pos_gain=float(pc.get('pos_gain', 1.0)),
            rot_gain=float(pc.get('rot_gain', 1.0)),
        ),
    )

    num_episodes = int(payload.get('num_episodes', 10))
    max_steps = int(payload.get('max_steps', 800))

    print('=' * 60)
    print(f'  Task:       {args.task}')
    print(f'  Policy:     RobotClosedloopPolicy')
    print(f'  Server:     {policy.cfg.model_address}'
          f':{policy.cfg.model_port}')
    print(f'  Episodes:   {num_episodes}')
    print(f'  Max steps:  {max_steps}')
    print('=' * 60)

    obs, _ = gym_env.reset()
    episode_count = 0
    step_count = 0
    max_global = num_episodes * max_steps * 2

    with torch.inference_mode():
        for _ in range(max_global):
            if not simulation_app.is_running():
                break
            actions = policy.get_actions(gym_env, obs)
            obs, _, terminated, truncated, _ = gym_env.step(
                actions,
            )
            step_count += 1
            done = terminated | truncated
            if done.any():
                env_ids = done.nonzero(
                    as_tuple=False,
                ).squeeze(-1)
                policy.reset(env_ids)
                episode_count += int(done.sum())
            if episode_count >= num_episodes:
                break

    print(
        f'\neval finished: {ctx.env_name} '
        f'episodes={episode_count} steps={step_count}'
    )

    policy.cleanup()
    gym_env.close()
    simulation_app.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
