#!/usr/bin/env python3
"""Replay recorded episodes for Bimanual.

Usage:
    python scripts/replay.py \
        --task sort_blocks \
        --config configs/replay/hdf5.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from isaaclab.app import AppLauncher


def load_yaml(path: str) -> dict:
    """Load YAML config file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Replay Bimanual recorded episodes.',
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
    """Replay entry point."""
    args = parse_args()
    payload = load_yaml(args.config)
    args.enable_cameras = bool(
        payload.get('enable_cameras', True)
    )

    replay_mode = str(payload.get('replay_mode', 'state'))
    episode = int(payload.get('episode', 0))
    dataset_path = str(
        payload.get(
            'dataset_path',
            '~/maniparena_output/recordings/',
        )
    )

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import torch

    from opencvpr.environment.builder import (
        build_replay_gym_env,
    )
    from opencvpr.environment.registry import (
        bootstrap_arena_registry,
    )
    from opencvpr.loops.replay_loop import (
        restore_initial_state,
        run_ee_replay,
        run_joint_replay,
        run_state_replay,
    )
    from opencvpr.terms.replay.reader import (
        ReplayDataReader,
    )

    bootstrap_arena_registry()

    ctx = build_replay_gym_env(
        args.task, payload,
        replay_mode=replay_mode,
        headless=bool(getattr(args, 'headless', False)),
        device=getattr(args, 'device', 'cuda:0'),
    )
    gym_env = ctx.gym_env

    reader = ReplayDataReader()
    replay_data = reader.read(
        dataset_path,
        replay_mode=replay_mode,
        episode=episode,
        device=str(gym_env.device),
    )

    gym_env.reset()
    gym_env.recorder_manager.reset([0])

    initial_state = reader.read_initial_state(
        replay_data, device=str(gym_env.device),
    )
    if initial_state is not None:
        restore_initial_state(gym_env, initial_state)

    print('=' * 60)
    print(f'  Task:     {args.task}')
    print(f'  Mode:     {replay_mode}')
    print(f'  Format:   {replay_data.dataset_format}')
    print(f'  Replay:   {replay_data.replay_description}')
    print(f'  Source:   {replay_data.dataset_label}')
    print(f'  Episode:  {episode}')
    print('=' * 60)

    if replay_mode == 'state':
        step_count = run_state_replay(
            gym_env, replay_data, simulation_app,
        )
    elif replay_mode == 'joint':
        step_count = run_joint_replay(
            gym_env, replay_data.joint_sequence,
            simulation_app,
        )
    else:
        step_count = run_ee_replay(
            gym_env, replay_data.ee_sequence,
            simulation_app,
        )

    if step_count > 0 and ctx.export_lerobot:
        rm = gym_env.recorder_manager
        rm.set_success_to_episodes(
            [0],
            torch.tensor(
                [True], dtype=torch.bool,
                device=gym_env.device,
            ),
        )
        rm.export_episodes([0])

        hdf5_bootstrap = (
            Path(ctx.output_dir)
            / f'{ctx.env_name}_bootstrap.hdf5'
        )
        if hdf5_bootstrap.exists():
            hdf5_bootstrap.unlink()

        print(
            f'\n[DONE] Replayed {step_count} frames, '
            f'exported LeRobot dataset.'
        )
        for p in ctx.exported_paths:
            print(f'  exported: {p}')
    elif step_count > 0:
        print(f'\n[DONE] Replayed {step_count} frames.')
    else:
        print(
            '\n[WARN] 0 frames replayed, skip export.'
        )

    replay_data.close()
    gym_env.close()
    simulation_app.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
