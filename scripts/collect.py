#!/usr/bin/env python3
"""Teleop data collection for EX001-6R.

Usage:
    python scripts/collect.py \
        --task sort_blocks --control-mode keyboard \
        --config configs/collect/keyboard.yaml

    python scripts/collect.py \
        --task sort_blocks --control-mode vr \
        --config configs/collect/vr.yaml

    python scripts/collect.py \
        --task sort_blocks --control-mode master_slave \
        --config configs/collect/master_slave.yaml
"""

from __future__ import annotations

import argparse

import yaml
from isaaclab.app import AppLauncher


def load_yaml(path: str) -> dict:
    """Load YAML config file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description='Collect EX001-6R teleop data.',
    )
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument(
        '--task', required=True,
        choices=[
            'sort_blocks', 'fruits_to_basket',
            'buttons_contact',
        ],
    )
    parser.add_argument(
        '--control-mode', required=True,
        choices=['keyboard', 'vr', 'master_slave'],
    )
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def _create_planner(control_mode: str, payload: dict):
    """Instantiate a TeleopPlanner for the given mode."""
    teleop_cfg = payload.get('teleop_config', {})
    step_hz = int(payload.get('step_hz', 30))
    max_steps = int(payload.get('max_steps', 400))

    if control_mode == 'keyboard':
        from opencvpr.planners.keyboard_teleop import (
            KeyboardTeleopPlanner,
            KeyboardTeleopSettings,
        )
        settings_fields = {
            k: v for k, v in teleop_cfg.items()
            if k in (
                KeyboardTeleopSettings.__dataclass_fields__
            )
        }
        planner = KeyboardTeleopPlanner()
        planner.settings = KeyboardTeleopSettings(
            step_hz=step_hz, max_steps=max_steps,
            **settings_fields,
        )
        return planner

    if control_mode == 'vr':
        from opencvpr.planners.vr_teleop import (
            VRTeleopPlanner,
            VRTeleopSettings,
        )
        settings_fields = {
            k: v for k, v in teleop_cfg.items()
            if k in VRTeleopSettings.__dataclass_fields__
        }
        planner = VRTeleopPlanner()
        planner.settings = VRTeleopSettings(
            step_hz=step_hz, max_steps=max_steps,
            **settings_fields,
        )
        return planner

    if control_mode == 'master_slave':
        from opencvpr.planners.master_slave_teleop import (
            MasterSlaveTeleopPlanner,
            MasterSlaveTeleopSettings,
        )
        settings_fields = {
            k: v for k, v in teleop_cfg.items()
            if k in (
                MasterSlaveTeleopSettings.__dataclass_fields__
            )
        }
        planner = MasterSlaveTeleopPlanner()
        planner.settings = MasterSlaveTeleopSettings(
            step_hz=step_hz, max_steps=max_steps,
            **settings_fields,
        )
        return planner

    raise ValueError(
        f'Unsupported control mode: {control_mode}'
    )


def main() -> int:
    """Collection entry point."""
    args = parse_args()
    control_mode = args.control_mode
    if (
        control_mode == 'keyboard'
        and bool(getattr(args, 'headless', False))
    ):
        raise ValueError(
            'Keyboard teleop requires windowed mode. '
            'Remove --headless.'
        )

    payload = load_yaml(args.config)
    args.enable_cameras = bool(
        payload.get('enable_cameras', True),
    )
    if control_mode == 'vr':
        args.xr = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    from opencvpr.environment.builder import (
        build_collect_gym_env,
    )
    from opencvpr.environment.registry import (
        bootstrap_arena_registry,
    )
    from opencvpr.loops.teleop_collection import (
        run_planner_collection,
    )

    bootstrap_arena_registry()
    ctx = build_collect_gym_env(
        args.task, payload,
        control_mode=control_mode,
        headless=bool(getattr(args, 'headless', False)),
        device=getattr(args, 'device', 'cuda:0'),
    )

    planner = _create_planner(control_mode, payload)

    if control_mode == 'keyboard':
        print(
            'keyboard teleop controls: '
            'B switch arm, K toggle gripper, '
            'WASDQE move, ZXTGCV rotate, '
            'H save success episode, '
            'R reset/skip episode.'
        )

    print(
        f'[INFO] collecting: task={args.task} '
        f'mode={control_mode} output={ctx.output_dir}'
    )

    result = run_planner_collection(
        ctx, planner, simulation_app,
    )

    print(
        f'\ncollection finished: {result.env_name} '
        f'episodes={result.num_episodes} '
        f'success={result.success_count}'
    )
    for p in result.exported_paths:
        print(f'exported: {p}')

    ctx.gym_env.close()
    simulation_app.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
