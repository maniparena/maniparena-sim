#!/usr/bin/env python3
"""Teleop data collection for Bimanual and ex001.

Usage:
    # Lab/Sim 6.0 defaults to headless; pass --viz kit for the Kit viewport.
    python scripts/collect.py \
        --robot bimanual --task sort_blocks --control-mode keyboard \
        --config configs/collect/keyboard.yaml --viz kit

    python scripts/collect.py \
        --robot ex001 --task sort_blocks --control-mode keyboard \
        --config configs/collect/keyboard.yaml --viz kit

    python scripts/collect.py \
        --robot ex001 --task sort_blocks --control-mode vuer \
        --config configs/collect/vuer.yaml --viz kit

    python scripts/collect.py \
        --task sort_blocks --control-mode vr \
        --config configs/collect/vr.yaml --viz kit

    python scripts/collect.py \
        --task sort_blocks --control-mode master_slave \
        --config configs/collect/master_slave.yaml --viz kit
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
        description='Collect Bimanual teleop data.',
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
            'buttons_contact', 'dummy_task',
            'put_bottle_on_woodshelf',
        ],
    )
    parser.add_argument(
        '--control-mode', required=True,
        choices=['keyboard', 'vr', 'master_slave', 'vuer'],
    )
    parser.add_argument('--config', required=True)
    return parser.parse_args()


def _create_planner(control_mode: str, payload: dict):
    """Instantiate a TeleopPlanner for the given mode."""
    teleop_cfg = payload.get('teleop_config', {})
    step_hz = int(payload.get('step_hz', 20))
    max_steps = int(payload.get('max_steps', 400))

    if control_mode == 'keyboard':
        from maniparena_sim.planners.keyboard_teleop import (
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
        from maniparena_sim.planners.vr_teleop import (
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
        from maniparena_sim.planners.master_slave_teleop import (
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


def _create_ex001_planner(control_mode: str, payload: dict):
    """Instantiate an ex001 TeleopPlanner for the given mode."""
    teleop_cfg = payload.get('teleop_config', {})
    step_hz = int(payload.get('step_hz', 20))
    max_steps = int(payload.get('max_steps', 400))

    if control_mode == 'keyboard':
        # Same keyboard planner/device as the desktop arm; base control turns
        # on automatically via the ex001 embodiment's diff-drive cfg.
        return _create_planner('keyboard', payload)

    if control_mode in ('vr', 'vuer'):
        from maniparena_sim.planners.ex001_vr_teleop import (
            Ex001VRTeleopPlanner, Ex001VRTeleopSettings,
        )
        fields = {k: v for k, v in teleop_cfg.items() if k in Ex001VRTeleopSettings.__dataclass_fields__}
        fields['input_backend'] = 'vuer' if control_mode == 'vuer' else 'openxr'
        planner = Ex001VRTeleopPlanner()
        planner.settings = Ex001VRTeleopSettings(step_hz=step_hz, max_steps=max_steps, **fields)
        return planner

    raise ValueError(f'Unsupported ex001 control mode: {control_mode}')


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

    from maniparena_sim.environment.builder import (
        EX001_SUPPORTED_TASKS,
        build_collect_gym_env,
    )
    from maniparena_sim.environment.registry import (
        bootstrap_arena_registry,
    )
    from maniparena_sim.loops.teleop_collection import (
        run_planner_collection,
    )

    bootstrap_arena_registry()

    if args.robot == 'ex001':
        if args.task not in EX001_SUPPORTED_TASKS:
            raise ValueError(
                "ex001 robot supports tasks "
                f"{list(EX001_SUPPORTED_TASKS)}, got: {args.task}"
            )
        if control_mode not in ('keyboard', 'vr', 'vuer'):
            raise ValueError(
                "ex001 robot supports control modes "
                "keyboard/vr/vuer, "
                f"got: {control_mode}"
            )
        from maniparena_sim.environment.builder import (
            build_ex001_collect_gym_env,
        )
        ctx = build_ex001_collect_gym_env(
            args.task,
            payload,
            control_mode=control_mode,
            headless=bool(getattr(args, 'headless', False)),
            device=getattr(args, 'device', 'cuda:0'),
        )
        planner = _create_ex001_planner(control_mode, payload)
    else:
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

    if args.robot == 'ex001':
        from maniparena_sim.terms.recorders.streaming.file_session import drain_recorder_async_exports
        drain_recorder_async_exports(ctx.gym_env)

    ctx.gym_env.close()
    simulation_app.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
