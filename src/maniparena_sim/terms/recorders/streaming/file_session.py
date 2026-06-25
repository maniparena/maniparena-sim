"""Per-episode HDF5 export helpers (trimmed port of manaenv file_session)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def drain_recorder_async_exports(gym_env: Any, *, timeout: float | None = None) -> None:
    rm = getattr(gym_env, "recorder_manager", None)
    if rm is None:
        return
    writer = getattr(rm, "_mana_async_writer", None)
    if writer is not None:
        writer.barrier(timeout=timeout)
    vm = getattr(rm, "_mana_sidecar_video_manager", None)
    if vm is not None:
        vm.barrier()


def install_episode_handler(gym_env: Any, *, output_file_stem: str, output_dir: str | None = None) -> Any:
    rm = gym_env.recorder_manager
    if output_dir is not None:
        rm.cfg.dataset_export_dir_path = output_dir
        os.makedirs(output_dir, exist_ok=True)
    rm.cfg.dataset_filename = output_file_stem
    target_dir = output_dir if output_dir is not None else rm.cfg.dataset_export_dir_path
    env_name = getattr(gym_env.cfg, "env_name", None)
    handler = rm.cfg.dataset_file_handler_class_type()
    handler.create(os.path.join(target_dir, output_file_stem), env_name=env_name)
    rm._dataset_file_handler = handler
    return handler


def close_episode_handler(gym_env: Any) -> None:
    rm = getattr(gym_env, "recorder_manager", None)
    if rm is None:
        return
    writer = getattr(rm, "_mana_async_writer", None)
    if writer is not None:
        try:
            writer.barrier()
        except Exception:
            pass
    handler = getattr(rm, "_dataset_file_handler", None)
    if handler is None:
        return
    try:
        handler.flush()
    except Exception:
        pass
    try:
        handler.close()
    except Exception:
        pass
    rm._dataset_file_handler = None


def export_recorded_episode_to_hdf5(
    gym_env: Any, *, output_dir: str, output_file_stem: str,
    env_ids: list[int] | None = None, mark_success: bool = True,
) -> str | None:
    import torch
    rm = getattr(gym_env, "recorder_manager", None)
    if rm is None:
        return None
    if env_ids is None:
        env_ids = list(range(int(getattr(gym_env, "num_envs", 1) or 1)))
    elif isinstance(env_ids, torch.Tensor):
        env_ids = env_ids.tolist()
    flags = [bool(mark_success)] * len(env_ids)
    rm.record_pre_reset(env_ids, force_export_or_skip=False)
    rm.set_success_to_episodes(env_ids, torch.tensor(flags, dtype=torch.bool, device=gym_env.device))
    install_episode_handler(gym_env, output_file_stem=output_file_stem, output_dir=output_dir)
    writer = getattr(rm, "_mana_async_writer", None)
    try:
        rm.export_episodes(env_ids)
    finally:
        if writer is not None:
            writer.barrier()
        close_episode_handler(gym_env)
    return str(Path(os.path.expanduser(output_dir)) / f"{output_file_stem}.hdf5")
