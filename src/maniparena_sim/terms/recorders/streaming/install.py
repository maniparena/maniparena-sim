"""Monkey-patch a live RecorderManager to stage obs off-GPU + stream MP4.

Trimmed port of manaenv recorder_runtime/install.py: keeps install_staging,
install_sync_export_sidecar_finalize, and install_async_writer. Drops
phase-segments, language-instruction stamping, and x2robot mirroring.
"""

from __future__ import annotations

from typing import Any

from maniparena_sim.terms.recorders.streaming.async_writer import AsyncRecorderWriter
from maniparena_sim.terms.recorders.streaming.staged_episode_data import (
    ManaStagedEpisodeData, is_staged_episode_data, staged_episode_data_class,
)
from maniparena_sim.terms.recorders.streaming.video_sidecar import SidecarVideoSession


def install_staging(recorder_manager: Any, *, pinned_memory: bool = True, chunk_size: int = 32, video_fps: float = 20.0) -> bool:
    rm = recorder_manager
    if rm is None or not getattr(rm, "active_terms", None):
        return False
    episodes = getattr(rm, "_episodes", None)
    if not isinstance(episodes, dict):
        return False
    staged_cls = staged_episode_data_class()
    vm = getattr(rm, "_mana_sidecar_video_manager", None)
    if vm is None:
        vm = SidecarVideoSession(recorder_manager=rm, fps=video_fps)
        rm._mana_sidecar_video_manager = vm
    for env_id, old in list(episodes.items()):
        new = ManaStagedEpisodeData(pinned=pinned_memory, chunk_size=chunk_size, video_sink=vm)
        new._env_id = env_id
        new._seed = getattr(old, "_seed", None)
        new._success = getattr(old, "_success", None)
        episodes[env_id] = new
    orig_reset = rm.reset
    orig_add = rm.add_to_episodes

    def _reset(env_ids=None, _orig=orig_reset, _rm=rm, _cls=staged_cls, _vm=vm, _p=pinned_memory, _c=chunk_size):
        targets = env_ids
        if targets is None:
            targets = list(_rm._episodes.keys())
        elif hasattr(targets, "tolist"):
            targets = targets.tolist()
        elif isinstance(targets, int):
            targets = [targets]
        for env_id in list(targets):
            _vm.reset_env(int(env_id))
        result = _orig(env_ids=env_ids)
        for env_id, ep in list(_rm._episodes.items()):
            if not isinstance(ep, _cls):
                r = ManaStagedEpisodeData(pinned=_p, chunk_size=_c, video_sink=_vm)
                r._env_id = env_id
                _rm._episodes[env_id] = r
        return result

    def _add(key, value, env_ids=None, _orig=orig_add, _rm=rm, _cls=staged_cls, _vm=vm, _p=pinned_memory, _c=chunk_size):
        for env_id, ep in list(_rm._episodes.items()):
            if not isinstance(ep, _cls):
                r = ManaStagedEpisodeData(pinned=_p, chunk_size=_c, video_sink=_vm)
                r._env_id = env_id
                _rm._episodes[env_id] = r
        return _orig(key, value, env_ids=env_ids)

    rm.reset = _reset
    rm.add_to_episodes = _add
    rm._mana_staging_installed = True
    assert is_staged_episode_data(next(iter(episodes.values())))
    return True


def install_sync_export_sidecar_finalize(recorder_manager: Any) -> None:
    rm = recorder_manager
    vm = getattr(rm, "_mana_sidecar_video_manager", None)
    if vm is None or getattr(rm, "_mana_sync_finalize", False):
        return
    orig_export = rm.export_episodes

    def _export(env_ids=None, demo_ids=None, _orig=orig_export, _vm=vm, _rm=rm):
        import torch
        if env_ids is None:
            env_ids = list(range(_rm._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        hdf5_path = ""
        handler = getattr(_rm, "_dataset_file_handler", None)
        stream = getattr(handler, "_hdf5_file_stream", None)
        if stream is not None:
            hdf5_path = getattr(stream, "filename", "") or ""
        keep_by_env = {}
        for env_id in env_ids:
            ep = _rm._episodes.get(env_id)
            keep_by_env[env_id] = ep is not None and not ep.is_empty()
        _orig(env_ids=env_ids, demo_ids=demo_ids)
        for env_id in env_ids:
            _vm.finish_export(env_id=env_id, hdf5_path=hdf5_path, keep=keep_by_env.get(env_id, False))
        _vm.barrier()

    rm.export_episodes = _export
    rm._mana_sync_finalize = True


def install_async_writer(recorder_manager: Any, *, max_queue: int = 8, pinned_memory: bool = True, chunk_size: int = 32) -> Any:
    rm = recorder_manager
    if rm is None or not getattr(rm, "active_terms", None):
        return None
    episodes = getattr(rm, "_episodes", None)
    if not isinstance(episodes, dict):
        return None
    orig_export = getattr(rm, "export_episodes", None)
    if not callable(orig_export):
        return None
    import torch
    from isaaclab.managers.recorder_manager import DatasetExportMode
    writer = AsyncRecorderWriter(max_queue=max_queue)
    orig_close = rm.close

    def _async_export(env_ids=None, demo_ids=None, _orig=orig_export, _w=writer):
        if not getattr(rm, "active_terms", None):
            return _orig(env_ids=env_ids, demo_ids=demo_ids)
        if env_ids is None:
            env_ids = list(range(rm._env.num_envs))
        if isinstance(env_ids, torch.Tensor):
            env_ids = env_ids.tolist()
        if isinstance(demo_ids, torch.Tensor):
            demo_ids = demo_ids.tolist()
        vm = getattr(rm, "_mana_sidecar_video_manager", None)
        snapshots = []
        for i, env_id in enumerate(env_ids):
            ep = episodes.get(env_id)
            if ep is None or ep.is_empty():
                continue
            repl = ManaStagedEpisodeData(pinned=pinned_memory, chunk_size=chunk_size, video_sink=vm)
            repl._env_id = env_id
            episodes[env_id] = repl
            snapshots.append((env_id, ep, demo_ids[i] if demo_ids is not None else None))
        if not snapshots:
            return None
        handler = rm._dataset_file_handler
        export_mode = rm.cfg.dataset_export_mode
        hdf5_path = ""
        stream = getattr(handler, "_hdf5_file_stream", None)
        if stream is not None:
            hdf5_path = getattr(stream, "filename", "") or ""

        def _do_write(snaps=snapshots, h=handler, mode=export_mode, vm=vm, hp=hdf5_path):
            need_flush = False
            for env_id, ep, demo_id in snaps:
                ep.pre_export()
                succeeded = ep.success
                target = h if (mode == DatasetExportMode.EXPORT_ALL or
                               (mode == DatasetExportMode.EXPORT_SUCCEEDED_ONLY and succeeded)) else None
                keep = target is not None
                if target is not None:
                    target.write_episode(ep, demo_id)
                    need_flush = True
                if vm is not None:
                    vm.finish_export(env_id=env_id, hdf5_path=hp, keep=keep)
            if need_flush and h is not None:
                h.flush()

        _w.submit(_do_write)
        return None

    def _close_with_drain(_orig=orig_close, _w=writer):
        vm = getattr(rm, "_mana_sidecar_video_manager", None)
        try:
            _w.barrier()
            if vm is not None:
                vm.close()
        finally:
            try:
                _orig()
            finally:
                _w.shutdown(drain=False, timeout=10.0)

    rm.export_episodes = _async_export
    rm.close = _close_with_drain
    rm._mana_async_writer = writer
    return writer
