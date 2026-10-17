"""Shared dataset export helpers for teleop collection and policy eval."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class RecordingLayout:
    """Resolved paths and handler settings for IsaacLab RecorderManager."""

    env_name: str
    output_dir: str
    prefix: str
    bootstrap_name: str
    exported_paths: list[str] = field(default_factory=list)
    is_direct_lerobot: bool = False
    handler_type: type | None = None
    fmt: str = 'bimanual_lerobot'


def resolve_recording_layout(
    *,
    working_path: str,
    env_name: str,
    task_prompt: str,
    device_tag: str,
    fmt: str = 'bimanual_lerobot',
    dataset_fps: float = 20.0,
) -> RecordingLayout:
    """Resolve output directory and optional LeRobot handler (same as collect)."""
    from maniparena_sim.terms.recorders.dataset_handlers.bimanual_lerobot.helpers import (
        create_bimanual_lerobot_handler_type,
        resolve_bimanual_lerobot_dataset_layout,
    )

    prefix = f'{env_name}_{device_tag}'
    bootstrap_name = f'{prefix}_bootstrap'
    handler_type = None
    exported_paths: list[str] = []
    is_direct_lerobot = False

    if fmt == 'bimanual_lerobot':
        layout = resolve_bimanual_lerobot_dataset_layout(
            save_path=working_path,
            env_name=env_name,
            device_name=device_tag,
        )
        handler_type = create_bimanual_lerobot_handler_type(
            save_path=working_path,
            env_name=env_name,
            task_name=task_prompt or env_name,
            fps=float(dataset_fps),
            device_name=device_tag,
        )
        exported_paths = [layout['joint_root'], layout['ee_root']]
        output_dir = str(Path(layout['base_root']).expanduser())
        is_direct_lerobot = True
    else:
        output_dir = str(Path(os.path.expanduser(working_path)) / env_name)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    return RecordingLayout(
        env_name=env_name,
        output_dir=output_dir,
        prefix=prefix,
        bootstrap_name=bootstrap_name,
        exported_paths=exported_paths,
        is_direct_lerobot=is_direct_lerobot,
        handler_type=handler_type,
        fmt=fmt,
    )


def configure_env_recorder(
    env_cfg: Any,
    layout: RecordingLayout,
    *,
    enable_cameras: bool,
    handler_type: type | None = None,
    export_on_reset: bool = False,
) -> None:
    """Attach RecorderManager cfg before ``gym.make`` (teleop + eval).

    ``export_on_reset`` must be True for policy eval: Isaac Lab clears the
    recorder episode buffer inside ``_reset_idx`` after ``record_pre_reset``,
    so demos that terminate inside ``step()`` are only flushed if export runs
    during ``record_pre_reset``. Teleop keeps False and flushes manually (H).
    """
    from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
    from isaaclab.managers import DatasetExportMode

    from maniparena_sim.terms.recorders.recording_terms import (
        PreStepCameraObservationsRecorderCfg,
        PreStepPolicyObservationsRecorderCfg,
    )

    env_cfg.recorders = ActionStateRecorderManagerCfg()
    env_cfg.recorders.dataset_export_dir_path = layout.output_dir
    env_cfg.recorders.dataset_filename = layout.bootstrap_name
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    env_cfg.recorders.export_in_record_pre_reset = bool(export_on_reset)
    resolved_handler = handler_type if handler_type is not None else layout.handler_type
    if resolved_handler is not None:
        env_cfg.recorders.dataset_file_handler_class_type = resolved_handler
    # Structured policy obs (eef/joint keys) for LeRobot EE export.
    env_cfg.recorders.record_pre_step_policy_observations = (
        PreStepPolicyObservationsRecorderCfg()
    )
    if enable_cameras:
        env_cfg.recorders.record_camera_obs = PreStepCameraObservationsRecorderCfg()


def finalize_bimanual_recorder(
    gym_env: Any,
    layout: RecordingLayout,
    *,
    export_on_reset: bool = False,
) -> None:
    """Post-``gym.make`` recorder cleanup shared by collect and eval."""
    from isaaclab.managers import DatasetExportMode

    rm = gym_env.recorder_manager
    rm.cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    rm.cfg.export_in_record_pre_reset = bool(export_on_reset)
    if not layout.is_direct_lerobot and rm._dataset_file_handler is not None:
        rm._dataset_file_handler.close()
        rm._dataset_file_handler = None
    boot_f = os.path.join(layout.output_dir, f'{layout.bootstrap_name}.hdf5')
    if not layout.is_direct_lerobot and os.path.exists(boot_f):
        os.remove(boot_f)


def finalize_quanta_x1_recorder(
    gym_env: Any,
    layout: RecordingLayout,
    payload: dict,
    *,
    export_on_reset: bool = False,
) -> None:
    """Install QUANTA_X1 streaming recorder stack (same as quanta_x1 collect)."""
    from isaaclab.managers import DatasetExportMode

    from maniparena_sim.terms.recorders.streaming.install import (
        install_async_writer,
        install_staging,
        install_sync_export_sidecar_finalize,
    )

    rm = gym_env.recorder_manager
    rm.cfg.dataset_export_mode = DatasetExportMode.EXPORT_ALL
    rm.cfg.export_in_record_pre_reset = bool(export_on_reset)
    video_fps = float(payload.get('step_hz', 20))
    install_staging(
        rm,
        pinned_memory=True,
        chunk_size=int(payload.get('stager_chunk_size', 32)),
        video_fps=video_fps,
    )
    if bool(payload.get('async_writer', False)):
        install_async_writer(
            rm,
            max_queue=int(payload.get('async_max_queue', 8)),
            pinned_memory=True,
            chunk_size=int(payload.get('stager_chunk_size', 32)),
        )
    else:
        install_sync_export_sidecar_finalize(rm)

    if not layout.is_direct_lerobot:
        # Eval writes one file per episode; replace the gym.make bootstrap handler.
        if rm._dataset_file_handler is not None:
            rm._dataset_file_handler.close()
        rm._dataset_file_handler = PerEpisodeFileHandler(layout, rm)
    boot_f = os.path.join(layout.output_dir, f'{layout.bootstrap_name}.hdf5')
    if os.path.exists(boot_f):
        os.remove(boot_f)


def next_episode_dataset_name(layout: RecordingLayout) -> str:
    """Return the next unused ``{prefix}_episode{N}`` basename."""
    idx = 0
    while True:
        name = f'{layout.prefix}_episode{idx}'
        path = os.path.join(layout.output_dir, f'{name}.hdf5')
        if not os.path.exists(path):
            return name
        idx += 1


class PerEpisodeFileHandler:
    """Recorder dataset handler that writes each exported episode to its own file.

    Isaac Lab exports a natural ``done`` (task success, dropped block) inside ``env.step()``
    via ``record_pre_reset``, where no per-episode file is open yet. A handler that is only
    created around explicit exports silently drops those episodes, so the file is opened on
    demand here and every export path (in-step or explicit) keeps its data.
    """

    def __init__(self, layout: RecordingLayout, rm: Any) -> None:
        self._layout = layout
        self._rm = rm
        self._env_args: dict | None = None
        self._unreported: list[str] = []
        self._num_episodes = 0

    @property
    def _hdf5_file_stream(self) -> Any:
        # The sidecar video manager reads the target HDF5 path from here *before* the write.
        from types import SimpleNamespace

        name = next_episode_dataset_name(self._layout)
        return SimpleNamespace(filename=os.path.join(self._layout.output_dir, f'{name}.hdf5'))

    def add_env_args(self, env_args: dict) -> None:
        self._env_args = env_args

    def get_num_episodes(self) -> int:
        return self._num_episodes

    def write_episode(self, episode: Any, demo_id: int | None = None, dataset_compression: bool = False) -> None:
        layout = self._layout
        name = next_episode_dataset_name(layout)
        fh = self._rm.cfg.dataset_file_handler_class_type()
        fh.create(os.path.join(layout.output_dir, name), env_name=layout.env_name)
        try:
            if self._env_args is not None:
                fh.add_env_args(self._env_args)
            fh.write_episode(episode, demo_id, dataset_compression)
            fh.flush()
        finally:
            fh.close()
        self._unreported.append(os.path.join(layout.output_dir, f'{name}.hdf5'))
        self._num_episodes += 1

    def pop_unreported(self) -> str | None:
        """Path of the newest episode written since the last call (None if nothing new)."""
        path = self._unreported[-1] if self._unreported else None
        self._unreported.clear()
        return path

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


def _episode_has_actions(rm: Any, env_id: int) -> bool:
    ep = rm._episodes.get(int(env_id))
    if ep is None or ep.is_empty():
        return False
    # QUANTA_X1 collection uses ManaStagedEpisodeData. Its action samples remain
    # in the GPU staging queue until pre_export(), so get_action() is empty until
    # the staged buffers are drained.
    pre_export = getattr(ep, "pre_export", None)
    if callable(pre_export):
        pre_export()
    return ep.get_action(0) is not None


def export_episode(
    gym_env: Any,
    layout: RecordingLayout,
    env_id: int,
    mark_success: bool,
) -> str | None:
    """Flush one finished episode to disk (teleop ``H`` key and eval share this).

    When ``export_in_record_pre_reset`` is enabled (policy eval), a natural
    ``done`` already flushes inside ``step()`` before the buffer is cleared.
    In that case this helper is a no-op and returns the dataset root if the
    LeRobot handler already has episodes.
    """
    rm = gym_env.recorder_manager
    env_ids = [int(env_id)]
    handler = getattr(rm, '_dataset_file_handler', None)
    before = (
        int(handler.get_num_episodes())
        if handler is not None and hasattr(handler, 'get_num_episodes')
        else 0
    )

    per_episode = isinstance(handler, PerEpisodeFileHandler)

    # Auto-reset path already exported + cleared the buffer.
    if not _episode_has_actions(rm, env_id):
        if per_episode:
            return handler.pop_unreported()
        if layout.is_direct_lerobot and before > 0:
            return layout.output_dir
        return None

    rm.record_pre_reset(env_ids, force_export_or_skip=False)
    rm.set_success_to_episodes(
        env_ids,
        torch.tensor([mark_success], dtype=torch.bool, device=gym_env.device),
    )

    if per_episode:
        rm.export_episodes(env_ids)
        return handler.pop_unreported()

    if layout.is_direct_lerobot:
        rm.export_episodes(env_ids)
        after = (
            int(handler.get_num_episodes())
            if handler is not None and hasattr(handler, 'get_num_episodes')
            else before
        )
        return layout.output_dir if after > before or before > 0 else None

    ep_name = next_episode_dataset_name(layout)
    rm.cfg.dataset_filename = ep_name
    fh = rm.cfg.dataset_file_handler_class_type()
    fh.create(
        os.path.join(layout.output_dir, ep_name),
        env_name=layout.env_name,
    )
    rm._dataset_file_handler = fh
    rm.export_episodes(env_ids)
    fh.close()
    rm._dataset_file_handler = None
    return os.path.join(layout.output_dir, f'{ep_name}.hdf5')


def recording_enabled(payload: dict) -> bool:
    return bool(payload.get('enable_recording', False))


def dataset_fps_from_scene(payload: dict, scene: Any) -> float:
    sim_fps = getattr(scene, 'sim_fps', None)
    render_decremental = getattr(scene, 'render_decremental', None)
    if sim_fps and render_decremental:
        return float(sim_fps) / float(render_decremental)
    return float(payload.get('step_hz', 20))
