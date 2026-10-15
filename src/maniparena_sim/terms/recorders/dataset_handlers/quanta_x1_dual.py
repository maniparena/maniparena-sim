"""Recorder handler that writes one QUANTA_X1 episode to HDF5 and LeRobot."""

from __future__ import annotations

from typing import Any

from maniparena_sim.terms.recorders.streaming.hdf5_handler import PlainHDF5DatasetFileHandler


class QuantaX1DualDatasetFileHandler:
    """Fan out RecorderManager writes to the canonical HDF5 and LeRobot files."""

    lerobot_handler_type: type | None = None

    def __init__(self):
        self._hdf5 = PlainHDF5DatasetFileHandler()
        self._lerobot = None

    @property
    def _hdf5_file_stream(self):
        # Streaming sidecar video discovery uses this attribute on the active
        # dataset handler to associate MP4s with the HDF5 episode.
        return self._hdf5._hdf5_file_stream

    @property
    def _hdf5_data_group(self):
        return self._hdf5._hdf5_data_group

    def create(self, file_path: str, env_name: str = None):
        self._hdf5.create(file_path, env_name=env_name)
        handler_type = getattr(type(self), "lerobot_handler_type", None)
        if handler_type is None:
            raise RuntimeError("QuantaX1DualDatasetFileHandler has no LeRobot handler type")
        self._lerobot = handler_type()
        self._lerobot.create(file_path, env_name=env_name)

    def open(self, file_path: str, mode: str = "r"):
        self._hdf5.open(file_path, mode=mode)
        handler_type = getattr(type(self), "lerobot_handler_type", None)
        if handler_type is None:
            raise RuntimeError("QuantaX1DualDatasetFileHandler has no LeRobot handler type")
        self._lerobot = handler_type()
        self._lerobot.open(file_path, mode=mode)

    def add_env_args(self, env_args: dict):
        self._hdf5.add_env_args(env_args)
        if self._lerobot is not None:
            self._lerobot.add_env_args(env_args)

    def get_env_name(self) -> str | None:
        return self._hdf5.get_env_name()

    def get_num_episodes(self) -> int:
        return self._hdf5.get_num_episodes()

    def write_episode(self, episode: Any, demo_id: int | None = None, dataset_compression: bool = False):
        # Keep both exports in the same RecorderManager transaction.  HDF5 is
        # written first so an auxiliary LeRobot failure never hides the raw log.
        self._hdf5.write_episode(episode, demo_id=demo_id, dataset_compression=dataset_compression)
        if self._lerobot is not None:
            self._lerobot.write_episode(episode, demo_id=demo_id, dataset_compression=dataset_compression)

    def flush(self):
        self._hdf5.flush()
        if self._lerobot is not None:
            self._lerobot.flush()

    def close(self):
        try:
            if self._lerobot is not None:
                self._lerobot.close()
        finally:
            self._hdf5.close()

    def load_episode(self, episode_name: str):
        raise NotImplementedError


def create_quanta_x1_dual_handler_type(*, output_root: str, task_name: str, fps: float) -> type:
    from maniparena_sim.terms.recorders.dataset_handlers.quanta_x1_lerobot import (
        create_quanta_x1_lerobot_handler_type,
    )

    lerobot_type = create_quanta_x1_lerobot_handler_type(
        output_root=output_root, task_name=task_name, fps=fps,
    )
    return type(
        "QuantaX1DualDataset",
        (QuantaX1DualDatasetFileHandler,),
        {"lerobot_handler_type": lerobot_type},
    )
