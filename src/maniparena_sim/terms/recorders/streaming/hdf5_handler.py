"""HDF5 helpers tuned for the maniparena-sim recording pipeline.

The default ``HDF5DatasetFileHandler.write_episode`` shipped with IsaacLab
creates each leaf dataset with synchronous gzip compression. In the recording
pipeline this triggers two pathological behaviours:

1. The Kit main thread can stall for seconds per episode while gzip runs.
2. Failures mid-write leave a corrupted, partially-flushed HDF5 file.

The handler defined here disables compression entirely and forces an
``flush()`` after every episode write.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import h5py


@runtime_checkable
class EpisodeLike(Protocol):
    """Structural interface consumed by ``PlainHDF5DatasetFileHandler.write_episode``.

    Any object that satisfies these four attributes is a valid episode,
    without requiring an inheritance relationship to IsaacLab types.
    """

    def is_empty(self) -> bool: ...  # noqa: E704

    @property
    def data(self) -> dict[str, Any]: ...  # noqa: E704

    @property
    def seed(self) -> Any: ...  # noqa: E704

    @property
    def success(self) -> Any: ...  # noqa: E704


class PlainHDF5DatasetFileHandler:
    """Write IsaacLab-compatible episode data directly to HDF5 without compression.

    This class intentionally implements the narrow dataset-handler surface used
    by IsaacLab's ``RecorderManager`` instead of importing
    ``isaaclab.utils.datasets`` at module import time. The package-level
    IsaacLab import pulls in ``pxr`` through unrelated mesh utilities, which
    makes non-Isaac stability tests fail before the handler logic is reached.
    """

    @staticmethod
    def write_action_metadata_attrs(group, metadata) -> None:
        """No-op shim: recordings carry no action-metadata sidecar."""
        return None

    def __init__(self) -> None:
        self._hdf5_file_stream = None
        self._hdf5_data_group = None
        self._demo_count = 0
        self._env_args: dict[str, Any] = {}

    def open(self, file_path: str, mode: str = "r") -> None:
        """Open an existing HDF5 dataset file."""
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        self._hdf5_file_stream = h5py.File(file_path, mode)
        self._hdf5_data_group = self._hdf5_file_stream["data"]
        self._demo_count = len(self._hdf5_data_group)

    def create(
        self,
        file_path: str,
        env_name: str = None,
    ) -> None:
        """Create HDF5 and fail loudly on existing files."""
        path = Path(file_path).expanduser()
        if path.suffix != ".hdf5":
            path = path.with_suffix(".hdf5")
        if path.exists():
            raise FileExistsError(f"HDF5 dataset file already exists: {path}")
        if self._hdf5_file_stream is not None:
            raise RuntimeError("HDF5 dataset file stream is already in use")
        path.parent.mkdir(parents=True, exist_ok=True)
        self._hdf5_file_stream = h5py.File(os.fspath(path), "w")
        self._hdf5_data_group = self._hdf5_file_stream.create_group("data")
        self._hdf5_data_group.attrs["total"] = 0
        self._demo_count = 0
        self.add_env_args({"env_name": env_name or "", "type": 2})

    def add_env_args(self, env_args: dict) -> None:
        """Add robomimic-compatible environment metadata."""
        self._raise_if_not_initialized()
        self._env_args.update(env_args)
        self._hdf5_data_group.attrs["env_args"] = json.dumps(self._env_args)

    def set_env_name(self, env_name: str) -> None:
        """Set the environment name stored in ``env_args``."""
        self.add_env_args({"env_name": env_name})

    def get_env_name(self) -> str | None:
        """Return the environment name stored in ``env_args``."""
        self._raise_if_not_initialized()
        env_args = json.loads(self._hdf5_data_group.attrs["env_args"])
        if "env_name" in env_args:
            return env_args["env_name"]
        return None

    def get_episode_names(self):
        """Return HDF5 episode group names."""
        self._raise_if_not_initialized()
        return self._hdf5_data_group.keys()

    def get_num_episodes(self) -> int:
        """Return number of episodes written or loaded."""
        return self._demo_count

    @property
    def demo_count(self) -> int:
        """Return number of demos collected so far."""
        return self._demo_count

    def _action_metadata(self) -> dict:
        metadata = getattr(self, "_mana_action_metadata", None)
        return metadata if isinstance(metadata, dict) else {}

    def write_episode(
        self,
        episode: EpisodeLike,
        demo_id: int | None = None,
    ) -> None:
        """Add one episode to the dataset without gzip compression."""
        self._raise_if_not_initialized()
        if episode.is_empty():
            return

        episode_group_name = self._episode_group_name(demo_id)
        if episode_group_name in self._hdf5_data_group:
            raise ValueError(
                f"Episode group '{episode_group_name}' already exists in the dataset",
            )
        h5_episode_group = self._hdf5_data_group.create_group(
            episode_group_name,
        )
        self._write_metadata(h5_episode_group, episode)
        self._write_episode_data(h5_episode_group, episode.data)

        self._hdf5_data_group.attrs["total"] += h5_episode_group.attrs["num_samples"]
        if demo_id is None:
            self._demo_count += 1

        self._hdf5_file_stream.flush()

    def flush(self) -> None:
        """Flush pending HDF5 writes to disk."""
        self._raise_if_not_initialized()
        self._hdf5_file_stream.flush()

    def close(self) -> None:
        """Close the dataset file stream."""
        if self._hdf5_file_stream is not None:
            self._hdf5_file_stream.close()
            self._hdf5_file_stream = None
            self._hdf5_data_group = None

    def _raise_if_not_initialized(self) -> None:
        if self._hdf5_file_stream is None:
            raise RuntimeError("HDF5 dataset file stream is not initialized")

    def _episode_group_name(self, demo_id: int | None) -> str:
        if demo_id is not None:
            return f"demo_{demo_id}"
        return f"demo_{self._demo_count}"

    def _write_metadata(self, h5_episode_group: Any, episode: EpisodeLike) -> None:
        action_metadata = self._action_metadata()
        self.write_action_metadata_attrs(self._hdf5_data_group, action_metadata)
        self.write_action_metadata_attrs(h5_episode_group, action_metadata)

        if "actions" in episode.data:
            h5_episode_group.attrs["num_samples"] = len(episode.data["actions"])
        else:
            h5_episode_group.attrs["num_samples"] = 0
        if episode.seed is not None:
            h5_episode_group.attrs["seed"] = episode.seed
        if episode.success is not None:
            h5_episode_group.attrs["success"] = episode.success
        language_instruction = getattr(episode, "language_instruction", None)
        if isinstance(language_instruction, str) and language_instruction.strip():
            h5_episode_group.attrs["language_instruction"] = language_instruction
        phase_segments = getattr(episode, "phase_segments", None)
        if isinstance(phase_segments, str) and phase_segments.strip():
            h5_episode_group.attrs["phase_segments"] = phase_segments

    def _write_episode_data(
        self,
        h5_episode_group: Any,
        episode_data: dict[str, Any],
    ) -> None:
        for key, value in episode_data.items():
            # Per-frame planner phase codes are collapsed into the
            # ``phase_segments`` attr at export; never write the raw leaf.
            if key == "planner_phase":
                continue
            self._write_dataset_value(h5_episode_group, key, value)

    def _write_dataset_value(self, group: Any, key: str, value: Any) -> None:
        if isinstance(value, dict):
            key_group = group.create_group(key)
            for sub_key, sub_value in value.items():
                self._write_dataset_value(key_group, sub_key, sub_value)
            return
        group.create_dataset(key, data=self._to_numpy(value))

    @staticmethod
    def _to_numpy(value: Any) -> Any:
        if hasattr(value, "cpu"):
            return value.cpu().numpy()
        return value
