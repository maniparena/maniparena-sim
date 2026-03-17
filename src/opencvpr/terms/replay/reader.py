"""Replay data reader for HDF5 and LeRobot datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np

from opencvpr.terms.recorders.dataset_handlers.bimanual_lerobot.handler import (  # noqa: E501
    build_ee_action,
    clip_gripper_channels,
    select_active_joint_positions,
)
from opencvpr.terms.recorders.dataset_handlers.utils import nested_get
from opencvpr.terms.replay.io_utils import (
    find_hdf5_files,
    find_lerobot_parquet_files,
    normalize_dataset_path,
    read_lerobot_episode,
    resolve_hdf5_episode_file,
)
from opencvpr.utils.math_utils import to_numpy_keep_shape as to_numpy


def _load_hdf5_episode_data_dict(
    hdf5_path: Union[str, Path],
    episode: int = 0,
) -> dict:
    """Load one episode data dict from HDF5 via IsaacLab."""
    from isaaclab.utils.datasets import HDF5DatasetFileHandler

    path = Path(hdf5_path).expanduser().resolve()
    if path.is_dir():
        file_path = resolve_hdf5_episode_file(path, episode)
        episode_index = 0
    else:
        file_path = path
        episode_index = episode

    dataset = HDF5DatasetFileHandler()
    dataset.open(str(file_path))
    try:
        names = list(dataset.get_episode_names())
        if not names or episode_index >= len(names):
            raise IndexError(
                f'Episode {episode_index} not in '
                f'{file_path} ({len(names)} episodes)'
            )
        ep = dataset.load_episode(
            names[episode_index], device='cpu',
        )
        if hasattr(ep, 'data') and ep.data:
            return dict(ep.data)
        return {}
    finally:
        dataset.close()


def _load_hdf5_joint_sequence(
    hdf5_path: Union[str, Path],
    episode: int = 0,
) -> np.ndarray:
    """Load 14D joint-position sequence from HDF5."""
    data = _load_hdf5_episode_data_dict(hdf5_path, episode)
    joint_states = to_numpy(
        nested_get(
            data, 'states', 'articulation',
            'robot', 'joint_position',
        )
    )
    obs_joint = to_numpy(nested_get(data, 'obs', 'joint_pos'))
    joint_source = (
        joint_states if joint_states is not None else obs_joint
    )
    if joint_source is None:
        raise ValueError(
            'HDF5 episode has no joint_pos or '
            'states.articulation.robot.joint_position'
        )
    joint_14d = select_active_joint_positions(joint_source)
    return clip_gripper_channels(joint_14d).astype(np.float32)


def _load_hdf5_ee_sequence(
    hdf5_path: Union[str, Path],
    episode: int = 0,
) -> np.ndarray:
    """Load 14D end-effector action sequence from HDF5."""
    data = _load_hdf5_episode_data_dict(hdf5_path, episode)
    left_pos = to_numpy(
        nested_get(data, 'obs', 'eef_delta_pos'),
    )
    left_quat = to_numpy(
        nested_get(data, 'obs', 'eef_delta_quat'),
    )
    right_pos = to_numpy(
        nested_get(data, 'obs', 'right_eef_delta_pos'),
    )
    right_quat = to_numpy(
        nested_get(data, 'obs', 'right_eef_delta_quat'),
    )
    joint_states = to_numpy(
        nested_get(
            data, 'states', 'articulation',
            'robot', 'joint_position',
        )
    )
    obs_joint = to_numpy(nested_get(data, 'obs', 'joint_pos'))
    joint_source = (
        joint_states if joint_states is not None else obs_joint
    )
    if joint_source is None or left_pos is None:
        raise ValueError(
            'HDF5 episode missing obs '
            '(eef_delta_pos/quat, joint_pos) or states.'
        )
    if left_quat is None:
        raise ValueError('HDF5 episode missing eef_delta_quat.')
    if right_pos is None or right_quat is None:
        raise ValueError(
            'HDF5 episode missing right_eef_delta_pos/quat.'
        )
    n = min(
        len(joint_source), len(left_pos),
        len(left_quat), len(right_pos), len(right_quat),
    )
    joint_14d = clip_gripper_channels(
        select_active_joint_positions(joint_source[:n])
    )
    return build_ee_action(
        left_pos=left_pos[:n],
        left_quat=left_quat[:n],
        right_pos=right_pos[:n],
        right_quat=right_quat[:n],
        joint_action=joint_14d,
    ).astype(np.float32)


@dataclass
class ReplayData:
    """Normalized replay input."""

    dataset_path: Path
    dataset_format: str
    replay_mode: str
    episode: int = 0
    representation: Optional[str] = None
    dataset_label: str = ''
    replay_description: str = ''
    resolved_hdf5: Optional[Path] = None
    resolved_source_hdf5: Optional[Path] = None
    joint_sequence: Any = None
    ee_sequence: Any = None
    hdf5_dataset: Any = None
    episode_data: Any = None

    def close(self) -> None:
        """Release opened dataset handlers."""
        if self.hdf5_dataset is None:
            return
        try:
            self.hdf5_dataset.close()
        finally:
            self.hdf5_dataset = None
            self.episode_data = None


class ReplayDataReader:
    """Read replay inputs from HDF5 or LeRobot datasets."""

    def detect_dataset_format(
        self,
        dataset_path: str,
        dataset_format: str = 'auto',
    ) -> str:
        """Auto-detect dataset format from path."""
        if dataset_format != 'auto':
            return dataset_format
        path = normalize_dataset_path(dataset_path)
        if (
            path.suffix.lower() == '.hdf5'
            or find_hdf5_files(path)
        ):
            return 'hdf5'
        if (
            path.suffix.lower() == '.parquet'
            or find_lerobot_parquet_files(path)
        ):
            return 'lerobot'
        raise FileNotFoundError(
            f'Unable to detect dataset format for: '
            f'{dataset_path}'
        )

    def _load_hdf5_episode_data(
        self, dataset_path: Path, device: str,
    ):
        """Load first episode from a single HDF5 file."""
        from isaaclab.utils.datasets import (
            HDF5DatasetFileHandler,
        )

        dataset = HDF5DatasetFileHandler()
        dataset.open(str(dataset_path))
        names = list(dataset.get_episode_names())
        if not names:
            dataset.close()
            raise RuntimeError(
                f'No episodes in {dataset_path}'
            )
        episode_data = dataset.load_episode(
            names[0], device=device,
        )
        return dataset, episode_data

    def ensure_episode_loaded(
        self,
        replay_data: ReplayData,
        device: str,
    ) -> ReplayData:
        """Materialize state replay episode data."""
        if replay_data.dataset_format != 'hdf5':
            return replay_data
        if replay_data.replay_mode != 'state':
            return replay_data
        if replay_data.episode_data is not None:
            return replay_data
        (
            replay_data.hdf5_dataset,
            replay_data.episode_data,
        ) = self._load_hdf5_episode_data(
            replay_data.resolved_hdf5,
            device=device,
        )
        return replay_data

    def read(
        self,
        dataset_path: str,
        *,
        replay_mode: str = 'state',
        dataset_format: str = 'auto',
        episode: int = 0,
        device: Optional[str] = None,
    ) -> ReplayData:
        """Read replay data into a normalized bundle."""
        fmt = self.detect_dataset_format(
            dataset_path, dataset_format,
        )
        rd = ReplayData(
            dataset_path=normalize_dataset_path(dataset_path),
            dataset_format=fmt,
            replay_mode=replay_mode,
            episode=episode,
            representation=(
                replay_mode if fmt == 'lerobot' else None
            ),
        )
        if fmt == 'hdf5':
            return self._read_hdf5(rd, device)
        return self._read_lerobot(rd)

    def _read_hdf5(
        self,
        rd: ReplayData,
        device: Optional[str],
    ) -> ReplayData:
        resolved = resolve_hdf5_episode_file(
            rd.dataset_path, episode=rd.episode,
        )
        rd.resolved_hdf5 = resolved
        rd.resolved_source_hdf5 = resolved
        rd.dataset_label = str(resolved)

        if rd.replay_mode == 'state':
            rd.replay_description = 'HDF5 exact state'
            if device is not None:
                self.ensure_episode_loaded(rd, device)
            return rd
        if rd.replay_mode == 'joint':
            rd.joint_sequence = _load_hdf5_joint_sequence(
                resolved, episode=rd.episode,
            )
            rd.replay_description = 'HDF5 joint replay'
            return rd
        if rd.replay_mode == 'ee':
            rd.ee_sequence = _load_hdf5_ee_sequence(
                resolved, episode=rd.episode,
            )
            rd.replay_description = 'HDF5 EE replay'
            return rd
        raise ValueError(
            f"Unsupported replay_mode '{rd.replay_mode}' "
            f'for HDF5.'
        )

    def _read_lerobot(self, rd: ReplayData) -> ReplayData:
        if rd.replay_mode not in ('joint', 'ee'):
            raise ValueError(
                'LeRobot replay requires replay_mode '
                'to be joint or ee.'
            )
        ep = read_lerobot_episode(
            rd.dataset_path,
            episode=rd.episode,
            representation=rd.replay_mode,
        )
        rd.representation = rd.replay_mode
        rd.dataset_label = str(ep['parquet_path'])
        if rd.replay_mode == 'joint':
            rd.joint_sequence = ep['action']
            rd.replay_description = 'LeRobot joint replay'
        else:
            rd.ee_sequence = ep['action']
            rd.replay_description = 'LeRobot ee replay'
        return rd

    def read_initial_state(
        self, replay_data: ReplayData, device: str,
    ):
        """Read the initial scene state for replay."""
        if replay_data.episode_data is not None:
            return replay_data.episode_data.get_initial_state()
        if replay_data.resolved_source_hdf5 is None:
            return None
        dataset, episode_data = self._load_hdf5_episode_data(
            replay_data.resolved_source_hdf5, device,
        )
        try:
            return episode_data.get_initial_state()
        finally:
            dataset.close()
