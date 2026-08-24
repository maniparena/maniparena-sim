"""Path resolution and dataset discovery for HDF5 / LeRobot replay."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Literal


def normalize_dataset_path(path: str | Path) -> Path:
    """Expand ``~`` and resolve to an absolute path."""
    return Path(path).expanduser().resolve()


def natural_sort_key(path: str | Path) -> list[Any]:
    """Sort key that orders embedded integers numerically."""
    basename = os.path.basename(str(path))
    return [
        int(c) if c.isdigit() else c.lower()
        for c in re.split(r'(\d+)', basename)
    ]


def find_hdf5_files(input_path: str | Path) -> list[Path]:
    """Find HDF5 files under *input_path*."""
    path = normalize_dataset_path(input_path)
    if path.is_file():
        return [path] if path.suffix.lower() == '.hdf5' else []
    return sorted(path.glob('**/*.hdf5'), key=natural_sort_key)


def resolve_hdf5_episode_file(
    input_path: str | Path,
    episode: int = 0,
) -> Path:
    """Resolve a single HDF5 episode file."""
    files = find_hdf5_files(input_path)
    if not files:
        raise FileNotFoundError(
            f'No HDF5 files found under: {input_path}'
        )
    if episode < 0 or episode >= len(files):
        raise IndexError(
            f'Episode {episode} out of range for '
            f'{len(files)} HDF5 file(s).'
        )
    return files[episode]


def find_lerobot_parquet_files(
    input_path: str | Path,
) -> list[Path]:
    """Find LeRobot parquet episode files."""
    path = normalize_dataset_path(input_path)
    if path.is_file():
        if path.suffix.lower() == '.parquet':
            return [path]
        return []
    data_dir = path / 'data'
    if not data_dir.is_dir():
        return []
    return sorted(
        data_dir.glob('chunk-*/episode_*.parquet'),
        key=natural_sort_key,
    )


def resolve_lerobot_episode_file(
    input_path: str | Path,
    episode: int = 0,
) -> Path:
    """Resolve a single LeRobot parquet episode file."""
    files = find_lerobot_parquet_files(input_path)
    if not files:
        raise FileNotFoundError(
            f'No LeRobot parquet files under: {input_path}'
        )
    for path in files:
        match = re.search(r'episode_(\d+)', path.stem)
        if match is not None and int(match.group(1)) == episode:
            return path
    if episode < 0 or episode >= len(files):
        raise IndexError(
            f'Episode {episode} out of range for '
            f'{len(files)} parquet file(s).'
        )
    return files[episode]


def read_lerobot_episode(
    input_path: str | Path,
    *,
    episode: int = 0,
    representation: Literal['joint', 'ee'],
) -> dict[str, Any]:
    """Load one LeRobot episode parquet."""
    import numpy as np
    import pyarrow.parquet as pq

    path = normalize_dataset_path(input_path)
    parquet_path = resolve_lerobot_episode_file(path, episode)
    if parquet_path.name.endswith('.parquet'):
        dataset_root = parquet_path.parent.parent.parent
    else:
        dataset_root = parquet_path.parent

    info_path = dataset_root / 'meta' / 'info.json'
    info: dict[str, Any] = {}
    if info_path.exists():
        info = json.loads(info_path.read_text())

    table = pq.read_table(
        parquet_path,
        columns=[
            'action', 'observation.state',
            'timestamp', 'episode_index',
        ],
    )
    action = np.asarray(
        table['action'].to_pylist(), dtype=np.float32,
    )
    state = np.asarray(
        table['observation.state'].to_pylist(),
        dtype=np.float32,
    )
    timestamp = np.asarray(
        table['timestamp'].to_pylist(), dtype=np.float32,
    )
    episode_index = (
        int(table['episode_index'][0].as_py())
        if len(table) > 0
        else episode
    )
    return {
        'parquet_path': parquet_path,
        'dataset_root': dataset_root,
        'representation': representation,
        'action': action,
        'state': state,
        'timestamp': timestamp,
        'episode_index': episode_index,
        'info': info,
    }
