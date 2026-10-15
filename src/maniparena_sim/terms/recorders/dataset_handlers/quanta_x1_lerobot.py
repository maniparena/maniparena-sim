"""LeRobot writer for QUANTA_X1 teleoperation episodes.

QUANTA_X1 teleoperation uses a 19D AbsIK action.  This writer keeps that
wire action intact instead of reusing the desktop bimanual 14D schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.utils.datasets import DatasetFileHandlerBase, EpisodeData

from maniparena_sim.terms.recorders.dataset_handlers.bimanual_lerobot.constants import (
    CHUNK_SIZE,
    DATA_PATH_TEMPLATE,
    LEROBOT_CAMERA_COLUMN_MAPPING,
    CAMERA_NAME_MAPPING,
    VIDEO_PATH_TEMPLATE,
)
from maniparena_sim.terms.recorders.dataset_handlers.bimanual_lerobot.handler import (
    get_video_metadata,
    write_video,
)
from maniparena_sim.terms.recorders.dataset_handlers.utils import (
    feature_stats,
    nested_get,
    read_json,
    read_jsonl,
    require_pyarrow,
    to_numpy,
    write_json,
    write_jsonl,
)


QUANTA_ACTION_NAMES = [
    "left_pos_x", "left_pos_y", "left_pos_z", "left_quat_x", "left_quat_y",
    "left_quat_z", "left_quat_w", "left_gripper", "right_pos_x", "right_pos_y",
    "right_pos_z", "right_quat_x", "right_quat_y", "right_quat_z", "right_quat_w",
    "right_gripper", "left_wheel", "right_wheel", "lift",
]
QUANTA_STATE_NAMES = [f"joint_{index}" for index in range(23)]


def _write_parquet(path: Path, action: np.ndarray, state: np.ndarray, episode_index: int, global_start: int, fps: float) -> None:
    pa, pq = require_pyarrow()
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(action.shape[0])
    action_dim = int(action.shape[1])
    state_dim = int(state.shape[1])
    schema = pa.schema([
        ("action", pa.list_(pa.float32(), list_size=action_dim)),
        ("observation.state", pa.list_(pa.float32(), list_size=state_dim)),
        ("timestamp", pa.float32()), ("frame_index", pa.int64()),
        ("episode_index", pa.int64()), ("index", pa.int64()), ("task_index", pa.int64()),
    ], metadata={b"huggingface": json.dumps({"info": {"features": {}}}).encode()})
    table = pa.Table.from_arrays([
        pa.array(action.tolist(), type=schema.field("action").type),
        pa.array(state.tolist(), type=schema.field("observation.state").type),
        pa.array(np.arange(n, dtype=np.float32) / float(fps), type=pa.float32()),
        pa.array(np.arange(n, dtype=np.int64), type=pa.int64()),
        pa.array(np.full(n, episode_index, dtype=np.int64), type=pa.int64()),
        pa.array(np.arange(global_start, global_start + n, dtype=np.int64), type=pa.int64()),
        pa.array(np.zeros(n, dtype=np.int64), type=pa.int64()),
    ], schema=schema)
    pq.write_table(table, path)


class QuantaX1LeRobotDatasetFileHandler(DatasetFileHandlerBase):
    """Write QUANTA_X1 19D teleop actions as a LeRobot dataset."""

    output_root: str = ""
    task_name: str = ""
    fps: float = 20.0

    def __init__(self):
        self._env_name = None
        self._session_path = None
        self._root = Path(getattr(type(self), "output_root", ""))
        self._episodes: list[dict[str, Any]] = []
        self._stats: list[dict[str, Any]] = []
        self._total_frames = 0
        self._next_episode_index = 0
        self._action_dim = 19
        self._state_dim = 23
        self._initialized = False

    def create(self, file_path: str, env_name: str = None):
        self._session_path = Path(file_path).expanduser()
        self._env_name = env_name or self._session_path.stem
        if not self._root:
            self._root = self._session_path.parent / f"{self._session_path.stem}_lerobot"
        (self._root / "meta").mkdir(parents=True, exist_ok=True)
        self._load_existing()
        self._initialized = True

    def open(self, file_path: str, mode: str = "r"):
        self.create(file_path, env_name=self._env_name)

    def add_env_args(self, env_args: dict):
        return None

    def get_env_name(self) -> str | None:
        return self._env_name

    def get_num_episodes(self) -> int:
        return len(self._episodes)

    def load_episode(self, episode_name: str) -> EpisodeData | None:
        raise NotImplementedError

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None, dataset_compression: bool = False):
        del dataset_compression
        if not self._initialized:
            raise RuntimeError("QuantaX1LeRobotDatasetFileHandler is not initialized")
        data = episode.data
        actions = to_numpy(data.get("actions"))
        state = to_numpy(nested_get(data, "obs", "joint_pos"))
        if actions is None:
            actions = to_numpy(data.get("processed_actions"))
        if state is None:
            state = to_numpy(nested_get(data, "states", "articulation", "robot", "joint_position"))
        if actions is None or state is None:
            print("[record] skip QUANTA_X1 LeRobot export: actions or joint state missing")
            return
        actions = np.asarray(actions, dtype=np.float32)
        state = np.asarray(state, dtype=np.float32)
        if actions.ndim != 2 or state.ndim != 2 or actions.shape[1] not in (19, 21):
            print(f"[record] skip QUANTA_X1 LeRobot export: unexpected shapes action={actions.shape} state={state.shape}")
            return
        n = min(len(actions), len(state))
        actions, state = actions[:n], state[:n]
        self._action_dim = int(actions.shape[1])
        self._state_dim = int(state.shape[1])
        index = int(self._next_episode_index if demo_id is None else demo_id)
        self._next_episode_index = max(self._next_episode_index, index + 1)
        chunk = index // CHUNK_SIZE
        _write_parquet(self._root / DATA_PATH_TEMPLATE.format(episode_chunk=chunk, episode_index=index), actions, state, index, self._total_frames, float(self.fps))
        row = {"episode_index": index, "tasks": [getattr(type(self), "task_name", "") or self._env_name or "task"], "length": n}
        self._episodes = [item for item in self._episodes if int(item.get("episode_index", -1)) != index] + [row]
        self._episodes.sort(key=lambda item: int(item["episode_index"]))
        ts = np.arange(n, dtype=np.float32) / float(self.fps)
        fi = np.arange(n, dtype=np.int64)
        gi = np.arange(self._total_frames, self._total_frames + n, dtype=np.int64)
        self._stats = [item for item in self._stats if int(item.get("episode_index", -1)) != index] + [{
            "episode_index": index,
            "stats": {"action": feature_stats(actions), "observation.state": feature_stats(state),
                       "timestamp": feature_stats(ts), "frame_index": feature_stats(fi),
                       "episode_index": feature_stats(np.full(n, index, dtype=np.int64)),
                       "index": feature_stats(gi), "task_index": feature_stats(np.zeros(n, dtype=np.int64))},
        }]
        self._stats.sort(key=lambda item: int(item["episode_index"]))
        self._total_frames += n
        self._write_metadata(actions.shape[1], state.shape[1])
        print(f"[record] QUANTA_X1 LeRobot episode {index} frames={n} action_dim={actions.shape[1]}")

    def flush(self):
        if self._initialized:
            self._write_metadata(self._action_dim, self._state_dim)

    def close(self):
        self.flush()
        self._initialized = False

    def _load_existing(self):
        meta = self._root / "meta"
        self._episodes = read_jsonl(meta / "episodes.jsonl")
        self._stats = read_jsonl(meta / "episodes_stats.jsonl")
        info = read_json(meta / "info.json")
        self._total_frames = int(info.get("total_frames", sum(int(x.get("length", 0)) for x in self._episodes)))
        features = info.get("features", {})
        self._action_dim = int((features.get("action", {}).get("shape") or [19])[0])
        self._state_dim = int((features.get("observation.state", {}).get("shape") or [23])[0])
        ids = [int(x.get("episode_index", -1)) for x in self._episodes]
        self._next_episode_index = max(ids, default=-1) + 1

    def _write_metadata(self, action_dim: int, state_dim: int):
        meta = self._root / "meta"
        write_jsonl(meta / "tasks.jsonl", [{"task_index": 0, "task": getattr(type(self), "task_name", "") or self._env_name or "task"}])
        write_jsonl(meta / "episodes.jsonl", self._episodes)
        write_jsonl(meta / "episodes_stats.jsonl", self._stats)
        action_names = QUANTA_ACTION_NAMES[:action_dim] + (["head_yaw", "head_pitch"] if action_dim == 21 else [])
        state_names = QUANTA_STATE_NAMES[:state_dim]
        write_json(meta / "info.json", {
            "codebase_version": "v2.1", "robot_type": "quanta_x1", "total_episodes": len(self._episodes),
            "total_frames": self._total_frames, "total_tasks": 1 if self._episodes else 0,
            "total_videos": 0, "total_chunks": (max((int(x["episode_index"]) for x in self._episodes), default=-1) // CHUNK_SIZE + 1) if self._episodes else 0,
            "chunks_size": CHUNK_SIZE, "fps": float(getattr(type(self), "fps", 20.0)),
            "splits": {"train": f"0:{len(self._episodes)}"}, "data_path": DATA_PATH_TEMPLATE, "video_path": None,
            "features": {"action": {"dtype": "float32", "shape": [action_dim], "names": action_names},
                         "observation.state": {"dtype": "float32", "shape": [state_dim], "names": state_names},
                         "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                         "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                         "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                         "index": {"dtype": "int64", "shape": [1], "names": None},
                         "task_index": {"dtype": "int64", "shape": [1], "names": None}},
        })


def create_quanta_x1_lerobot_handler_type(*, output_root: str, task_name: str, fps: float) -> type:
    return type("QuantaX1LeRobotDataset", (QuantaX1LeRobotDatasetFileHandler,), {
        "output_root": str(Path(output_root).expanduser()), "task_name": task_name, "fps": float(fps),
    })
