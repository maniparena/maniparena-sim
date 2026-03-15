"""Direct EX001-6R to LeRobot dataset writer."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
from isaaclab.utils.datasets import DatasetFileHandlerBase, EpisodeData

from opencvpr.terms.recorders.dataset_handlers.ex001_6r_lerobot.constants import (
    ACTIVE_JOINT_INDICES,
    CAMERA_NAME_MAPPING,
    CHUNK_SIZE,
    DATA_PATH_TEMPLATE,
    EE_VECTOR_NAMES,
    GRIPPER_CLIP_MAX,
    GRIPPER_CLIP_MIN,
    JOINT_VECTOR_NAMES,
    LEROBOT_CAMERA_COLUMN_MAPPING,
    ROBOT_TYPE,
    VECTOR_SIZE,
    VIDEO_PATH_TEMPLATE,
)
from opencvpr.terms.recorders.dataset_handlers.utils import (
    feature_stats,
    nested_get,
    read_json,
    read_jsonl,
    require_pyarrow,
    to_numpy,
    write_json,
    write_jsonl,
)
from opencvpr.utils.math_utils import quat_xyzw_to_euler_xyz


def select_active_joint_positions(joint_pos: np.ndarray) -> np.ndarray:
    joint_pos = np.asarray(joint_pos, dtype=np.float32)
    if joint_pos.shape[1] == VECTOR_SIZE:
        return joint_pos.astype(np.float32)
    return joint_pos[:, ACTIVE_JOINT_INDICES].astype(np.float32)


def clip_gripper_channels(action_14d: np.ndarray) -> np.ndarray:
    out = action_14d.astype(np.float32, copy=True)
    out[:, 6] = np.clip(out[:, 6], GRIPPER_CLIP_MIN, GRIPPER_CLIP_MAX)
    out[:, 13] = np.clip(out[:, 13], GRIPPER_CLIP_MIN, GRIPPER_CLIP_MAX)
    return out


def build_lagged_state(action: np.ndarray) -> np.ndarray:
    state = np.empty_like(action)
    state[0] = action[0]
    if len(action) > 1:
        state[1:] = action[:-1]
    return state


def build_ee_action(*, left_pos: np.ndarray, left_quat: np.ndarray, right_pos: np.ndarray, right_quat: np.ndarray, joint_action: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(left_pos, dtype=np.float32),
            quat_xyzw_to_euler_xyz(left_quat, unwrap=True),
            joint_action[:, 6:7],
            np.asarray(right_pos, dtype=np.float32),
            quat_xyzw_to_euler_xyz(right_quat, unwrap=True),
            joint_action[:, 13:14],
        ],
        axis=1,
    ).astype(np.float32)


def build_huggingface_schema_metadata(vector_size: int) -> dict[bytes, bytes]:
    payload = {
        "info": {
            "features": {
                "action": {"feature": {"dtype": "float32", "_type": "Value"}, "length": vector_size, "_type": "Sequence"},
                "observation.state": {"feature": {"dtype": "float32", "_type": "Value"}, "length": vector_size, "_type": "Sequence"},
                "timestamp": {"dtype": "float32", "_type": "Value"},
                "frame_index": {"dtype": "int64", "_type": "Value"},
                "episode_index": {"dtype": "int64", "_type": "Value"},
                "index": {"dtype": "int64", "_type": "Value"},
                "task_index": {"dtype": "int64", "_type": "Value"},
            }
        }
    }
    return {b"huggingface": json.dumps(payload).encode("utf-8")}


def write_lerobot_parquet(parquet_path: str | Path, *, action: np.ndarray, state: np.ndarray, episode_index: int, global_index_start: int, task_index: int, fps: float) -> Path:
    pa, pq = require_pyarrow()
    parquet_path = Path(parquet_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = int(action.shape[0])
    vector_size = int(action.shape[1])
    timestamp = np.arange(num_frames, dtype=np.float32) / float(fps)
    schema = pa.schema(
        [
            ("action", pa.list_(pa.float32(), list_size=vector_size)),
            ("observation.state", pa.list_(pa.float32(), list_size=vector_size)),
            ("timestamp", pa.float32()),
            ("frame_index", pa.int64()),
            ("episode_index", pa.int64()),
            ("index", pa.int64()),
            ("task_index", pa.int64()),
        ],
        metadata=build_huggingface_schema_metadata(vector_size),
    )
    table = pa.Table.from_arrays(
        [
            pa.array(action.tolist(), type=schema.field("action").type),
            pa.array(state.tolist(), type=schema.field("observation.state").type),
            pa.array(timestamp, type=pa.float32()),
            pa.array(np.arange(num_frames, dtype=np.int64), type=pa.int64()),
            pa.array(np.full(num_frames, episode_index, dtype=np.int64), type=pa.int64()),
            pa.array(np.arange(global_index_start, global_index_start + num_frames, dtype=np.int64), type=pa.int64()),
            pa.array(np.full(num_frames, task_index, dtype=np.int64), type=pa.int64()),
        ],
        schema=schema,
    )
    pq.write_table(table, parquet_path)
    return parquet_path


def write_video(frames: np.ndarray, video_path: str | Path, fps: float) -> None:
    frames = np.asarray(frames, dtype=np.uint8)
    if len(frames) == 0:
        return
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = int(frames.shape[1]), int(frames.shape[2])
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    try:
        process.stdin.write(np.ascontiguousarray(frames).tobytes())
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed for {video_path}: {stderr}")
    finally:
        if process.poll() is None:
            process.kill()


def get_video_metadata(video_path: str | Path) -> dict[str, Any] | None:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=height,width,codec_name,pix_fmt,r_frame_rate",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        output = subprocess.check_output(command).decode("utf-8")
        stream = json.loads(output)["streams"][0]
        fps_num, fps_den = map(int, stream["r_frame_rate"].split("/"))
        fps = fps_num / fps_den if fps_den != 0 else 0.0
        return {
            "dtype": "video",
            "shape": [int(stream["height"]), int(stream["width"]), 3],
            "names": ["height", "width", "channels"],
            "video_info": {
                "video.height": int(stream["height"]),
                "video.width": int(stream["width"]),
                "video.fps": float(fps),
                "video.codec": stream["codec_name"],
                "video.pix_fmt": stream["pix_fmt"],
                "video.channels": 3,
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    except Exception:
        return None


def build_dataset_features(*, vector_names: list[str], video_features: dict[str, dict[str, Any]]) -> dict[str, Any]:
    features: dict[str, Any] = dict(video_features)
    features["observation.state"] = {"dtype": "float32", "shape": [VECTOR_SIZE], "names": vector_names}
    features["action"] = {"dtype": "float32", "shape": [VECTOR_SIZE], "names": vector_names}
    features["timestamp"] = {"dtype": "float32", "shape": [1], "names": None}
    features["frame_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["episode_index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["index"] = {"dtype": "int64", "shape": [1], "names": None}
    features["task_index"] = {"dtype": "int64", "shape": [1], "names": None}
    return features


def _build_episode_stats(
    action: np.ndarray,
    state: np.ndarray,
    num_frames: int,
    episode_index: int,
    global_index_start: int,
    fps: float,
) -> dict[str, Any]:
    """Build per-episode statistics matching LeRobot v2.1 format."""
    ts = np.arange(num_frames, dtype=np.float32) / float(fps)
    fi = np.arange(num_frames, dtype=np.int64)
    ei = np.full(num_frames, episode_index, dtype=np.int64)
    gi = np.arange(
        global_index_start,
        global_index_start + num_frames,
        dtype=np.int64,
    )
    ti = np.zeros(num_frames, dtype=np.int64)
    return {
        "episode_index": int(episode_index),
        "stats": {
            "action": feature_stats(action),
            "observation.state": feature_stats(state),
            "timestamp": feature_stats(ts),
            "frame_index": feature_stats(fi),
            "episode_index": feature_stats(ei),
            "index": feature_stats(gi),
            "task_index": feature_stats(ti),
        },
    }


class EX0016RLeRobotDatasetFileHandler(DatasetFileHandlerBase):
    """Recorder backend that writes EX001-6R LeRobot datasets directly."""

    output_layout: dict[str, str] = {}
    task_name: str = ""
    fps: float = 30.0

    def __init__(self):
        self._env_name = None
        self._session_path = None
        self._output_layout = dict(getattr(type(self), "output_layout", {}) or {})
        self._task_name = getattr(type(self), "task_name", "")
        self._fps = float(getattr(type(self), "fps", 30.0))
        self._dataset_roots: dict[str, Path] = {}
        self._episodes: dict[str, list] = {"joint": [], "ee": []}
        self._episode_stats: dict[str, list] = {"joint": [], "ee": []}
        self._video_features: dict[str, dict] = {"joint": {}, "ee": {}}
        self._total_frames: dict[str, int] = {"joint": 0, "ee": 0}
        self._next_episode_index = 0
        self._initialized = False

    def open(self, file_path: str, mode: str = "r"):
        self._session_path = Path(file_path).expanduser()
        self._env_name = self._env_name or self._session_path.stem
        self._initialize_roots()
        self._load_existing_metadata()
        self._initialized = True

    def create(self, file_path: str, env_name: str = None):
        self._session_path = Path(file_path).expanduser()
        self._env_name = env_name or self._session_path.stem
        self._initialize_roots()
        self._load_existing_metadata()
        self._initialized = True

    def add_env_args(self, env_args: dict):
        return None

    def get_env_name(self) -> str | None:
        return self._env_name

    def write_episode(self, episode: EpisodeData, demo_id: int | None = None):
        self._raise_if_not_initialized()
        if episode.is_empty() or episode.get_action(0) is None:
            return
        payload = self._extract_episode_payload(episode)
        if payload is None:
            return
        joint_action, ee_action, camera_frames, num_frames = payload
        episode_index = int(self._next_episode_index if demo_id is None else demo_id)
        self._next_episode_index = max(self._next_episode_index, episode_index + 1)
        chunk = episode_index // CHUNK_SIZE
        for representation, action, names in (("joint", joint_action, JOINT_VECTOR_NAMES), ("ee", ee_action, EE_VECTOR_NAMES)):
            state = build_lagged_state(action)
            global_start = self._total_frames[representation]
            dataset_root = self._dataset_roots[representation]
            parquet_path = dataset_root / DATA_PATH_TEMPLATE.format(episode_chunk=chunk, episode_index=episode_index)
            write_lerobot_parquet(
                parquet_path,
                action=action,
                state=state,
                episode_index=episode_index,
                global_index_start=global_start,
                task_index=0,
                fps=self._fps,
            )
            for camera_key, frames in camera_frames.items():
                video_key = LEROBOT_CAMERA_COLUMN_MAPPING[CAMERA_NAME_MAPPING[camera_key]]
                video_path = dataset_root / VIDEO_PATH_TEMPLATE.format(episode_chunk=chunk, video_key=video_key, episode_index=episode_index)
                write_video(frames=frames, video_path=video_path, fps=self._fps)
                if video_key not in self._video_features[representation]:
                    metadata = get_video_metadata(video_path)
                    if metadata is not None:
                        self._video_features[representation][video_key] = metadata
            self._episodes[representation].append(
                {"episode_index": episode_index, "tasks": [self._task_name or self._env_name or "task"], "length": num_frames}
            )
            self._episodes[representation].sort(key=lambda row: int(row["episode_index"]))
            self._episode_stats[representation].append(
                _build_episode_stats(
                    action, state, num_frames,
                    episode_index, global_start,
                    self._fps,
                )
            )
            self._episode_stats[representation].sort(
                key=lambda row: int(row["episode_index"])
            )
            self._total_frames[representation] += num_frames
            self._write_metadata(representation, names)

    def flush(self):
        if self._initialized:
            self._write_metadata("joint", JOINT_VECTOR_NAMES)
            self._write_metadata("ee", EE_VECTOR_NAMES)

    def close(self):
        self.flush()
        self._initialized = False

    def load_episode(self, episode_name: str) -> EpisodeData | None:
        raise NotImplementedError

    def get_num_episodes(self) -> int:
        return len(self._episodes["joint"])

    def _initialize_roots(self) -> None:
        if not self._output_layout:
            from opencvpr.terms.recorders.dataset_handlers.ex001_6r_lerobot.helpers import resolve_ex001_6r_lerobot_dataset_layout

            self._output_layout = resolve_ex001_6r_lerobot_dataset_layout(save_path=str(self._session_path.parent), env_name=self._env_name or self._session_path.stem)
        self._dataset_roots = {"joint": Path(self._output_layout["joint_root"]), "ee": Path(self._output_layout["ee_root"])}
        for root in self._dataset_roots.values():
            (root / "meta").mkdir(parents=True, exist_ok=True)

    def _load_existing_metadata(self) -> None:
        for representation, root in self._dataset_roots.items():
            meta_dir = root / "meta"
            self._episodes[representation] = read_jsonl(meta_dir / "episodes.jsonl")
            self._episode_stats[representation] = read_jsonl(meta_dir / "episodes_stats.jsonl")
            info = read_json(meta_dir / "info.json")
            features = info.get("features", {})
            self._video_features[representation] = {
                key: value for key, value in features.items() if isinstance(value, dict) and value.get("dtype") == "video"
            }
            self._total_frames[representation] = int(info.get("total_frames", sum(int(row.get("length", 0)) for row in self._episodes[representation])))
            existing_ids = [int(row.get("episode_index", -1)) for row in self._episodes[representation]]
            if existing_ids:
                self._next_episode_index = max(self._next_episode_index, max(existing_ids) + 1)

    def _extract_episode_payload(self, episode: EpisodeData):
        data = episode.data
        joint_states = to_numpy(nested_get(data, "states", "articulation", "robot", "joint_position"))
        obs_joint = to_numpy(nested_get(data, "obs", "joint_pos"))
        left_pos = to_numpy(nested_get(data, "obs", "eef_delta_pos"))
        left_quat = to_numpy(nested_get(data, "obs", "eef_delta_quat"))
        right_pos = to_numpy(nested_get(data, "obs", "right_eef_delta_pos"))
        right_quat = to_numpy(nested_get(data, "obs", "right_eef_delta_quat"))
        joint_source = joint_states if joint_states is not None else obs_joint
        if joint_source is None or left_pos is None or left_quat is None or right_pos is None or right_quat is None:
            raise ValueError("EpisodeData is missing required EX001-6R keys.")
        camera_data = nested_get(data, "camera_obs") or {}
        camera_frames = {}
        num_frames = min(len(joint_source), len(left_pos), len(left_quat), len(right_pos), len(right_quat))
        for camera_key in CAMERA_NAME_MAPPING:
            raw_frames = camera_data.get(camera_key)
            if raw_frames is None:
                continue
            frames = to_numpy(raw_frames)
            if frames is None:
                continue
            camera_frames[camera_key] = np.ascontiguousarray(np.clip(frames[:num_frames], 0, 255).astype(np.uint8))
        joint_action = clip_gripper_channels(select_active_joint_positions(joint_source[:num_frames]))
        ee_action = build_ee_action(
            left_pos=left_pos[:num_frames],
            left_quat=left_quat[:num_frames],
            right_pos=right_pos[:num_frames],
            right_quat=right_quat[:num_frames],
            joint_action=joint_action,
        )
        return joint_action, ee_action, camera_frames, num_frames

    def _write_metadata(self, representation: str, vector_names: list[str]) -> None:
        dataset_root = self._dataset_roots[representation]
        meta_dir = dataset_root / "meta"
        episodes = self._episodes[representation]
        write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": self._task_name or self._env_name or "task"}])
        write_jsonl(meta_dir / "episodes.jsonl", episodes)
        write_jsonl(meta_dir / "episodes_stats.jsonl", self._episode_stats[representation])
        features = build_dataset_features(vector_names=vector_names, video_features=self._video_features[representation])
        payload = {
            "codebase_version": "v2.1",
            "robot_type": ROBOT_TYPE,
            "total_episodes": len(episodes),
            "total_frames": self._total_frames[representation],
            "total_tasks": 1 if episodes else 0,
            "total_videos": len(episodes) * len(self._video_features[representation]),
            "total_chunks": (max((int(row["episode_index"]) for row in episodes), default=-1) // CHUNK_SIZE + 1) if episodes else 0,
            "chunks_size": CHUNK_SIZE,
            "fps": float(self._fps),
            "splits": {"train": f"0:{len(episodes)}"},
            "data_path": DATA_PATH_TEMPLATE,
            "video_path": VIDEO_PATH_TEMPLATE if self._video_features[representation] else None,
            "features": features,
            "stats_preview": feature_stats(np.zeros((1, VECTOR_SIZE), dtype=np.float32)),
        }
        write_json(meta_dir / "info.json", payload)

    def _raise_if_not_initialized(self) -> None:
        if not self._initialized:
            raise RuntimeError("EX0016RLeRobotDatasetFileHandler is not initialized")
