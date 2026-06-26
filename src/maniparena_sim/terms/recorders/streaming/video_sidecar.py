"""Streaming MP4 sidecar writer (RGB only).

Encodes RGB uint8 frame chunks to per-(env,key) temp MP4s on a background
thread (PyAV/libx264); on finish, renames the temp file to
``<hdf5_stem>_videos/<key>/episode_NNNNNN.mp4``.
"""

from __future__ import annotations

import queue
import threading
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np

_STOP = object()


class StreamingVideoWriter:
    """Append RGB frames to one MP4 via PyAV/libx264."""

    def __init__(self, video_path: str | Path, fps: float, *, crf: int = 23, preset: str = "medium") -> None:
        self.video_path = Path(video_path)
        self.fps = float(fps)
        self.crf = int(crf)
        self.preset = str(preset)
        self._container = None
        self._stream = None
        self._frame_shape: tuple[int, int] | None = None

    def append_frame(self, frame: np.ndarray) -> None:
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected (H,W,3), got {frame.shape}")
        h, w = int(frame.shape[0]), int(frame.shape[1])
        if self._container is None:
            self.video_path.parent.mkdir(parents=True, exist_ok=True)
            self._container = av.open(str(self.video_path), mode="w")
            self._stream = self._container.add_stream("libx264", rate=Fraction(str(self.fps)))
            self._stream.width = w
            self._stream.height = h
            self._stream.pix_fmt = "yuv420p"
            self._stream.options = {"crf": str(self.crf), "preset": self.preset}
            self._frame_shape = (h, w)
        elif self._frame_shape != (h, w):
            raise ValueError(f"frame size changed {self._frame_shape}->{(h, w)} for {self.video_path}")
        video_frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(frame), format="rgb24")
        for packet in self._stream.encode(video_frame):
            self._container.mux(packet)

    def close(self) -> None:
        container, stream = self._container, self._stream
        self._container = self._stream = None
        if container is None or stream is None:
            return
        try:
            for packet in stream.encode():
                container.mux(packet)
        finally:
            container.close()


class SidecarVideoSession:
    """Background thread that owns per-(env,key) StreamingVideoWriters."""

    def __init__(self, *, recorder_manager: Any, fps: float = 20.0, max_queue: int = 4) -> None:
        self._rm = recorder_manager
        self._fps = float(fps)
        self._queue: queue.Queue = queue.Queue(maxsize=max(1, int(max_queue)))
        self._writers: dict[Path, StreamingVideoWriter] = {}
        self._paths_by_env: dict[int, dict[str, Path]] = {}
        self._stream_index = 0
        self._episode_index_by_dir: dict[Path, int] = {}
        self._exception: BaseException | None = None
        self._lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(target=self._loop, name="ex001-sidecar-video", daemon=True)
        self._thread.start()

    def _hdf5_path(self) -> Path | None:
        handler = getattr(self._rm, "_dataset_file_handler", None)
        stream = getattr(handler, "_hdf5_file_stream", None)
        filename = getattr(stream, "filename", None)
        if filename:
            return Path(filename).expanduser()
        cfg = getattr(self._rm, "cfg", None)
        d = getattr(cfg, "dataset_export_dir_path", None)
        s = getattr(cfg, "dataset_filename", None)
        if d and s:
            return Path(d).expanduser() / f"{s}.hdf5"
        return None

    def _temp_path(self, hdf5_path: Path, key: str, env_id: int, idx: int) -> Path:
        safe = str(key).replace("/", "_")
        return hdf5_path.parent / f"{hdf5_path.stem}_videos" / f".{safe}_env{env_id}_{idx}.tmp.mp4"

    def _final_path(self, hdf5_path: Path, key: str, episode_index: int) -> Path:
        safe = str(key).replace("/", "_")
        return hdf5_path.parent / f"{hdf5_path.stem}_videos" / safe / f"episode_{episode_index:06d}.mp4"

    def append_chunk(self, *, env_id: int, key: str, frames: Any, ready_event: Any = None) -> None:
        hdf5_path = self._hdf5_path()
        if hdf5_path is None:
            return
        env_id, key = int(env_id), str(key)
        paths = self._paths_by_env.setdefault(env_id, {})
        if key not in paths:
            paths[key] = self._temp_path(hdf5_path, key, env_id, self._stream_index)
            self._stream_index += 1
        self._submit(("append", paths[key], frames, ready_event))

    def reset_env(self, env_id: int) -> None:
        paths = self._paths_by_env.pop(int(env_id), {})
        for p in paths.values():
            self._submit(("close", p, False, None))

    def finish_export(self, *, env_id: int, hdf5_path: str | Path, keep: bool = True) -> None:
        hp = Path(hdf5_path).expanduser()
        paths = self._paths_by_env.pop(int(env_id), {})
        for key, temp in paths.items():
            final = None
            if keep:
                final_dir = self._final_path(hp, key, 0).parent
                idx = self._episode_index_by_dir.get(final_dir, self._scan_next_index(final_dir))
                self._episode_index_by_dir[final_dir] = idx + 1
                final = self._final_path(hp, key, idx)
            self._submit(("close", temp, bool(keep), final))

    @staticmethod
    def _scan_next_index(final_dir: Path) -> int:
        if not final_dir.is_dir():
            return 0
        mx = -1
        for p in final_dir.glob("episode_*.mp4"):
            try:
                mx = max(mx, int(p.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return mx + 1

    def barrier(self) -> None:
        self._queue.join()
        self._raise()

    def close(self) -> None:
        if self._closed:
            self._raise()
            return
        self.barrier()
        self._queue.put(_STOP)
        self._thread.join(timeout=10.0)
        self._closed = True
        self._raise()

    def _submit(self, item) -> None:
        self._raise()
        if self._closed:
            raise RuntimeError("SidecarVideoSession is closed")
        self._queue.put(item)

    def _loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is _STOP:
                    for w in list(self._writers.values()):
                        w.close()
                    self._writers.clear()
                    return
                op = item[0]
                if op == "append":
                    _, path, frames, ready_event = item
                    self._append(path, frames, ready_event)
                elif op == "close":
                    _, path, keep, final = item
                    self._close(path, keep, final)
            except BaseException as exc:  # noqa: BLE001
                with self._lock:
                    if self._exception is None:
                        self._exception = exc
            finally:
                self._queue.task_done()

    def _append(self, path: Path, frames: Any, ready_event: Any) -> None:
        if ready_event is not None:
            ready_event.synchronize()
        arr = frames.detach().cpu().numpy() if hasattr(frames, "detach") else np.asarray(frames)
        if arr.ndim == 3:
            arr = arr[None, ...]
        arr = np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))
        writer = self._writers.get(path)
        if writer is None:
            writer = StreamingVideoWriter(path, self._fps)
            self._writers[path] = writer
        for frame in arr:
            writer.append_frame(frame)

    def _close(self, path: Path, keep: bool, final: Path | None) -> None:
        writer = self._writers.pop(path, None)
        if writer is not None:
            writer.close()
        if keep and final is not None and path.exists():
            final.parent.mkdir(parents=True, exist_ok=True)
            if final.exists():
                final.unlink()
            path.replace(final)
        elif not keep and path.exists():
            path.unlink()

    def _raise(self) -> None:
        with self._lock:
            if self._exception is None:
                return
            exc, self._exception = self._exception, None
        raise exc
