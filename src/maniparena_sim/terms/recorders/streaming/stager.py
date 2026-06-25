"""Chunked GPU staging buffer with dedicated D2H CUDA stream.

This is Slice B of ``recorder_staging.md`` — the asynchronous,
chunked variant of ``ManaStagedEpisodeData``. While Slice A removed
unbounded GPU growth by doing a synchronous ``.to('cpu')`` per
``add()``, every D2H copy still serialized on the simulation main
stream and incurred per-frame PCIe overhead.

Slice B keeps the per-frame add path on the GPU (writing into a
preallocated ring buffer of shape ``(chunk_size, *frame_shape)``),
and only triggers a D2H batch copy when the buffer fills up or
``pre_export`` is called. The batch copy runs on a **dedicated
CUDA stream** so the simulation main stream is not blocked by
PCIe transfers.

Stream isolation decision: one **shared** dedicated D2H stream per
device — avoids both per-instance stream proliferation (which would
contend with other GPU workloads on the same hardware queues) and the
serialization cost of reusing the simulation stream. Those workloads
already run on their own streams; our D2H stream only contends
with them on raw memcpy bandwidth, not on the compute scheduler.

This module is intentionally device-aware but *not* thread-aware.
Slice C (``AsyncRecorderWriter``) will add the writer-thread layer
on top; Slice B alone is enough to flatten the GPU staging cost.

Constraints:

* No upstream modifications. ``ChunkedStager`` is held inside
  ``ManaStagedEpisodeData`` instances; the upstream
  ``RecorderManager`` sees the same ``EpisodeData`` duck-type.
* HDF5 schema unchanged. After ``drain()`` the per-key buffer is
  presented as ``list[CPU tensor]`` matching upstream ``add`` /
  ``pre_export`` semantics so downstream consumers (mimic / replay)
  cannot tell whether chunking was on or off.
"""

from __future__ import annotations

import threading
from typing import Any

import torch

# Module-level shared D2H stream cache keyed by ``device.index`` for CUDA
# devices. CPU-only tests never hit this path. We deliberately share
# across all ``ChunkedStager`` instances on the same device — see module
# docstring for the rationale.
_D2H_STREAM_CACHE: dict[int, torch.cuda.Stream] = {}
_D2H_STREAM_LOCK = threading.Lock()


class ChunkedStager:
    """Per-key GPU ring buffer + batched D2H on a dedicated CUDA stream.

    Buffer layout per ``key``:

        ``_gpu_buf[key]``  : CUDA tensor of shape ``(chunk_size, *frame_shape)``
        ``_pos[key]``      : int — next write index in the GPU ring
        ``_cpu_chunks[key]``: list of pinned CPU tensors of shape
            ``(chunk_n, *frame_shape)`` already copied D2H
        ``_inflight[key]`` : list of ``torch.cuda.Event`` recorded on the
            D2H stream after each chunk copy; ``drain`` synchronises on
            them before returning to the caller.

    Thread safety: a single ``ChunkedStager`` is owned by one
    ``ManaStagedEpisodeData`` instance, which itself is only touched
    from the simulation thread. The shared D2H stream is the only
    cross-instance state and is initialised under a lock.
    """

    def __init__(self, *, chunk_size: int, pinned_memory: bool = True, video_sink: Any = None) -> None:
        if chunk_size <= 0:
            raise ValueError(f"ChunkedStager.chunk_size must be > 0 (got {chunk_size})")
        self._chunk_size = int(chunk_size)
        self._pinned = bool(pinned_memory)
        self._video_sink = video_sink
        self._gpu_buf: dict[str, torch.Tensor] = {}
        self._pos: dict[str, int] = {}
        self._cpu_chunks: dict[str, list[torch.Tensor]] = {}
        self._inflight: dict[str, list[Any]] = {}
        self._video_keys: set[str] = set()
        self._video_env_ids: dict[str, int] = {}

    @staticmethod
    def _d2h_stream(device: torch.device) -> torch.cuda.Stream | None:
        """Return the shared dedicated D2H stream for ``device``."""
        if device.type != "cuda":
            return None
        idx = device.index if device.index is not None else torch.cuda.current_device()
        with _D2H_STREAM_LOCK:
            stream = _D2H_STREAM_CACHE.get(idx)
            if stream is None:
                stream = torch.cuda.Stream(device=idx)
                _D2H_STREAM_CACHE[idx] = stream
            return stream

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    def add(self, key: str, value: torch.Tensor) -> None:
        """Append ``value`` to the chunk buffer for ``key``.

        ``value`` must be a CUDA tensor — the caller (``ManaStagedEpisodeData``)
        routes CPU tensors and non-tensors through the synchronous path.
        """
        if not isinstance(value, torch.Tensor):  # defensive — caller guards
            raise TypeError(f"ChunkedStager.add expects torch.Tensor, got {type(value)!r}")

        buf = self._gpu_buf.get(key)
        if buf is None:
            buf = torch.empty(
                (self._chunk_size, *value.shape),
                dtype=value.dtype,
                device=value.device,
            )
            self._gpu_buf[key] = buf
            self._pos[key] = 0

        pos = self._pos[key]
        buf[pos].copy_(value)
        self._pos[key] = pos + 1

        if self._pos[key] == self._chunk_size:
            self._flush(key, n=self._chunk_size)

    def add_video(self, key: str, value: torch.Tensor, *, env_id: int) -> None:
        """Append one RGB frame to a video-backed chunk buffer."""
        if self._video_sink is None:
            raise RuntimeError("ChunkedStager.add_video requires a video_sink")
        self._video_keys.add(key)
        self._video_env_ids[key] = int(env_id)
        self.add(key, value)

    def has_pending(self, key: str) -> bool:
        """True if any data (in-flight events, CPU chunks, or partial GPU)
        exists for ``key``."""
        return bool(self._cpu_chunks.get(key)) or self._pos.get(key, 0) > 0 or bool(self._inflight.get(key))

    def count(self, key: str) -> int:
        """Return the number of buffered frames for ``key``."""
        total = int(self._pos.get(key, 0))
        for chunk in self._cpu_chunks.get(key, []):
            total += int(chunk.shape[0])
        return total

    def truncate(self, key: str, num_steps: int) -> None:
        """Drop buffered frames beyond ``num_steps`` for ``key``.

        Used by batched state replay to trim padded tail frames from envs
        whose source episode ended before the batch maximum.  Video keys
        are ignored — sidecar MP4 streams are sealed separately via
        ``drain_video`` / ``seal_env``.
        """
        if num_steps < 0:
            raise ValueError(f"ChunkedStager.truncate num_steps must be >= 0 (got {num_steps})")
        if key in self._video_keys:
            return
        if self.count(key) <= num_steps:
            return

        for event in self._inflight.get(key, []):
            event.synchronize()
        self._inflight[key] = []

        partial = self._pos.get(key, 0)
        if partial > 0:
            self._flush(key, n=partial)
            for event in self._inflight.get(key, []):
                event.synchronize()
            self._inflight[key] = []

        chunks = self._cpu_chunks.get(key, [])
        if not chunks:
            return

        stacked = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
        self._cpu_chunks[key] = [stacked[:num_steps].contiguous()]
        self._pos[key] = 0
        self._gpu_buf.pop(key, None)

    def keys(self) -> list[str]:
        return sorted(set(self._gpu_buf.keys()) | set(self._cpu_chunks.keys()))

    def drain(self, key: str) -> torch.Tensor | None:
        """Materialize all pending frames for ``key`` as a single stacked CPU tensor.

        Any partial GPU chunk is flushed first; all in-flight D2H events
        are awaited. The internal buffers for ``key`` are reset to empty
        so subsequent ``add`` calls start fresh.

        Returns a tensor of shape ``(N, *frame_shape)`` or ``None`` if
        no frames were buffered. Returning a pre-concatenated tensor
        avoids the intermediate ``list[per-frame tensor]`` that upstream's
        ``pre_export`` would ``torch.stack`` again, eliminating a
        transient 2x memory spike.
        """
        partial = self._pos.get(key, 0)
        if partial > 0:
            self._flush(key, n=partial)

        for event in self._inflight.get(key, []):
            event.synchronize()
        self._inflight[key] = []

        chunks = self._cpu_chunks.pop(key, [])
        self._gpu_buf.pop(key, None)
        self._pos[key] = 0

        if not chunks:
            return None
        if len(chunks) == 1:
            return chunks[0]
        result = torch.cat(chunks, dim=0)
        del chunks
        return result

    def drain_video(self, key: str) -> None:
        """Flush pending video frames without materialising them into ``_data``."""
        partial = self._pos.get(key, 0)
        if partial > 0:
            self._flush(key, n=partial)
        self._gpu_buf.pop(key, None)
        self._pos[key] = 0
        self._inflight[key] = []
        self._cpu_chunks.pop(key, None)

    def reset(self) -> None:
        """Drop all buffered state without waiting for in-flight events.

        Used on episode discard / env reset. In-flight D2H events will
        complete on their own and the underlying CUDA buffer is freed
        when Python releases the references.
        """
        self._gpu_buf.clear()
        self._pos.clear()
        self._cpu_chunks.clear()
        self._inflight.clear()
        self._video_keys.clear()
        self._video_env_ids.clear()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _flush(self, key: str, *, n: int) -> None:
        """Copy the first ``n`` entries of the GPU ring for ``key`` to CPU."""
        if n <= 0:
            return
        buf = self._gpu_buf[key]
        view = buf[:n]
        device = view.device

        stream = self._d2h_stream(device)
        is_video = key in self._video_keys
        if stream is None:
            host = view.to("cpu", copy=True).contiguous()
            if is_video:
                self._submit_video_chunk(key, host, ready_event=None)
            else:
                self._cpu_chunks.setdefault(key, []).append(host)
            self._pos[key] = 0
            return

        try:
            host = torch.empty((n, *view.shape[1:]), dtype=view.dtype, pin_memory=True)
        except RuntimeError:
            host = torch.empty((n, *view.shape[1:]), dtype=view.dtype)

        # Order: D2H stream waits until simulation main stream finishes
        # the in-place writes that just landed in ``view`` (the per-step
        # ``buf[pos].copy_(value)`` calls above). Then issue the copy
        # asynchronously and record an event the drain path will sync on.
        sim_stream = torch.cuda.current_stream(device)
        stream.wait_stream(sim_stream)
        with torch.cuda.stream(stream):
            host.copy_(view, non_blocking=self._pinned)
        event = torch.cuda.Event()
        event.record(stream)

        if is_video:
            self._submit_video_chunk(key, host, ready_event=event)
        else:
            self._cpu_chunks.setdefault(key, []).append(host)
            self._inflight.setdefault(key, []).append(event)
        self._pos[key] = 0

    def _submit_video_chunk(self, key: str, host: torch.Tensor, *, ready_event: Any = None) -> None:
        if self._video_sink is None:
            return
        env_id = self._video_env_ids.get(key, 0)
        self._video_sink.append_chunk(
            env_id=env_id,
            key=key,
            frames=host,
            ready_event=ready_event,
        )
