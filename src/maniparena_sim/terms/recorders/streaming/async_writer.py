"""Background HDF5 writer thread for ``RecorderManager.export_episodes``.

This is Slice C of ``recorder_staging.md`` — the asynchronous writer
that finally moves the per-episode ``pre_export() + write_episode() +
flush()`` work off the simulation main thread.

Slice A removed unbounded GPU growth; Slice B made the per-step D2H
copies batched and asynchronous on a dedicated stream; both are
sim-thread-synchronous at episode boundaries because upstream's
``RecorderManager.export_episodes`` still inlines the stack + HDF5
write loop. For long-episode + multi-camera workloads the HDF5 write
alone can dominate per-episode wallclock and stall the sim.

This module provides a single-writer-thread queue. The companion
``install_async_writer`` in ``recorder_install.py`` wraps
``rm.export_episodes`` so it:

1. Snapshots every ``EpisodeData`` instance for the requested env_ids.
2. Immediately swaps in fresh ``ManaStagedEpisodeData`` instances at
   ``rm._episodes[env_id]`` so the simulation thread can resume
   ``add_to_episodes(...)`` without contending with the writer.
3. Submits a single closure to this queue that runs ``pre_export()``
   (drains the snapshot's ChunkedStager + ``torch.stack``) and
   ``handler.write_episode(...)`` + ``handler.flush()`` on the
   writer thread.

Why a single worker (not a pool)?

* Upstream HDF5 dataset handlers (``HDF5DatasetFileHandler``) wrap
  one ``h5py.File`` instance whose object handles are not safe to
  use from multiple Python threads simultaneously. Routing every
  write through one worker keeps the file-state model trivial.
* HDF5 write throughput is bounded by disk + h5py serialization
  overhead, not Python compute, so adding workers would not help
  for the single-file case anyway.
* Multi-file fanout (e.g. per-episode files via
  ``episode_handler_session``) is supported because the worker
  reads ``rm._dataset_file_handler`` at submit time; each call
  carries its own handler reference into the closure.

Invariants the caller must uphold:

* Before flushing or closing a handler held by ``rm`` (e.g. in
  ``recorder_file_session.close_episode_handler`` or
  ``RecorderManager.close``), call ``writer.barrier()`` so any
  queued write that targets the same handler completes first.
  Without that barrier the worker could ``handler.write_episode``
  on an already-closed handle.
* Worker exceptions are captured and re-raised on the next
  ``submit`` / ``barrier`` call (fail-fast semantics — the
  simulation cannot pretend a write succeeded when HDF5 IO failed).
"""

from __future__ import annotations

import queue
import threading
from typing import Any, Callable

# Sentinel pushed onto the queue in shutdown(); the worker exits as
# soon as it pops a value identical to this object.
_POISON_PILL = object()


class AsyncRecorderWriter:
    """Single-writer-thread queue for HDF5 write offloading.

    Public API:

    * ``submit(fn, *args, **kwargs)`` — enqueue ``fn`` for the worker
      thread. Returns immediately; the work runs in FIFO order on
      the writer thread. Raises any captured worker exception
      synchronously to fail-fast on the simulation side.
    * ``barrier(timeout=None)`` — block the calling thread until
      every previously-submitted job has completed. Re-raises any
      captured worker exception. Use before flushing / closing the
      file handler the worker may still be writing to.
    * ``shutdown(drain=True, timeout=...)`` — drain the queue (when
      ``drain=True``) and stop the worker thread. Always runs from
      the main / sim thread; safe to call multiple times.
    * ``pending`` — current queue depth (for diagnostics only;
      not authoritative).
    """

    def __init__(self, *, max_queue: int = 8, name: str = "mana-recorder-writer") -> None:
        if max_queue <= 0:
            raise ValueError(f"AsyncRecorderWriter.max_queue must be > 0 (got {max_queue})")
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._exception: BaseException | None = None
        self._exception_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    def submit(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        """Queue ``fn(*args, **kwargs)`` for the writer thread.

        Blocks if the queue is full (back-pressure: sim cannot run
        faster than the writer can persist). Raises any pending
        worker exception synchronously before queueing the new job
        so the caller sees IO failures right away.
        """
        self._raise_pending()
        if self._stop.is_set():
            raise RuntimeError("AsyncRecorderWriter has been shut down; cannot submit new work.")
        self._queue.put((fn, args, kwargs))

    def barrier(self, timeout: float | None = None) -> None:
        """Block until every submitted job has completed.

        The wait intentionally pumps the Kit app between polls. HDF5
        writes can take long enough for livestream/WebRTC to time out if
        the simulation thread sits in ``Queue.join()`` without giving Kit
        a chance to push frames.
        """
        import time

        deadline = time.monotonic() + timeout if timeout is not None else None
        while True:
            if self._queue.unfinished_tasks == 0:
                break
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"AsyncRecorderWriter.barrier timed out after {timeout:.1f}s")
            self._update_kit_app()
            time.sleep(0.02)
        self._raise_pending()

    def shutdown(self, *, drain: bool = True, timeout: float | None = 30.0) -> None:
        """Stop the worker thread.

        When ``drain=True`` (default) wait for all queued jobs to
        finish first; otherwise stop after the in-flight job
        completes and discard the rest. Always idempotent.
        """
        with self._shutdown_lock:
            if self._shutdown_done:
                return
            self._shutdown_done = True

        if drain:
            try:
                self.barrier(timeout=timeout)
            except TimeoutError:
                # Caller asked for a bounded shutdown; proceed to
                # poison even if some jobs are still queued. Their
                # exceptions (if any) will surface on _raise_pending
                # at exit.
                pass

        self._stop.set()
        try:
            self._queue.put_nowait(_POISON_PILL)
        except queue.Full:
            # Queue full + already-running worker means it'll see
            # _stop on next loop iteration after the in-flight job;
            # safe to skip the poison pill in that case.
            pass

        self._thread.join(timeout=timeout)
        # Re-raise any captured worker exception so caller sees IO
        # failures even when shutdown happens during finally clauses.
        self._raise_pending()

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _loop(self) -> None:
        while True:
            try:
                item = self._queue.get()
            except Exception:  # noqa: BLE001 — defensive; queue.get rarely raises
                continue
            try:
                if item is _POISON_PILL:
                    return
                fn, args, kwargs = item
                fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 — capture all to surface on submit/barrier
                with self._exception_lock:
                    if self._exception is None:
                        self._exception = exc
            finally:
                # task_done must always run, even for the poison
                # pill, otherwise barrier() blocks forever.
                try:
                    self._queue.task_done()
                except ValueError:
                    # task_done called more times than put — defensive
                    # guard for unexpected shutdown re-entry.
                    pass
            if self._stop.is_set() and self._queue.empty():
                return

    def _raise_pending(self) -> None:
        with self._exception_lock:
            if self._exception is None:
                return
            exc, self._exception = self._exception, None
        raise exc

    @staticmethod
    def _update_kit_app() -> None:
        try:
            import omni.kit.app
        except Exception:
            return
        try:
            omni.kit.app.get_app().update()
        except Exception:
            return
