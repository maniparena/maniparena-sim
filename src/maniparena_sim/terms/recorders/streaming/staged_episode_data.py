"""CPU-staged drop-in replacement for upstream ``EpisodeData``."""

from __future__ import annotations

import torch

from maniparena_sim.terms.recorders.streaming.stager import ChunkedStager
from maniparena_sim.utils.math_utils import to_numpy_keep_shape


class StagedEpisodeFrameUtils:
    """Frame/key helpers for staged episode recording."""

    @staticmethod
    def is_depth_sidecar_key(key: str) -> bool:
        return False

    @staticmethod
    def is_rgb_video_frame(value) -> bool:
        if not hasattr(value, "shape"):
            return False
        shape = tuple(value.shape)
        if len(shape) < 3 or shape[-1] != 3:
            return False
        dtype = getattr(value, "dtype", None)
        return dtype is not None and "uint8" in str(dtype)

    @classmethod
    def is_sidecar_frame(cls, key: str, value) -> bool:
        if not hasattr(value, "shape"):
            return False
        shape = tuple(value.shape)
        if len(shape) < 2:
            return False
        return cls.is_rgb_video_frame(value)

    @staticmethod
    def is_semantic_sidecar_key(key: str) -> bool:
        return "semantic_segmentation" in str(key)

    @staticmethod
    def episode_env_id(episode) -> int:
        env_id = getattr(episode, "_env_id", 0)
        return int(env_id) if env_id is not None else 0

    @staticmethod
    def stack_and_release(data: dict) -> None:
        for key in list(data.keys()):
            value = data[key]
            if isinstance(value, list):
                data[key] = torch.stack(value)
                value.clear()
            elif isinstance(value, dict):
                StagedEpisodeFrameUtils.stack_and_release(value)


class StagedEpisodeDataRegistry:
    """Lazy builder for the runtime ``ManaStagedEpisodeData`` subclass."""

    _runtime_class: type | None = None

    @classmethod
    def runtime_class(cls) -> type:
        if cls._runtime_class is None:
            cls._runtime_class = cls._build_runtime_class()
        return cls._runtime_class

    @classmethod
    def construct(cls, pinned: bool = False, chunk_size: int = 0, video_sink=None):
        return cls.runtime_class()(pinned=pinned, chunk_size=chunk_size, video_sink=video_sink)

    @classmethod
    def is_instance(cls, obj) -> bool:
        if cls._runtime_class is None:
            return False
        return isinstance(obj, cls._runtime_class)

    @classmethod
    def _build_runtime_class(cls) -> type:
        from isaaclab.utils.datasets.episode_data import EpisodeData

        utils = StagedEpisodeFrameUtils

        class _ManaStagedEpisodeData(EpisodeData):
            def __init__(self, pinned: bool = False, chunk_size: int = 0, video_sink=None) -> None:
                super().__init__()
                self._pinned = pinned
                self._chunk_size = int(chunk_size)
                self._video_sink = video_sink
                self._stager: ChunkedStager | None = (
                    ChunkedStager(
                        chunk_size=self._chunk_size,
                        pinned_memory=pinned,
                        video_sink=video_sink,
                    )
                    if self._chunk_size > 0
                    else None
                )
                self._stager_keys: set[str] = set()
                self._video_keys: set[str] = set()

            def _to_cpu(self, value, *, clone: bool = True):
                if not isinstance(value, torch.Tensor):
                    return value
                if value.device.type == "cpu":
                    # Lab recorder_manager may clone the batch once and pass
                    # clone=False; avoid a redundant host clone in that case.
                    if clone:
                        return value.detach().clone().contiguous()
                    return value.detach().contiguous()
                if self._pinned:
                    try:
                        host = torch.empty(value.shape, dtype=value.dtype, pin_memory=True)
                        host.copy_(value, non_blocking=True)
                        return host
                    except RuntimeError:
                        pass
                # Device→host always materializes a new tensor.
                return value.detach().to(device="cpu", copy=True).contiguous()

            def _set_at_path(self, key: str, value) -> None:
                sub_keys = key.split("/")
                cursor = self._data
                for idx, sub in enumerate(sub_keys):
                    if idx == len(sub_keys) - 1:
                        cursor[sub] = value
                        return
                    if sub not in cursor or not isinstance(cursor[sub], dict):
                        cursor[sub] = dict()
                    cursor = cursor[sub]

            def add(self, key: str, value, clone: bool = True):  # type: ignore[override]
                """Match Lab ``EpisodeData.add(..., clone=)`` (Lab 3 recorder API)."""
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        self.add(f"{key}/{sub_key}", sub_value, clone=clone)
                    return

                if key.startswith("camera_obs/"):
                    if utils.is_depth_sidecar_key(key) or utils.is_semantic_sidecar_key(key):
                        if self._video_sink is not None:
                            env_id = utils.episode_env_id(self)
                            self._video_keys.add(key)
                            if self._stager is not None and isinstance(value, torch.Tensor) and value.device.type == "cuda":
                                self._stager.add_video(key, value, env_id=env_id)
                            else:
                                staged = self._to_cpu(value, clone=clone)
                                frames = to_numpy_keep_shape(staged)
                                self._video_sink.append_chunk(env_id=env_id, key=key, frames=frames[None, ...])
                        return
                    if self._video_sink is not None and utils.is_sidecar_frame(key, value):
                        env_id = utils.episode_env_id(self)
                        self._video_keys.add(key)
                        if self._stager is not None and isinstance(value, torch.Tensor) and value.device.type == "cuda":
                            self._stager.add_video(key, value, env_id=env_id)
                        else:
                            staged = self._to_cpu(value, clone=clone)
                            frames = to_numpy_keep_shape(staged)
                            self._video_sink.append_chunk(env_id=env_id, key=key, frames=frames[None, ...])
                        return

                if self._stager is not None and isinstance(value, torch.Tensor) and value.device.type == "cuda":
                    self._stager.add(key, value)
                    self._stager_keys.add(key)
                    return

                sub_keys = key.split("/")
                cursor = self._data
                for idx, sub in enumerate(sub_keys):
                    if idx == len(sub_keys) - 1:
                        staged = self._to_cpu(value, clone=clone)
                        if sub not in cursor:
                            cursor[sub] = [staged]
                        else:
                            cursor[sub].append(staged)
                        return
                    if sub not in cursor:
                        cursor[sub] = dict()
                    cursor = cursor[sub]

            def is_empty(self):  # type: ignore[override]
                if self._stager_keys:
                    return False
                if self._video_keys:
                    return False
                return not bool(self._data)

            def pre_export(self):  # type: ignore[override]
                if self._stager is not None and self._stager_keys:
                    for key in sorted(self._stager_keys):
                        stacked = self._stager.drain(key)
                        if stacked is not None:
                            self._set_at_path(key, stacked)
                    self._stager_keys.clear()

                if self._stager is not None and self._video_keys:
                    for key in sorted(self._video_keys):
                        self._stager.drain_video(key)
                    self._video_keys.clear()

                if self._stager is not None:
                    self._stager.reset()

                utils.stack_and_release(self._data)

        _ManaStagedEpisodeData.__name__ = "ManaStagedEpisodeData"
        _ManaStagedEpisodeData.__qualname__ = "ManaStagedEpisodeData"
        return _ManaStagedEpisodeData


class ManaStagedEpisodeData:
    """Public facade for constructing staged episode buffers lazily."""

    def __new__(cls, pinned: bool = False, chunk_size: int = 0, video_sink=None):
        return StagedEpisodeDataRegistry.construct(
            pinned=pinned,
            chunk_size=chunk_size,
            video_sink=video_sink,
        )


def staged_episode_data_class() -> type:
    return StagedEpisodeDataRegistry.runtime_class()


def is_staged_episode_data(obj) -> bool:
    return StagedEpisodeDataRegistry.is_instance(obj)
