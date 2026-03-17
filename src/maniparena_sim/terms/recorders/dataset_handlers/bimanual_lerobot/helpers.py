"""Helpers for binding a Bimanual LeRobot handler to one output layout."""

from __future__ import annotations

from maniparena_sim.terms.recorders.dataset_handlers.bimanual_lerobot.handler import BimanualLeRobotDatasetFileHandler
from maniparena_sim.terms.recorders.dataset_handlers.utils import resolve_root_path, safe_name


def resolve_bimanual_lerobot_dataset_layout(*, save_path: str, env_name: str, device_name: str | None = None) -> dict[str, str]:
    root = resolve_root_path(save_path)
    dataset_dir_name = safe_name(env_name)
    if device_name:
        dataset_dir_name = f"{dataset_dir_name}_{safe_name(device_name)}"
    base_root = root / dataset_dir_name
    return {"base_root": str(base_root), "joint_root": str(base_root / "joint"), "ee_root": str(base_root / "ee")}


def create_bimanual_lerobot_handler_type(
    *,
    save_path: str,
    env_name: str,
    task_name: str,
    fps: float,
    device_name: str | None = None,
) -> type:
    layout = resolve_bimanual_lerobot_dataset_layout(save_path=save_path, env_name=env_name, device_name=device_name)
    class_name = f"BimanualLeRobot_{safe_name(env_name)}_{safe_name(device_name or 'default')}_DatasetFileHandler"
    return type(
        class_name,
        (BimanualLeRobotDatasetFileHandler,),
        {"output_layout": layout, "task_name": task_name or env_name, "fps": float(fps)},
    )


def is_bimanual_lerobot_handler_type(handler_cls: type | None) -> bool:
    return bool(handler_cls is not None and issubclass(handler_cls, BimanualLeRobotDatasetFileHandler))
