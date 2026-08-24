"""Task base configs and shared dataclasses."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, ClassVar, Dict, List, Literal, Optional

ConstraintCollection = Dict[str, Any] | list[Any]

BUILTIN_TASK_TYPES = ("sort_blocks", "fruits_to_basket", "buttons_contact")


@dataclass
class ConstraintCFG:
    mode: int = 0
    tags: str = ""


@dataclass
class PositionRandomizationCfg:
    AVAILABLE_METHODS: ClassVar[tuple[str, ...]] = ("uniform_offset",)
    method: str = "uniform_offset"
    range_x: list = field(default_factory=lambda: [0.0, 0.0])
    range_y: list = field(default_factory=lambda: [0.0, 0.0])
    range_z: list = field(default_factory=lambda: [0.0, 0.0])

    def to_range_dict(self) -> dict:
        return {"x": list(self.range_x), "y": list(self.range_y), "z": list(self.range_z)}


@dataclass
class OrientationRandomizationCfg:
    AVAILABLE_METHODS: ClassVar[tuple[str, ...]] = ("uniform_yaw",)
    method: str = "uniform_yaw"
    yaw_range: list = field(default_factory=lambda: [-3.14159, 3.14159])


@dataclass
class DomainRandomizationCfg:
    object_position: PositionRandomizationCfg | None = None
    object_orientation: OrientationRandomizationCfg | None = None

    @property
    def any_enabled(self) -> bool:
        return self.object_position is not None or self.object_orientation is not None


@dataclass
class TaskCFG:
    class_type: type | None = None
    task_type: str = "default_task"
    gen_method: Literal[0, 1, 2, 3] = 0
    tags: str = ""
    num_episodes: int = 100
    max_steps_per_episode: int = 100
    # None → Arena TaskBase default (20 s). Eval YAML can override.
    episode_length_s: Optional[float] = None
    seed: Optional[int] = None
    scene_constraints: ConstraintCollection = field(default_factory=dict)
    robot_constraints: ConstraintCollection = field(default_factory=dict)
    domain_randomization_cfg: DomainRandomizationCfg | None = None
    task_instruction: str = ""
    instruction: Optional[str] = None
    topic: Optional[str] = None
    subtasks: List[Any] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    is_long_horizon: bool = False
    num_actions: int = 0
    num_subtasks: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
