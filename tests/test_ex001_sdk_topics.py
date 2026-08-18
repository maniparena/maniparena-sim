"""SDK topic-contract tests (no Isaac / ROS runtime required)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

from maniparena_sim.ros.ex001_sdk_topics import (
    EX001_SDK_PUBLISH_TOPICS,
    EX001_SDK_SUBSCRIBE_TOPICS,
    iter_banned_sdk_topics,
    sdk_gripper_to_sim,
    sim_gripper_to_sdk,
)

_ROS_DIR = Path(__file__).resolve().parents[1] / "src" / "maniparena_sim" / "ros"
_COMMUNICATOR = _ROS_DIR / "ex001_ros_communicator.py"


def _dict_keys_from_assign(source: str, name: str) -> set[str]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "EX001RosCommunicator":
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == name:
                            assert isinstance(item.value, ast.Dict)
                            keys: set[str] = set()
                            for key in item.value.keys:
                                assert isinstance(key, ast.Constant) and isinstance(key.value, str)
                                keys.add(key.value)
                            return keys
    raise AssertionError(f"{name} not found on EX001RosCommunicator")


def test_communicator_publishers_match_sdk_contract():
    keys = _dict_keys_from_assign(_COMMUNICATOR.read_text(encoding="utf-8"), "PUBLISHERS")
    assert keys == EX001_SDK_PUBLISH_TOPICS


def test_communicator_subscribers_match_sdk_contract():
    keys = _dict_keys_from_assign(_COMMUNICATOR.read_text(encoding="utf-8"), "SUBSCRIBERS")
    assert keys == EX001_SDK_SUBSCRIBE_TOPICS


def test_no_banned_sdk_topics_on_surface():
    source = _COMMUNICATOR.read_text(encoding="utf-8")
    pubs = _dict_keys_from_assign(source, "PUBLISHERS")
    subs = _dict_keys_from_assign(source, "SUBSCRIBERS")
    assert iter_banned_sdk_topics(pubs | subs) == []


def test_ros_sources_reject_legacy_literals():
    pattern = re.compile(r"/mock_robot_interface|/chassis/odom|/manaenv/")
    offenders: list[str] = []
    for path in (
        _ROS_DIR / "ex001_ros_communicator.py",
        _ROS_DIR / "ex001_control_callbacks.py",
        _ROS_DIR / "ex001_data_acquirers.py",
        _ROS_DIR / "ros_bridge.py",
    ):
        text = path.read_text(encoding="utf-8")
        if pattern.search(text):
            offenders.append(path.name)
    assert offenders == []


def test_banned_helper_flags_legacy_names():
    banned = iter_banned_sdk_topics(
        {
            "/odom",
            "/chassis/odom",
            "/mock_robot_interface/state",
            "/manaenv/tracked_pose",
            "/scan",
        }
    )
    assert banned == [
        "/chassis/odom",
        "/manaenv/tracked_pose",
        "/mock_robot_interface/state",
    ]


def test_gripper_scale_roundtrip_endpoints():
    assert sdk_gripper_to_sim(0.0) == 0.0
    assert abs(sdk_gripper_to_sim(4.5) - 1.89) < 1e-9
    assert sim_gripper_to_sdk(0.0) == 0.0
    assert abs(sim_gripper_to_sdk(1.89) - 4.5) < 1e-9
