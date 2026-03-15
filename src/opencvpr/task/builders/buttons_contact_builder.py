"""Builder for buttons-contact runtime configs."""

from __future__ import annotations

import numpy as np
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import SceneEntityCfg, TerminationTermCfg
from isaaclab.sensors.contact_sensor.contact_sensor_cfg import ContactSensorCfg
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_arena.utils.configclass import make_configclass

from opencvpr.task.buttons_contact import ButtonsContactTask, ButtonsContactTerminationsCfg
from opencvpr.task.utils import build_default_events_cfg, find_assets_by_tag
from opencvpr.terms.terminations import contacts_all_buttons_once


class ButtonsContactBuilder:
    DEFAULT_POSE_RANGE = {"x": (0.0, 0.3), "y": (-0.7, 0.15), "z": (-0.17, -0.17)}

    def finalize(self, task_cfg, scene):
        buttons = find_assets_by_tag(scene, "button")
        if len(buttons) < 3:
            raise ValueError("ButtonsContactTask needs at least 3 buttons")
        task = ButtonsContactTask(button_1=buttons[0], button_2=buttons[1], button_3=buttons[2])
        fields = []
        for index, button in enumerate(buttons):
            sensor_cfg = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*_arm_gripper_.*_link",
                filter_prim_paths_expr=[button.get_prim_path(), f"{button.get_prim_path()}/.*"],
            )
            fields.append((f"button_{index}_contact_sensor", ContactSensorCfg, sensor_cfg))
        task.scene_config = make_configclass("ButtonsContactSceneCfg", fields)()
        task.events_cfg = build_default_events_cfg(buttons, domain_randomization_cfg=getattr(task_cfg, "domain_randomization_cfg", None), pose_range=self.DEFAULT_POSE_RANGE)
        task.termination_cfg = ButtonsContactTerminationsCfg(
            success=TerminationTermCfg(
                func=contacts_all_buttons_once,
                params={
                    "contact_sensor_cfgs": [SceneEntityCfg(f"button_{index}_contact_sensor") for index in range(len(buttons))],
                    "force_threshold": 1.0,
                },
            )
        )
        task.viewer_cfg = get_viewer_cfg_look_at_object(lookat_object=buttons[0], offset=np.array([-1.0, 0.0, 0.5])) if buttons else ViewerCfg()
        return task
