"""Manual SDK session with no task objectives, episode endings or recording."""

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.tasks.no_task import NoTask


@configclass
class SdkResetEventsCfg:
    reset_scene_to_default = EventTermCfg(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )


class SdkTask(NoTask):
    """Keep the scene interactive until the user resets or closes it."""

    def __init__(self):
        super().__init__()
        self.viewer_cfg = super().get_viewer_cfg()

    def get_viewer_cfg(self):
        return self.viewer_cfg

    def get_events_cfg(self):
        # InitialStateCfg supplies default buffers; an explicit event must write
        # them to physics both at startup and when the user presses R.
        return SdkResetEventsCfg()

    def get_metrics(self):
        return []


def configure_sdk_session(env_cfg):
    """Remove task and recorder managers before Gym registration.

    Scene and embodiment reset events remain available for the user's reset
    action; no fruit-task success, failure or randomization terms are installed.
    """
    env_cfg.terminations = {}
    env_cfg.rewards = None
    env_cfg.metrics = None
    env_cfg.recorders = None
    env_cfg.demo_recorder_config = None
    env_cfg.episode_recorders = None
    return env_cfg
