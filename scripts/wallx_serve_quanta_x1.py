"""Start the Wall-X WebSocket server for the QUANTA_X1 checkpoint.

Run inside the Wall-X environment (``pip install -e .`` of https://github.com/X-Square-Robot/wall-x):

    python scripts/wallx_serve_quanta_x1.py --checkpoint-path /path/to/quanta-x1-wallx-20k --port 8000

It starts the stock Wall-X server (``wall_x._vendor.harrix.serving.launch_serving``) with the
settings used for QUANTA_X1 (``--robot-type ex001``, serialized actions, action horizon 32).

One server method is replaced at start-up. In this checkpoint the ``head_pos`` slot carries the base
velocity ``(vx, vy)``. The stock ``EX001Robot._get_head_action`` answers that slot with the request
state whenever the request contains ``head_pos``, so the model prediction would be dropped and the
base would not start moving. The replacement returns the model prediction. Wall-X itself is not modified.

Arguments after ``--`` are forwarded to ``launch_serving``.
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", required=True, help="Checkpoint directory (model.safetensors, config.yml, ...).")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1", help="Bind address. The server has no authentication.")
    parser.add_argument("--cuda-id", default="0", help="Value of CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--action-horizon", type=int, default=32)
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def patch_head_action() -> None:
    import numpy as np
    from wall_x._vendor.harrix.serving._wallx_infer import robot

    stock = robot.EX001Robot._get_head_action

    def _get_head_action(self, robot_state_action_data):
        data = robot_state_action_data.data
        if data.get("action_head_actions") is not None and data.get("state_head_actions") is not None:
            state = np.asarray(data["state_head_actions"], dtype=np.float64).reshape(1, -1)
            action = np.asarray(data["action_head_actions"], dtype=np.float64)
            if action.ndim == 1:
                action = action.reshape(-1, state.shape[-1])
            return np.concatenate([state, action], axis=0)
        return stock(self, robot_state_action_data)

    robot.EX001Robot._get_head_action = _get_head_action


def main() -> None:
    args, extra = parse_args(sys.argv[1:])
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    patch_head_action()
    sys.argv = [
        "launch_serving",
        "--env", "X2ROBOT",
        "--host", args.host,
        "--port", str(args.port),
        "--image-passing-mode", "base64",
        "--serialize-actions",
        "model-config:server-model-config",
        "--model-config.checkpoint-path", args.checkpoint_path,
        "--model-config.action-horizon", str(args.action_horizon),
        "--model-config.robot-type", "ex001",
        "--model-config.robot-action-interpolate-multiplier", "1",
        "--model-config.robot-action-end-ratio", "1.0",
        "--model-config.model-device", "cuda",
        *extra,
    ]
    runpy.run_module("wall_x._vendor.harrix.serving.launch_serving", run_name="__main__")


if __name__ == "__main__":
    main()
