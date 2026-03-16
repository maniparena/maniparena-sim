# OpenCVPR

OpenCVPR is a data collection, replay, and policy evaluation framework for the **Desktop** bimanual robot, built on [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and [IsaacLab-Arena](https://github.com/LightwheelAI/IsaacLab-Arena).

## Features

- **Desktop** bimanual robot with SE(3) differential-IK and joint-position action spaces
- **3 tabletop tasks**: `sort_blocks`, `fruits_to_basket`, `buttons_contact`
- **3 teleoperation modes**: keyboard, VR (OpenXR), master-slave
- **Trajectory replay**: state / joint / end-effector modes from HDF5 or LeRobot datasets
- **Policy evaluation**: closed-loop inference via WebSocket server-client architecture
- **Data export**: `desktop_lerobot` (Parquet + video) and `hdf5` formats

## Prerequisites

- **OS**: Ubuntu 22.04 / 24.04 with NVIDIA GPU
- **Python**: 3.11
- **CUDA**: 12.8 (recommended)
- **NVIDIA Driver**: 570+ (recommended)
- Isaac Sim, Isaac Lab, and IsaacLab-Arena (not included; see their respective installation guides)

## Installation

```bash
git clone https://github.com/maniparena/opencvpr.git
cd opencvpr
pip install -e .
```

USD assets under `assets/` are tracked via [Git LFS](https://git-lfs.github.com/). Make sure LFS is installed before cloning:

```bash
git lfs install
git clone https://github.com/maniparena/opencvpr.git
```

## Project Structure

```text
opencvpr/
├── assets/           # USD scene & object assets (Git LFS)
├── configs/          # YAML configurations (collect, eval, replay, tasks)
├── scripts/          # Entry-point scripts (collect, eval, replay)
└── src/opencvpr/     # Python package
    ├── assets/       # Asset registration (environments, objects)
    ├── embodiment/    # Robot, actions, sensors, teleop devices
    ├── environment/   # Environment assembly & scene building
    ├── task/          # Task definitions & builders
    ├── terms/         # Recorders, replay, terminations, metrics
    ├── planners/      # Teleoperation planners
    ├── loops/         # Collection, replay loops
    ├── policy/        # Policy inference client
    └── utils/         # Math & camera utilities
```

## Usage

### Data Collection

Collect teleoperation demonstrations with keyboard, VR, or master-slave control:

```bash
# Keyboard
python scripts/collect.py \
    --task sort_blocks \
    --control-mode keyboard \
    --config configs/collect/keyboard.yaml

# VR (OpenXR)
python scripts/collect.py \
    --task sort_blocks \
    --control-mode vr \
    --config configs/collect/vr.yaml

# Master-slave arm
python scripts/collect.py \
    --task sort_blocks \
    --control-mode master_slave \
    --config configs/collect/master_slave.yaml
```

#### Keyboard Controls

The keyboard controller drives the **active arm** (left by default) in SE(3) space.

**Translation:**

| Key | Axis | Direction |
|-----|------|-----------|
| `W` | X    | +X (forward) |
| `S` | X    | −X (backward) |
| `A` | Y    | +Y (left) |
| `D` | Y    | −Y (right) |
| `Q` | Z    | +Z (up) |
| `E` | Z    | −Z (down) |

**Rotation:**

| Key | Axis | Direction |
|-----|------|-----------|
| `Z` | Roll  | +Roll (around X) |
| `X` | Roll  | −Roll (around X) |
| `T` | Pitch | +Pitch (around Y) |
| `G` | Pitch | −Pitch (around Y) |
| `C` | Yaw   | +Yaw (around Z) |
| `V` | Yaw   | −Yaw (around Z) |

**Function keys:**

| Key | Action |
|-----|--------|
| `B` | Switch active arm (left ↔ right) |
| `K` | Toggle gripper (open / close) |
| `H` | Save current episode as success |
| `R` | Reset / skip current episode |

#### VR Control

VR teleoperation supports Apple Vision Pro (dual-hand tracking) and PICO / Quest (vr-controller, vr-hand). For configuration and setup, refer to the Isaac Sim documentation:

<https://isaac-sim.github.io/IsaacLab/main/source/how-to/cloudxr_teleoperation.html#cloudxr-teleoperation>

Controller quick guide (PICO / Quest):
- **Trigger**: open / close gripper
- **Joystick**: move the base
- **Grip button**: toggle base movement mode (rotate in place or translate)
- By default, the right controller operates the robot. For dual-arm control, use the left controller's trigger or joystick for the second arm.

### Trajectory Replay

Replay collected demonstrations for verification or LeRobot format export:

```bash
# HDF5 replay (state / joint / ee)
python scripts/replay.py \
    --task sort_blocks \
    --config configs/replay/hdf5.yaml

# LeRobot replay (joint / ee)
python scripts/replay.py \
    --task sort_blocks \
    --config configs/replay/lerobot.yaml
```

Set `export_lerobot: true` in `configs/replay/hdf5.yaml` to export LeRobot format during HDF5 replay.

### Policy Evaluation

Evaluate a remote policy via WebSocket server-client architecture:

```bash
python scripts/eval.py \
    --task sort_blocks \
    --config configs/eval/robot.yaml
```

Configure the policy server address in `configs/eval/robot.yaml`:

```yaml
policy_config:
  model_address: "localhost"
  model_port: 8000
  instruction: "sort the blocks"
```

The policy server must be running and accessible before launching evaluation.

## Configuration

All runtime parameters are driven by YAML files under `configs/`:

| Config | Description |
|--------|-------------|
| `configs/collect/keyboard.yaml` | Keyboard collection settings |
| `configs/collect/vr.yaml` | VR collection settings |
| `configs/collect/master_slave.yaml` | Master-slave collection settings |
| `configs/replay/hdf5.yaml` | HDF5 replay (with optional LeRobot export) |
| `configs/replay/lerobot.yaml` | LeRobot replay |
| `configs/eval/robot.yaml` | Robot policy evaluation |
| `configs/tasks/*.yaml` | Task-specific parameters |

## Supported Tasks

| Task | Description |
|------|-------------|
| `sort_blocks` | Sort colored blocks onto matching colored papers |
| `fruits_to_basket` | Pick fruits and place them into a basket |
| `buttons_contact` | Press all three colored buttons in sequence |

## License

Apache License 2.0
