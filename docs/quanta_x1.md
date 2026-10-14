# QUANTA_X1 使用指南

# 遥操作采集

采集入口为 `scripts/collect.py`，配置文件按控制方式区分（`keyboard` / `vuer`）。

## 支持的任务

远程数采场景配置对齐说明见 [quanta_x1_task_alignment.md](quanta_x1_task_alignment.md)。

| Task | 底盘 | 说明 |
|------|------|------|
| `sort_blocks` | 固定 | 将三个彩色方块放到对应颜色的纸上 |
| `fruits_to_basket` | 固定 | 将苹果、香蕉、梨放入篮子，面包为干扰物 |
| `move_bottle_to_bin` | 移动 | 将桌上的瓶子放入垃圾桶 |
| `move_fruit_to_basket` | 移动 | 将水果桌上的水果放入另一张桌上的篮子 |
| `put_bottle_on_woodshelf` | 移动 | 将桌上的瓶子放到木架上 |

物体位置每回合随机，移动任务的机器人起点同样随机。成功条件为物体放入目标后双臂回到初始姿态。`buttons_contact` 仅支持 bimanual。

固定底盘任务使用 `assets/green_booth/`，移动任务使用 `assets/put_bottle_on_woodshelf/` 和 `assets/mobile_manip/`（垃圾桶、瓶子、水果、篮子，成员见 `assets/mobile_manip/pools.json`）。

`move_bottle_to_bin` 和 `move_fruit_to_basket` 的瓶子、水果、篮子在一个进程内固定为池中的一个，默认随机选取，可在评测配置里通过 `scene_options`（`seed`、`bottle`、`fruit`、`basket`）指定。

底盘差速使用仿真有效几何：`wheel_radius=0.084`、`wheel_track_width=0.458`
（见 `embodiment/robots/quanta_x1.py`）。Wall-X 全身推理（21D，含 head/lift/底盘）入口：

```bash
python scripts/eval.py \
  --robot quanta_x1 --task put_bottle_on_woodshelf \
  --config configs/eval/quanta_x1_put_bottle.yaml \
  --viz kit
```

## 采集命令

### 键盘

Isaac Lab / Sim **6.0** 默认无界面；需要 Kit 视口时加 `--viz kit`（键盘遥操作必须）。

```bash
python scripts/collect.py \
  --robot quanta_x1 \
  --task fruits_to_basket \
  --control-mode keyboard \
  --config configs/collect/keyboard.yaml \
  --viz kit
```

将 `fruits_to_basket` 换成上表其他任务即可切换。

### Vuer（Pico WebXR）

```bash
python scripts/collect.py \
  --robot quanta_x1 \
  --task fruits_to_basket \
  --control-mode vuer \
  --config configs/collect/vuer.yaml \
  --viz kit
```

启动后默认 WebSocket 端口 `8012`（可在 `configs/collect/vuer.yaml` 的 `teleop_config.vuer_port` 修改）。Pico 走 USB `adb reverse`，浏览器打开 Vuer 客户端并连到本机端口：

```bash
adb reverse tcp:8012 tcp:8012
```

Pico 浏览器打开：

```text
https://vuer.ai?ws=ws://localhost:8012
```

进页面后点 Enter VR；终端应出现 `Vuer WebSocket connected`。Left-X 开始遥操，Left-Y 标记成功，Right-A 重置。

## 键盘控制

启动后**点击 Isaac Sim 视口**使窗口获得焦点，否则按键无效。

| 按键 | 功能 |
|------|------|
| `B` | 切换控制目标：左臂 → 右臂 → 底盘 |
| `W` / `S` / `A` / `D` / `Q` / `E` | 平移当前目标（手臂或底盘） |
| `Z` / `X` / `T` / `G` / `C` / `V` | 旋转当前手臂（底盘模式下无效） |
| `K` | 切换当前手臂夹爪开/合（映射到关节 `0` ~ `1.89`） |
| `H` | 标记本 episode 成功并导出 |
| `R` | 放弃本 episode / 重置 |

底盘模式下：`W`/`S` 前进后退，`A`/`D` 或 `Q`/`E` 转向。

灵敏度可在 `configs/collect/keyboard.yaml` 调整：

```yaml
teleop_config:
  position_scale: 0.02
  rotation_scale: 0.2
```

## Vuer 控制

首次进入需用**左手 X** 开始遥操作（捕获参考位姿），之后：

| 输入 | 功能 |
|------|------|
| 双手手柄位姿 | 双臂绝对 IK 目标（增量映射到基座坐标系） |
| 扳机（Trigger） | 夹爪连续开合（`0` ~ `1.89`） |
| 左摇杆 | 底盘线速度 / 角速度 |
| 右摇杆 | 升降台（lift） |
| 左手 X | 开始 / 重新捕获遥操作参考 |
| 左手 Y | 标记本 episode 成功并导出 |
| 右手 A | 放弃本 episode / 重置 |

## 录制输出

- 默认目录：`~/maniparena_output/recordings/quanta_x1_<task>_collect/`
- 格式：HDF5 episode + sidecar MP4（左腕 / 右腕 / 头 / 底盘相机）
- 帧率：由配置 `step_hz` 决定，当前统一为 **20 Hz**

示例：

```text
~/maniparena_output/recordings/quanta_x1_fruits_to_basket_collect/
  quanta_x1_fruits_to_basket_collect_vuer_episode0.hdf5
  quanta_x1_fruits_to_basket_collect_vuer_episode0_videos/
    ...
```

## 演示

Vuer 遥操作 `fruits_to_basket` 三相机拼接预览（左腕 / 头部 / 右腕）：

![quanta_x1 vuer fruits_to_basket](../media/quanta_x1_vuer_fruits_to_basket.gif)

---

# SDK ROS2

## ROS 环境配置

ROS2 桥依赖 Isaac Sim 的 `isaacsim.ros2.bridge` 扩展。当前为 Isaac Sim **6.0**（Python 3.12），按官方 ROS 2 Jazzy 安装即可：

参考：<https://docs.ros.org/en/jazzy/Installation.html>

## 启动命令

先按主 [README](../README.md) 激活仿真环境，再 source ROS 2，最后启动：

```bash
source 3rd/isaaclabarena/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y
export ENABLE_CAMERAS=1
source /opt/ros/jazzy/local_setup.bash
python scripts/sdk_ros2.py \
  --config configs/sdk_ros2/quanta_x1_sdk_ros2.yaml \
  --viz kit
```

初始 Kit 视口对准机器人一次（`viewer_cfg.origin_type: world`），之后不跟车。可在
`configs/sdk_ros2/quanta_x1_sdk_ros2.yaml` 改 `eye` / `lookat`（世界系，单位米）。

启动后键盘控制（先点击 Isaac Sim 视口获取焦点）：

| 按键 | 功能 |
|------|------|
| `W` / `S` | 底盘前进 / 后退 |
| `A` / `D` 或 `Q` / `E` | 底盘转向 |
| `R` | 重置机器人到初始位姿 |

外部 ROS2 节点同时可通过 `/chassis/cmd_vel` 驱动底盘，速度与键盘叠加。

## ROS2 话题

ROS 节点名为 `sx001_ros_communicator`，话题合同见 `src/maniparena_sim/ros/quanta_x1_sdk_topics.py`。

| Topic Name | Message Type |
|------------|--------------|
| **PUBLISHERS** | |
| `/joint_states` | `JointState` |
| `/left_arm/joint_states` / `/right_arm/joint_states` | `JointState` |
| `/left_gripper/joint_states` / `/right_gripper/joint_states` | `JointState`（0–1.89） |
| `/head/joint_states` | `JointState` |
| `/lift/joint_states` | `JointState` |
| `/left_arm/end_pose` / `/right_arm/end_pose` | `PoseStamped` |
| `/odom` | `Odometry` |
| `/tracked_pose` | `PoseStamped` |
| `/hal/chassis/imu` | `Imu` |
| `/scan` | `LaserScan`（RTX 直出） |
| `/tf` / `/tf_static` | TF（`a_d_laser` 在 static） |
| `/clock` | `Clock`（`use_sim_time: true`） |
| `/camera_chassis_front/depth/points` | `PointCloud2` |
| `/camera1/...` / `/camera3/...` | 左右腕 RGB |
| `/camera_head_front/color/image_raw/compressed` | 头 RGB |
| **SUBSCRIBERS** | |
| `/left_arm_cartesian_controller/pose_cmd` / `/right_arm_cartesian_controller/pose_cmd` | `PoseStamped`（`ros.arm_control: ee`） |
| `/left_arm_joint_controller/commands` / `/right_arm_joint_controller/commands` | `Float64MultiArray`（`ros.arm_control: joint`） |
| `/left_gripper_controller/commands` | `Float64MultiArray`（0–1.89） |
| `/right_gripper_controller/commands` | `Float64MultiArray`（0–1.89） |
| `/head_position_controller/commands` | `Float64MultiArray` |
| `/lift_position_controller/commands` | `Float64MultiArray` |
| `/chassis/cmd_vel` | `Twist`（超时 0.25 s 清零） |

双臂控制由 YAML `ros.arm_control` 选择，默认 **`ee`**：

```yaml
ros:
  arm_control: ee    # 或 joint
```

## 演示

`nav_f16` 场景中的 QUANTA_X1 仿真：

![quanta_x1 SDK ROS2 sim](../media/quanta_x1_sdk_ros2_sim.png)

RViz 中点云、`/scan` 与 TF 可视化：

![quanta_x1 SDK ROS2 RViz](../media/quanta_x1_sdk_ros2_rviz.png)
