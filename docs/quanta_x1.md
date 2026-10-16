# QUANTA_X1 使用指南

# 遥操作采集

采集入口为 `scripts/collect.py`，配置文件按控制方式区分（`keyboard` / `vuer` / `vr`）。

## 支持的任务

| Task | 底盘 | 说明 |
|------|------|------|
| `sort_blocks` | 固定 | 将三个彩色方块放到对应颜色的纸上 |
| `fruits_to_basket` | 固定 | 将苹果、香蕉、梨放入篮子，面包为干扰物 |
| `move_bottle_to_bin` | 移动 | 将桌上的瓶子放入垃圾桶 |
| `move_fruit_to_basket` | 移动 | 将水果桌上的水果放入另一张桌上的篮子 |
| `put_bottle_on_woodshelf` | 移动 | 将桌上的瓶子放到木架上 |

五个任务都支持键盘和 Vuer WebXR，`vr` 是 `vuer` 的别名。每回合物体随机，移动任务的机器人起点也随机；成功条件是物体到达目标且双臂回到初始姿态。

固定底盘任务使用 `assets/green_booth/`；移动任务使用 `assets/put_bottle_on_woodshelf/` 和 `assets/mobile_manip/`。`move_bottle_to_bin`、`move_fruit_to_basket` 的对象默认从对象池随机选择，并在当前进程内保持不变；可通过 `scene_options.seed`、`bottle`、`fruit`、`basket` 指定。

底盘参数（标定值，与 `QUANTA_X1_WHEEL_RADIUS_M` / `QUANTA_X1_WHEEL_TRACK_WIDTH_M` 一致）：`wheel_radius=0.0845`、`wheel_track_width=0.415`。Wall-X 全身推理见下文「Wall-X 评测」。

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
- 格式：每个 episode 同时写 HDF5 和 `lerobot/` 下的 LeRobot parquet；相机仍写为 HDF5 同目录的 sidecar MP4（左腕 / 右腕 / 头部 RGB，相机深度不写入 MP4）
- HDF5 每帧保存 action、processed action、机器人状态、物体状态和观测；QUANTA_X1 遥操 action 为 19D，评测 action 为 21D
- LeRobot 的 `action` 保留遥操实际 19D；`observation.state` 保存 23D 关节状态，不需要再做格式转换。若只需旧的 HDF5 输出，可在采集 YAML 中设置 `format: "hdf5"`。
- QUANTA_X1 评测记录同样双写；评测 LeRobot `action` 为 21D，并包含额外的 `head_yaw` / `head_pitch` 两列。
- 帧率：由配置 `step_hz` 决定，当前统一为 **20 Hz**

示例：

```text
~/maniparena_output/recordings/quanta_x1_fruits_to_basket_collect/
  quanta_x1_fruits_to_basket_collect_vr_episode0.hdf5
  lerobot/
    data/chunk-000/episode_000000.parquet
    meta/info.json
  quanta_x1_fruits_to_basket_collect_vr_episode0_videos/
    ...
```

## 演示

Vuer 遥操作 `fruits_to_basket` 三相机拼接预览（左腕 / 头部 / 右腕）：

![quanta_x1 vuer fruits_to_basket](../media/quanta_x1_vuer_fruits_to_basket.gif)

---

# Wall-X 评测

Wall-X（Wall-OSS）全身策略（21D：左右臂位姿+夹爪、左右轮速、lift、头部 yaw/pitch）在 Isaac Sim 里闭环评测，入口 `scripts/eval.py`。先在服务端起好 Wall-X 推理服务，再运行：

```bash
python scripts/eval.py \
  --robot quanta_x1 --task put_bottle_on_woodshelf \
  --config configs/eval/quanta_x1_put_bottle.yaml \
  --server-address 127.0.0.1 --server-port 8000 \
  --headless --enable_cameras --device cuda:0
```

`--task` 可选 `sort_blocks`、`fruits_to_basket`、`move_bottle_to_bin`、`move_fruit_to_basket`、`put_bottle_on_woodshelf`。仓库内配置：`quanta_x1_put_bottle.yaml`、`quanta_x1_move_bottle_to_bin.yaml`、`quanta_x1_move_fruit_to_basket.yaml`；固定底盘的两个任务沿用相同的 `policy_config`，`max_steps` 取 2400。每回合超时后重置并计一回合。

## 与数采对齐的评测设置

评测使用独立的环境构建（`build_quanta_x1_eval_gym_env`），只影响 `eval.py`，采集和 SDK 不变。

| 项 | 取值 | 说明 |
|----|------|------|
| 物理 / 环境步长 | 240 Hz / 30 Hz（decimation 8） | 与数采 manaenv 一致，相机每个环境步渲染一次 |
| 动作块 | `action_horizon=32`、`action_chunk_length=32` | 整块执行，不截断 |
| 插值 | `interpolation_multiplier=1.5` | 一个动作块播放 48 个环境步 |
| 超时 | 固定底盘 2400 步，移动任务 3000 步 | |
| 指令 | 与训练数据任务文本一致 | 见各 yaml |
| 手臂 IK | 在 `lift_link` 坐标系下求解 | lift 运动不影响手臂控制，与 manaenv 的 `isaac_base_body` 一致 |
| 相机 | 头部 1280×720 针孔，腕部鱼眼 | 通过 `enable_usd_lens_models()` 写入 USD 镜头畸变模型；评测中去掉只含深度的底盘相机（否则 RTX 不再渲染彩色图） |
| 物体与布局 | `sort_blocks`、`fruits_to_basket` 使用 `assets/quanta_x1_objects/` 里带 `_2` 后缀的资产 | 与桌面臂的 `sort_blocks` 资产、布局分开维护（`scene_builder.py` 中独立的场景构建函数） |
| 固定底盘出生位姿 | sort：`pos=(-0.52,-0.26,-0.985)`、lift 0.455；fruits：`pos=(-0.60,-0.26,-0.985)`、lift 0.379；head pitch 0.35 | 见 `task/quanta_tabletop_layouts.py` |

### Lift 固定为数采值

数采里每个任务的 lift 指令全程恒定（`sort_blocks` 0.455，其余四个任务 0.379；实测值分别约 0.441 和 0.365）。现有 checkpoint 的 lift 输出不稳定：在 `fruits_to_basket` 和移动任务里，lift 会在 30~120 步内漂到 0.45 左右（相当于 sort 的高度），手臂实际抬高约 7 cm，导致抓取偏上、抓不到。所以评测中 **lift 不作为推理控制量**：`eval.py` 按任务把 lift 指令固定为数采值（`quanta_x1_demo_lift_command`），模型输出的 lift 被忽略，lift 状态仍然作为输入送给模型。可以用 `policy_config.lift_hold` 覆盖该值。

非 sort 任务的机器人出生 lift 也设为数采实测状态（指令 − 0.014 m），保证第一次推理时的 lift 状态与训练数据一致。

### 服务端字段映射

训练数据（LeRobot v3.0，26D，30 fps）末尾顺序是 `[底盘 twist(3), lift, head(2)]`，而推理服务按 `dof_config` 的名字 `[head(2), height, velocity(3)]` 填充。客户端 `wire_layout: dataset`（默认）按数据顺序重新映射请求和响应字段，不用改服务端。

### 底盘

- 轮速由 `twist_to_wheel_vel` 换算，默认使用标定几何 `QUANTA_X1_WHEEL_RADIUS_M=0.0845`、`QUANTA_X1_WHEEL_TRACK_WIDTH_M=0.415`，不要在 yaml 里再写旧值（0.084 / 0.458 会让转向偏快约 10%）。
- 万向轮摩擦为 0 且摩擦合并模式取 min，驱动轮为球体碰撞体，用来消除低速转向死区，见 SDK 的 `fix(assets)` 提交。
- 第一次推理需要完整的 32 行动作块：从静止起步时模型前约 15 行输出接近 0，之后才开始加速。

### 录制与视频

`enable_recording: true` 时每个回合单独导出一个 hdf5（含成功回合）和三路相机 mp4（头部、左腕、右腕）。

## 已知问题

- checkpoint 对 lift 状态敏感：不固定 lift 时，lift 状态超过约 0.41 会触发输出 0.453（sort 的值），形成正反馈。固定 lift 后 `put_bottle_on_woodshelf` 单回合成功，`fruits_to_basket` 能抓起并放入部分水果，`move_bottle_to_bin`、`move_fruit_to_basket` 仍会停在离物体较远的位置、抓不到。
- 单回合结果有随机性，同一 checkpoint 多次评测的成功与否会变化，需要多回合统计。

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
