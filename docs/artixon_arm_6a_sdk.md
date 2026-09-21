# ArtiXon Arm-6A SDK ROS2

ArtiXon Arm-6A 是仓库已有桌面双臂的公开名称，直接复用 `BimanualEmbodiment`
和 `assets/bimanual_robot/bimanual_robot.usd`，SDK 层配置法兰坐标和控制接口。
默认加载抓水果的桌面场景，但任务为空：没有成功判定、超时或自动重置，也不录制数据。
依赖和环境安装见 [安装说明](install.md)。

## 启动

```bash
source 3rd/isaaclabarena/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y ENABLE_CAMERAS=1
source /opt/ros/jazzy/local_setup.bash
python scripts/sdk_ros2.py --robot artixon_arm_6a --viz kit
```

配置：`configs/sdk_ros2/artixon_arm_6a_sdk_ros2.yaml`。可用 `--config PATH` 指定自定义配置。
无界面运行使用 `--viz none`；窗口聚焦后按 R 重置，关闭窗口或 Ctrl-C 退出。

## 控制与状态

默认 `ros.arm_control: ee`；切换为 `joint` 后使用关节指令。两种模式互斥。
每臂关节顺序为 `left_arm_joint1`…`6` / `right_arm_joint1`…`6`，角度单位为弧度。

| 接口 | 话题 | 类型 / 语义 |
| --- | --- | --- |
| 末端指令（ee） | `/left_arm_cartesian_controller/pose_cmd`、`/right_arm_cartesian_controller/pose_cmd` | `PoseStamped`，`base_link` 下绝对法兰位姿 |
| 关节指令（joint） | `/left_arm_joint_controller/commands`、`/right_arm_joint_controller/commands` | `Float64MultiArray`，每臂 6 值 |
| 夹爪指令 | `/left_gripper_controller/commands`、`/right_gripper_controller/commands` | `Float64MultiArray`，单值 0–5.717175 |
| 双臂状态 | `/left_arm/joint_states`、`/right_arm/joint_states` | `JointState` |
| 法兰状态 | `/left_arm/end_pose`、`/right_arm/end_pose` | `PoseStamped`，`base_link` 下绝对位姿 |
| 夹爪状态 | `/left_gripper/joint_states`、`/right_gripper/joint_states` | `JointState` |
| 全关节状态 | `/joint_states` | `JointState` |

末端为 `left_arm_link6` / `right_arm_link6` 法兰。位置单位米，四元数 **XYZW**。
指令 `header.frame_id` 使用 `base_link`（空值也按该坐标系处理）；先读取 `/left_arm/end_pose`
或右臂对应状态，再发送附近目标。EE 使用仿真 Jacobian 的 differential IK。

启用相机时发布 `/camera1/usb_cam1/image_raw/image_compressed`、
`/camera3/usb_cam3/image_raw/image_compressed` 和 `/camera_head_front/color/image_raw/compressed`。
`use_sim_time: true` 时发布 `/clock`。本机型没有底盘、升降和头部关节控制接口。
默认物理 120 Hz、控制 20 Hz、渲染调度 60 Hz；实际运行帧率另受计算与渲染耗时影响。
