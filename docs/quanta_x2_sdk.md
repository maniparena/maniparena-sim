# quanta_x2 SDK ROS2

仅包含仿真 SDK 接口。默认加载空导航场景，没有成功判定、超时或自动重置，也不录制数据。
依赖和环境安装见 [安装说明](install.md)。

## 启动

整机模型位于 `assets/quanta_x2/quanta_x2.usd`，模型自身包含双臂和双 G 夹爪，
直接加载这个 USD，依赖均从仓库本地加载。
按安装说明执行 `git lfs pull` 取得资产即可。关节顺序见
`src/maniparena_sim/ros/sdk_profiles.py`。自定义模型可修改 `robot.usd_path`，相对路径按仓库根目录解析；使用自定义 USD 时请选 `ros.ik_backend: arm`。全身 QP 的模型数据与仓库自带 USD 校验绑定。

```bash
source 3rd/isaaclabarena/.venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES ACCEPT_EULA=Y ENABLE_CAMERAS=1
source /opt/ros/jazzy/local_setup.bash
python scripts/sdk_ros2.py --robot quanta_x2 --viz kit
```

配置：`configs/sdk_ros2/quanta_x2_sdk_ros2.yaml`。可用 `--config PATH` 指定自定义配置。
无界面运行使用 `--viz none`。窗口聚焦后，W/S 前后、A/D 转向、R 重置；关闭窗口或 Ctrl-C 退出。
键盘速度与 `/chassis/cmd_vel` 叠加；ROS 速度超过 0.25 秒仿真时间未更新会清零。
启动和按 R 均恢复双臂默认姿态：按 SDK 关节顺序，左臂为
`[-90, 90, 90, -90, 0, 0, 0]°`，右臂为 `[-90, -90, 90, -90, 0, 0, 0]°`。
此处角度用于说明姿态，ROS 关节指令仍使用弧度。

## 演示

空导航场景中的 quanta_x2 仿真：

![quanta_x2 SDK ROS2 sim](../media/quanta_x2_sdk_ros2_sim.png)

## 控制与状态

姿态采用 ROS **XYZW**，距离为米，关节角为弧度。

| 接口 | 话题 | 类型 / 语义 |
| --- | --- | --- |
| 末端指令（ee） | `/left_arm/pose_command/root_frame`、`/right_arm/pose_command/root_frame` | `PoseStamped`，相对 `base_link` 的绝对末端位姿 |
| 关节指令（joint） | `/smoother_command_input_left`、`/smoother_command_input_right` | `Float64MultiArray`，每臂 7 值 |
| 夹爪指令 | `/left_g_gripper_controller/commands`、`/right_g_gripper_controller/commands` | `Float64MultiArray`，单值 0–1.89 |
| 头部指令 | `/head_position_controller/commands` | `Float64MultiArray`，pitch、yaw |
| 腰部指令 | `/waist_forward_position_controller/commands` | `Float64MultiArray`，依次为 `bow_pitch_joint_01`、`02`、`03`、`bow_yaw_joint` |
| 底盘指令 | `/chassis/cmd_vel` | `Twist`，linear.x 与 angular.z |
| 双臂状态 | `/left_arm/state/joint`、`/right_arm/state/joint` | `JointState` |
| 末端状态 | `/whole_body_controller/left_wrist_pose`、`/whole_body_controller/right_wrist_pose` | `PoseStamped`，绝对 `base_link` 坐标 |
| 夹爪状态 | `/left_g_gripper_states/joint_states`、`/right_g_gripper_states/joint_states` | `JointState` |
| 其他状态 | `/joint_states`、`/waist/state/joint`、`/head/joint_states`、`/odom`、`/tracked_pose` | 关节、里程计、世界位姿 |

末端为整机模型的 `left_gripper_base_link` / `right_gripper_base_link`，位置不加夹爪尖端偏移。
`header.frame_id` 使用 `base_link`（空值也按该坐标系处理）。先读取末端状态，再发送附近目标；
不要把零位姿当作“保持当前位置”。

### 带腰部的全身 QP IK

X2 默认配置使用 `ros.ik_backend: whole_body_qp`。它同时求解 4 个腰部关节和双臂各 7 个关节，
无需 RL 模型。设为 `arm` 可恢复原来的独立手臂 IK；其他机器人配置继续使用原 IK。
依赖 `scipy>=1.11`、`osqp>=1.0,<2`，已经列入项目安装依赖。

- EE 模式下 QP 统一写入腰部和双臂目标。腰部话题保留订阅以兼容 SDK 接口，但独立腰部命令会被忽略，防止两个控制器争用；需要直接控制腰部时切到 joint 模式。
- 第一个有效 EE 命令到来时捕获测量位姿；只给一只手发命令时，另一只手保持捕获的位姿。已有另一只手目标时会继续追踪该目标。启动、重置及切换后，在收到新 EE 命令前保持原关节目标。
- 关节位置/速度、重心投影和采样几何间隙进入 QP；下发目标的速度上限为腰部 0.2 rad/s、手臂 0.8 rad/s。控制周期随环境 `step_dt`，支持 10 Hz 及以上，默认 20 Hz。
- 有界测量反馈补偿仿真驱动挠曲；任务未满足或换解途中冻结积分，热启动使用上一次下发的关节目标，以免静态挠曲推动腰部持续漂移。换解完成且测量位姿稳定后允许补偿更大的重力挠曲。
- 目标稳定超过 1 秒而内部误差仍超过 10 mm 或 0.035 rad 时，在后台尝试其他关节解分支。搜索期间保持关节指令；找到通过采样几何检查的路径后，以五次插值和相同速度上限执行，再恢复末端反馈。直达受阻时可沿先前指令路径退回，或优化一个中间姿态绕行。新目标、重置和模式切换会取消旧恢复任务。
- `control.ik_results` 给出各手基于测量的误差和收敛状态；`control.qp.info` 给出内部求解状态与几何余量。不可达或局部未收敛时维持有界求解，不把低精度结果标记为收敛。

附近目标使用在线 QP。大范围目标的恢复搜索最多 20 秒，路径执行另计；
恢复时允许末端绕行，不保证笛卡尔直线，也不保证另一只手在途中完全静止，但双手终点目标不变。
有限次分支搜索不能证明整个工作空间可达；几何约束是包围盒/采样点近似，不代表完整网格碰撞检查。
当前精度结论来自本地无负载 PhysX 仿真；带载、实机和全工作空间需要另行验证。

2026-10-23 本地回归结果：

| 用例 | 通过数 | 最差稳态位置误差 | 最差稳态姿态误差 |
| --- | --- | --- | --- |
| 常规、随机和带腰目标 | 87/87 | 0.015 mm | 0.0016° |
| 大范围目标（含原 IK 可达点对照） | 44/44 | 0.086 mm | 0.0081° |
| 连续弯腰、转腰、回正 | 15/15 | 0.0043 mm | 0.00044° |
| 10 Hz 单手指令与另一只手保持 | 8/8 | 0.0025 mm | 0.00028° |

误差取最后 20 个控制周期双手测量误差的 P95，再取各用例最大值；它衡量终点精度，
不代表大范围换解过程中的轨迹误差或到达速度。困难点可能需要约 50 秒仿真时间完成恢复。
另有 40 项单元测试通过，ROS 模式切换回归 15/15 通过；30 秒保持的 Z 向波动不超过 0.0042 mm。

本地物理回归（直接调用真实 SDK 命令回调）与独立 ROS 客户端验证脚本如下。
这些测试脚本按仓库规则保留在本地，不随安装包发布：

```bash
PYTHONPATH=src python tests/closed_loop/run_whole_body_qp.py \
  --backends baseline,qp --extended --ticks 160 --out /tmp/x2_qp.json

source /opt/ros/jazzy/setup.bash
/usr/bin/python3 tests/closed_loop/run_qp_ros.py --out-dir /tmp/x2_qp_ros
```

### 控制模式与切换服务

仿真环境统一使用关节位置控制。`ee` 模式在 ROS 桥内用 `maniparena_sim.kinematics` 的求解器
把末端位姿解成关节目标，
`joint` 模式直接写入关节目标，两种模式共用同一套关节驱动。

`ros.arm_control`（`ee` 或 `joint`，默认 `ee`）只决定启动时的模式。运行中用
`std_srvs/srv/SetBool` 服务 `/set_arm_control_mode` 切换（名称可用 `ros.mode_service` 修改）：

- `data: true` 为 `ee`，`data: false` 为 `joint`。
- 服务端保存当前模式。响应的 `message` 始终是生效后的模式（`ee` 或 `joint`），`success: false` 表示切换失败，模式保持不变。
- 请求的模式与当前相同时不做任何改动，直接返回该模式。SetBool 没有只读查询，请求总是被当作期望模式。
- 切换时手臂保持当前关节目标。服务端只订阅当前模式的臂指令话题，切换后请等待约 0.1 s 让订阅重新建立，再发送新模式的指令。
- 切换后 `end_pose` / 关节状态在两种模式下都会发布。

```bash
ros2 service call /set_arm_control_mode std_srvs/srv/SetBool "{data: false}"   # -> message: joint
```



相机启用时发布左右腕和头部压缩 RGB、头部深度及底盘点云；IMU、激光雷达仅在对应传感器可用时发布。
`/hal/chassis/imu` 的线加速度沿用 SDK 的 g 单位（1 g ≈ 9.81 m/s²）。
`/clock` 在 `use_sim_time: true` 时发布。若资产的传感器路径不同，可通过 `robot.camera_paths` 覆盖，
或设置 `enable_cameras: false`。默认物理 120 Hz、控制 20 Hz、渲染调度 60 Hz；这些是仿真时间频率，
实际运行速度取决于硬件和渲染负载。
