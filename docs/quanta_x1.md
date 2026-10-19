# QUANTA_X1 使用指南

本文档介绍 QUANTA_X1 的遥操作数据采集和 Wall-X 闭环评测；SDK ROS2 接口见 [QUANTA_X1 SDK ROS2](quanta_x1_sdk.md)。

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

手臂目标位姿数值统一在 `lift_link` 坐标系下表达，升降（lift）运动不会改变手臂指令；采集记录的 19D 动作、Wall-X 评测和 SDK 的末端位姿接口使用同一坐标系语义。EE `PoseStamped.header.frame_id` 统一为 `neutral`。

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

QUANTA_X1 支持使用开源的 [Wall-X](https://github.com/X-Square-Robot/wall-x)（WALL-OSS 系列模型）进行闭环评测。评测入口为 `scripts/eval.py`，通过 WebSocket 连接 Wall-X 推理服务，服务的启动方式参见 Wall-X 仓库 README 中的 “WebSocket Serving”。更多评测任务可参考 [X2Real](https://github.com/X-Square-Robot/x2real)。

## 模型与数据

| 内容 | 链接 |
|------|------|
| QUANTA_X1 Wall-OSS 0.5 checkpoint（20k 步，仅推理文件） | [yunxiao13/quanta-x1-wallx-20k](https://huggingface.co/yunxiao13/quanta-x1-wallx-20k) |
| 训练数据（LeRobot v3.0，1066 条，5 个任务） | [yunxiao13/quanta-x1-demos](https://huggingface.co/datasets/yunxiao13/quanta-x1-demos) |

两个仓库均为公开仓库，下载无需登录。使用 [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub) 的命令行工具 `hf` 下载：

```bash
pip install -U huggingface_hub

hf download yunxiao13/quanta-x1-wallx-20k --local-dir ./quanta-x1-wallx-20k
hf download yunxiao13/quanta-x1-demos --repo-type dataset --local-dir ./quanta-x1-demos
```

checkpoint 约 18 GB，数据集约 5 GB。下载中断后重复执行同一条命令即可续传。无法直接访问 huggingface.co 时，可设置镜像地址 `export HF_ENDPOINT=<镜像地址>` 后再下载。数据集可直接用 LeRobot 加载：

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("yunxiao13/quanta-x1-demos")
```

## 启动推理服务

推理服务使用开源的 [Wall-X](https://github.com/X-Square-Robot/wall-x)。先按其 README 创建环境并安装（Python 3.10、`requirements.txt`、`flash-attn`、LeRobot，最后 `pip install -e .`），然后在该环境中运行本仓库的启动脚本：

```bash
python scripts/wallx_serve_quanta_x1.py \
  --checkpoint-path ./quanta-x1-wallx-20k \
  --port 8000
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint-path` | 必填 | checkpoint 目录 |
| `--port` | 8000 | WebSocket 端口，与评测命令的 `--server-port` 一致 |
| `--host` | 127.0.0.1 | 监听地址；服务没有鉴权，需要远程访问时请自行限制来源 |
| `--cuda-id` | 0 | 使用的 GPU，对应 `CUDA_VISIBLE_DEVICES` |
| `--action-horizon` | 32 | 动作块长度 |

脚本等价于 Wall-X 的 `scripts/run_serving.sh`，固定使用 `--robot-type ex001` 并返回序列化动作；其他参数可写在 `--` 之后，原样传给 Wall-X 的 `launch_serving`。该 checkpoint 的 `head_pos` 槽位对应底盘速度，Wall-X 默认实现在请求带有 `head_pos` 时会回显请求中的状态而不是模型预测，脚本在启动时替换了这一处方法，不修改 Wall-X 源码。

## 运行

推理服务就绪后运行（以 `put_bottle_on_woodshelf` 为例）：

```bash
python scripts/eval.py \
  --robot quanta_x1 --task put_bottle_on_woodshelf \
  --config configs/eval/quanta_x1_put_bottle.yaml \
  --server-address 127.0.0.1 --server-port 8000 \
  --viz kit
```

需要 Kit 视口时使用 `--viz kit`；批量评测无需界面时改用 `--viz none`。

每个任务对应的配置文件：

| `--task` | `--config` | `max_steps` |
|----------|-----------|-------------|
| `sort_blocks` | `configs/eval/quanta_x1_sort_blocks.yaml` | 800 |
| `fruits_to_basket` | `configs/eval/quanta_x1_fruits_to_basket.yaml` | 800 |
| `move_bottle_to_bin` | `configs/eval/quanta_x1_move_bottle_to_bin.yaml` | 3000 |
| `move_fruit_to_basket` | `configs/eval/quanta_x1_move_fruit_to_basket.yaml` | 3000 |
| `put_bottle_on_woodshelf` | `configs/eval/quanta_x1_put_bottle.yaml` | 3000 |

回合超时后环境重置并计为一个回合。`--server-address` / `--server-port` 指向 Wall-X 推理服务。

## 评测设置

评测使用独立的环境构建（`build_quanta_x1_eval_gym_env`），仅作用于 `eval.py`，不影响采集和 SDK。

| 项 | 取值 | 说明 |
|----|------|------|
| 仿真步长 | 物理 240 Hz，环境 30 Hz（decimation 8） | 相机每个环境步渲染一次 |
| 动作块 | `action_horizon=32`、`action_chunk_length=32` | 整个动作块执行后重新推理 |
| 插值 | `interpolation_multiplier=1.5` | 一个动作块对应 48 个环境步 |
| 回合步数上限 | 固定底盘任务 800，移动任务 3000 | 由 `max_steps` 设置 |
| 指令 | 与训练数据中对应任务的文本一致 | 见各配置文件的 `instruction` |
| 手臂坐标系 | `lift_link` 数值语义，`neutral` 消息 frame ID | 手臂位姿数值仍在 `lift_link` 下表达，EE PoseStamped 的 `header.frame_id` 统一为 `neutral`，lift 运动不改变手臂指令 |
| 相机 | 头部 1280×720 标定针孔相机，腕部鱼眼相机 | 与采集、SDK 一致，镜头畸变模型由 `enable_usd_lens_models()` 写入相机 USD；评测中不启用仅输出深度的底盘相机 |
| 物体与布局 | `sort_blocks`、`fruits_to_basket` 使用 `assets/quanta_x1_objects/` 下带 `_2` 后缀的资产 | 场景与桌面臂任务分别构建（`scene_builder.py`） |
| 固定底盘出生位姿 | `sort_blocks`：`pos=(-0.52, -0.26, -0.985)`；`fruits_to_basket`：`pos=(-0.60, -0.26, -0.985)`；head pitch 均为 0.35 | 见 `task/quanta_tabletop_layouts.py` |

### Lift 指令

训练数据中各任务的 lift 指令全程恒定：`sort_blocks` 为 0.455 m，其余任务为 0.379 m。评测时 lift 不由模型控制，`eval.py` 按任务固定该值，可用 `policy_config.lift_hold` 覆盖。

### 字段映射

训练数据（LeRobot v3.0，26 维，30 fps）状态与动作的末尾 6 维按 `[底盘速度 (vx, vy, wz), lift, 头部 (yaw, pitch)]` 排列，与推理服务按字段名组织的顺序不同。评测客户端（`wire_layout: dataset`，默认）按训练数据的顺序填写请求中的 `state`，并从响应中还原对应的指令：

| 训练数据中的量 | 请求 `state` 字段 | 响应字段 |
|----------------|-------------------|----------|
| 底盘速度 vx、vy | `head_pos` | `head_pos` |
| 底盘角速度 wz | `lift` | `lift` |
| lift、头部 yaw、pitch | `velocity_decomposed` | `velocity_decomposed_odom` |

该映射要求推理服务将 `head_pos` 槽位按模型预测值返回，`scripts/wallx_serve_quanta_x1.py` 已满足这一要求。

### 录制

`enable_recording: true` 时，每个回合（包括成功回合）单独导出一个 hdf5 文件，以及头部、左腕、右腕三路相机的 mp4。

### 评测示例

Wall-X 闭环评测中的头部相机视角（加速播放）：

`sort_blocks`：

![quanta_x1 Wall-X eval sort_blocks](../media/quanta_x1_wallx_eval_sort_blocks.gif)

`put_bottle_on_woodshelf`：

![quanta_x1 Wall-X eval put_bottle_on_woodshelf](../media/quanta_x1_wallx_eval_put_bottle_on_woodshelf.gif)
