# QUANTA_X1 五任务对齐说明

本文记录 `maniparena-sim` 与远程 `cloud_autodata_reference` 数采配置的对齐合同。
它描述的是场景和成功判定，运行时的模型服务地址仍由各自的评测 YAML 提供。本仓库提供遥操作采集和模型评测入口，不包含远程自动数采（gendata）流程。

## 任务清单

| 任务 | 底座 | 远程对应 | 成功条件 |
| --- | --- | --- | --- |
| `move_bottle_to_bin` | 移动 | `move_bottle_to_bin` | 瓶子进入垃圾桶有效 footprint，保持 5 步，双臂回默认位 |
| `move_fruit_to_basket` | 移动 | `move_fruit_to_basket` | 水果进入篮子有效 footprint，保持 5 步，双臂回默认位 |
| `put_bottle_on_woodshelf` | 移动 | `put_bottle_on_woodshelf` | 瓶子在木架坐标框内、保持直立和低速 5 步，双臂回默认位 |
| `sort_blocks` | 固定 | `tabletop_fixed_sort_blocks` | 三个方块分别在对应纸面，随后双臂回默认位 |
| `fruits_to_basket` | 固定 | `tabletop_fixed_fruits_to_basket` | 苹果、香蕉、梨分别进入篮子，面包留在桌面，随后双臂回默认位 |

## 固定水果场景

远程 v11 的 reset 槽位按每回合随机排列四个物体，并对每个物体加入 XY ±0.01 m、yaw ±0.4 rad 的扰动：

```text
(0.04,  0.10, -0.2348)
(0.16,  0.03, -0.2348)  # v11 将原 x=0.20 的远侧槽位移到 x=0.16
(0.04, -0.62, -0.2348)
(0.20, -0.55, -0.2348)
```

场景名义值仍与远程 scene vnext 一致；只有 reset 槽位使用 v11 的第二个坐标。梨 USD 的静态、动态摩擦系数均为 1.5，这是远程 v11 为避免抓取后滑落采用的已确认设置。

固定块场景使用远程名义坐标 `(0.05,-0.16/-0.26/-0.36,-0.213)`，reset 在 `x∈[-0.04,0.06]`、`y∈[-0.46,-0.06]`、yaw ±0.3 rad 中联合采样，物体中心最小间距 0.11 m。

## 移动场景

移动任务沿用远程 ID 配置的坐标和范围：机器人起点 `x∈[-0.7,0.18]`、`y∈[-1.1,1.1]`，瓶子桌面局部范围 `x∈[-0.25,0.25]`、`y∈[-0.05,0.15]`，水果桌面局部范围 `x∈[-0.26,0.26]`、`y∈[-0.12,0.12]`，篮子局部范围 `x∈[-0.08,0.08]`、`y∈[-0.06,0.06]`。木架任务使用相同的 case bank 锚点和瓶子根部高度 0.84 m。

## 启动方式

五个任务都通过 `scripts/collect.py` 和 `scripts/eval.py` 的 `--robot quanta_x1` 参数注册。遥操作命令见 [QUANTA_X1 使用指南](quanta_x1.md)。以下命令连接 Wall-X 服务进行闭环评测：

```bash
source 3rd/isaaclabarena/.venv/bin/activate
export PYTHONPATH="$PWD/src"
python scripts/eval.py --robot quanta_x1 --task move_bottle_to_bin \
  --config configs/eval/quanta_x1_move_bottle_to_bin.yaml --headless
```

其他任务需替换 `--task` 并配置相应的评测 YAML 和任务 instruction。此命令需要可用的 Wall-X 服务；任务成功率由完整策略轨迹评估。
