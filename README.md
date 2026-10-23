# agx-retargeting

[![Version](https://img.shields.io/badge/version-2026.10.22-blue)](CHANGELOG) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

[English](#english) | [中文](#中文)

## English

Convert tracked human hands into robot joint targets. Input and output adapters connect devices to a shared retargeting core.

| Component | Included support |
| --- | --- |
| Input | Quest / WebXR |
| Hand models | Revo2, Wuji Hand; left and right |
| Hardware output | Revo2 through the arm controller software development kit (SDK) |
| Visualization | Robot hand, tracked skeleton, and fitting errors |

### Quickstart

Requires Python 3.10 or later. Run commands from the repository root.

1. Install the dependencies.

   ```bash
   python -m pip install -e '.[webxr,visualization]'
   ```

2. Generate the development certificate.

   ```bash
   python -m agx_retargeting.inputs.webXR --init-certs
   ```

3. Start left Revo2 retargeting with visualization.

   ```bash
   python -m agx_retargeting.inputs.webXR --hand left --viewer
   ```

Connect the browsers:

1. Open `https://<computer-IP>:9443/` in the Quest browser.
2. Accept the development certificate.
3. Enter an extended reality (XR) session with hand tracking.
4. Open `http://127.0.0.1:8080/` on the computer.

| Session | Command |
| --- | --- |
| Right Wuji Hand | `python -m agx_retargeting.inputs.webXR --model wuji_hand --hand right` |
| Both hands, right-hand visualization | `python -m agx_retargeting.inputs.webXR --hand both --viewer --viewer-hand right` |
| All options | `python -m agx_retargeting.inputs.webXR --help` |

USB forwarding uses `adb reverse tcp:9443 tcp:9443`. The Quest address is `https://localhost:9443/`.

The recording directory must exist. `--output <path>.jsonl` records per-frame results.

### Revo2 example

The Revo2 must be connected to its arm controller and enabled. This example uses Linux SocketCAN at 1 Mbit/s.

1. Install the Revo2 SDK dependency.

   ```bash
   python -m pip install -e '.[revo2]'
   ```

2. Set the Controller Area Network (CAN) bitrate.

   ```bash
   sudo ip link set can0 type can bitrate 1000000
   ```

3. Enable the CAN interface.

   ```bash
   sudo ip link set can0 up
   ```

4. Start hardware output.

   ```bash
   python -m agx_retargeting.inputs.webXR --hand left --revo2-left-can can0 --revo2-robot nero --revo2-firmware v120
   ```

The command selects Nero `v120`. The default arrival time is 40 ms. `--revo2-robot` and `--revo2-firmware` select the controller profile.

Right-hand output uses `--hand right --revo2-right-can can0`. Each hand requires a separate CAN interface. Visualization displays joint targets.

### Configuration

Runtime defaults come from [session.py](agx_retargeting/inputs/webXR/session.py). `--session <path>` loads a session YAML. Command-line arguments override session values. Configuration changes take effect after a restart.

Hand geometry, keypoint mapping, coordinates, and motion settings use [revo2_params.yaml](agx_retargeting/config/revo2_params.yaml) or [wuji_hand_params.yaml](agx_retargeting/config/wuji_hand_params.yaml). Residual and solver settings use [optimizer_params.yaml](agx_retargeting/config/optimizer_params.yaml).

| Optimization field | Default | Meaning |
| --- | --- | --- |
| `initializer.position_sigma`, `smooth_weight`, `max_evaluations` | `0.05`, `2.5`, `10` | Bone-end position standard deviation (m), joint-reference weight, initialization evaluation limit |
| `residual.pair_sigma`, `shape_sigma`, `bone_sigma` | `0.0075`, `0.02`, `0.6` | Standard deviations: pinch vectors (m), wrist-to-fingertip positions (m), terminal directions (rad) |
| `residual.pinch.center_distance`, `slope` | `0.04`, `100` | Complementary pinch/shape activation centre (m) and slope (1/m) |
| `residual.motion_sigma`, `motion_weight` | `0.7`, `0.1` | Standard deviation of world position-error change (m/s) and task weight |
| `residual.huber_delta` | `2` | Block Huber threshold after residual normalization |
| `solver.backend` | `proxqp` | Backend for initialization and online solves: `proxqp` or `dogbox` |
| `solver.max_evaluations` | `20` | Online function-evaluation budget |
| `solver.proxqp.initial_damping` | `1.0e-4` | Initial damping of the scaled Gauss–Newton model |
| `solver.proxqp.absolute_tolerance`, `max_iterations` | `1.0e-9`, `100` | Quadratic program (QP): absolute tolerance ceiling and iteration limit |

Smaller residual standard deviations increase error penalties. `residual.wrist_sigma` and `residual.wrist_process_sigma` control wrist priors. `residual.acceleration_sigma` controls the joint motion prior.

`motion.velocity_decay_s` controls the Ornstein–Uhlenbeck (OU) velocity-prior decay time. The hand YAML defaults are `0.02` s for Revo2 and `0.03` s for Wuji Hand.

The session field `output.arrival_time` sets the hardware arrival time in seconds.

### Adapt inputs and outputs

Adapter interfaces use [adapter.py](agx_retargeting/adapter.py). Shared data types use [types.py](agx_retargeting/types.py).

| Import | Content |
| --- | --- |
| [agx_retargeting.core](agx_retargeting/core/__init__.py) | Hand models and retargeting |
| [agx_retargeting.opt](agx_retargeting/opt/__init__.py) | Optimizers |
| `agx_retargeting.config.CONFIG_DIR` | Bundled configuration directory |

| Skill | Provide |
| --- | --- |
| [agx-adapt-input](skills/agx-adapt-input/SKILL.md) | Device SDK, keypoint layout, coordinates, units, and timestamps |
| [agx-adapt-output](skills/agx-adapt-output/SKILL.md) | Unified Robot Description Format (URDF) file, meshes, SDK, and actuator calibration |

```text
Use skills/agx-adapt-input/SKILL.md to adapt <device SDK> to HandInput.
Use skills/agx-adapt-output/SKILL.md to integrate <hand URDF and output SDK>.
```

### Pipeline

```mermaid
flowchart TD
    subgraph INPUT["1 · Input"]
        RAW["Latest sample<br/>WebXR hand tracking"]
        IA["InputAdapter<br/>25 → 21 keypoints · xyzw · metres · timestamps"]
        TRACK{"Tracked?"}
        RESET["Reset history<br/>Wait for tracking recovery"]
        PALM["Palm observation<br/>Singular value decomposition (SVD) plane<br/>Wrist → middle metacarpophalangeal (MCP) joint"]
        LOCAL["Local targets<br/>Wrist centring · robot basis · scale"]
        RAW --> IA --> TRACK
        TRACK -->|yes| PALM --> LOCAL
        TRACK -->|no| RESET
    end

    subgraph INIT["2 · Initialization"]
        MODEL["Hand model<br/>URDF · mimic limits<br/>Palm basis: config / neutral forward kinematics (FK)"]
        HISTORY{"Previous state?"}
        ALIGN["q-method alignment<br/>Reference FK ↔ observed bone endpoints"]
        FIT["Initial joint fit<br/>Bounded bone-end error + reference prior"]
        PREDICT["Motion prediction<br/>OU q, v · source timestamp interval"]
        SEED["Solver seed<br/>Fitted or predicted q + wrist correction eta"]
        MODEL --> ALIGN
        HISTORY -->|first frame / recovery| ALIGN --> FIT --> SEED
        HISTORY -->|continuous tracking| PREDICT --> SEED
    end

    subgraph TASKS["3 · Residuals"]
        SHAPE["Shape<br/>5 wrist-to-fingertip (TIP) position errors"]
        PINCH["Pinch<br/>4 thumb-to-finger vector errors"]
        DIRECTION["Bone direction<br/>5 spherical Log errors on fixed reference tangent planes"]
        GATE["Sigmoid activation<br/>Source pinch distance → a<br/>Pinch a · shape 1-a · thumb shape 1-max(a)"]
        GEOMETRY["Geometry residuals<br/>sqrt_info → block Huber → pinch/shape sqrt weights"]
        WRIST["Wrist SE(3) priors<br/>Observation + temporal correction"]
        JOINT["Joint motion<br/>OU q-v process · initial velocity prior"]
        WORLD["World motion<br/>World TIP position-error change / dt"]
        SHAPE & PINCH & DIRECTION & GATE --> GEOMETRY
    end

    subgraph SOLVE["4 · Optimization"]
        REDUCE["Velocity elimination<br/>Candidate q, eta → conditional bounded v"]
        EVALUATE["Residuals and Jacobians<br/>Pinocchio FK + analytic derivatives"]
        STEP["Default backend<br/>Damped Gauss-Newton + ProxQP box-constrained step"]
        CHECK["Nonlinear cost check<br/>Accept step · adjust damping"]
        REDUCE --> EVALUATE --> STEP --> CHECK
        CHECK -->|iterate| REDUCE
    end

    subgraph OUTPUT["5 · Output"]
        RESULT["Recover state<br/>Joint q · velocity v · wrist pose"]
        STATE["Save history<br/>q, v, eta · world TIP errors"]
        OA["OutputAdapter<br/>Joint names + limits → actuator values"]
        SDK["Revo2 SDK<br/>6 positions in 0–100 inclusive · arrival time in 10 ms units"]
        VIEW["Visualization and recording<br/>Input skeleton · commanded FK · fitting errors"]
        RESULT --> STATE
        RESULT -->|optional hardware| OA --> SDK
        RESULT --> VIEW
    end

    LOCAL --> HISTORY
    LOCAL --> SHAPE & PINCH & DIRECTION & GATE
    PALM --> WRIST & JOINT & WORLD
    MODEL -.->|basis| LOCAL
    MODEL -.->|joint limits| STEP
    SEED --> REDUCE
    GEOMETRY & WRIST & JOINT & WORLD --> EVALUATE
    CHECK -->|tolerance / evaluation budget| RESULT
    STATE -.->|next frame| HISTORY
    STATE -.-> PREDICT & WRIST & JOINT & WORLD
    LOCAL --> VIEW
    SDK -.->|position / motor-status feedback| VIEW
```

### References

| Method | Paper | Official resources |
| --- | --- | --- |
| AKO | [Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation](https://arxiv.org/abs/2506.09384) | [Project](https://mingrui-yu.github.io/retargeting/) · [Code](https://github.com/Mingrui-Yu/retargeting) |
| GeoRT | [Geometric Retargeting: A Principled, Ultrafast Neural Hand Retargeting Algorithm](https://arxiv.org/abs/2503.07541) | [Project](https://zhaohengyin.github.io/geort/) · [Code](https://github.com/facebookresearch/GeoRT) |
| TopoRetarget | [TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation](https://arxiv.org/abs/2606.16272) | [Project](https://toporetarget2026.github.io/TopoRetarget/) |

## 中文

将人手追踪数据转换为机器人关节目标。输入和输出适配器连接统一的重定向核心。

| 组件 | 已支持 |
| --- | --- |
| 输入 | Quest / WebXR |
| 手部模型 | Revo2、Wuji Hand，均支持左手和右手 |
| 硬件输出 | Revo2，通过机械臂控制器的软件开发工具包（SDK）连接 |
| 可视化 | 机器人手部、追踪骨架和拟合误差 |

### 快速开始

Python 版本必须不低于 3.10。命令的执行目录为仓库根目录。

1. 安装依赖。

   ```bash
   python -m pip install -e '.[webxr,visualization]'
   ```

2. 生成开发证书。

   ```bash
   python -m agx_retargeting.inputs.webXR --init-certs
   ```

3. 启动左手 Revo2 重定向和可视化。

   ```bash
   python -m agx_retargeting.inputs.webXR --hand left --viewer
   ```

连接浏览器：

1. 在 Quest 浏览器打开 `https://<电脑IP>:9443/`。
2. 接受开发证书。
3. 进入支持手部追踪的扩展现实（XR）会话。
4. 在电脑浏览器打开 `http://127.0.0.1:8080/`。

| 会话 | 命令 |
| --- | --- |
| Wuji Hand 右手 | `python -m agx_retargeting.inputs.webXR --model wuji_hand --hand right` |
| 双手输入，显示右手 | `python -m agx_retargeting.inputs.webXR --hand both --viewer --viewer-hand right` |
| 全部选项 | `python -m agx_retargeting.inputs.webXR --help` |

USB 转发命令为 `adb reverse tcp:9443 tcp:9443`。Quest 访问地址为 `https://localhost:9443/`。

录制目录必须已经存在。`--output <路径>.jsonl` 保存逐帧结果。

### Revo2 示例

Revo2 必须已经连接机械臂控制器并启用。本示例使用 Linux SocketCAN，通信速率为 1 Mbit/s。

1. 安装 Revo2 SDK 依赖。

   ```bash
   python -m pip install -e '.[revo2]'
   ```

2. 设置控制器局域网（CAN）的通信速率。

   ```bash
   sudo ip link set can0 type can bitrate 1000000
   ```

3. 启用 CAN 接口。

   ```bash
   sudo ip link set can0 up
   ```

4. 启动硬件输出。

   ```bash
   python -m agx_retargeting.inputs.webXR --hand left --revo2-left-can can0 --revo2-robot nero --revo2-firmware v120
   ```

命令选择 Nero `v120`，默认到达时间为 40 ms。`--revo2-robot` 和 `--revo2-firmware` 指定控制器配置。

右手输出使用 `--hand right --revo2-right-can can0`。每只手需要独立的 CAN 接口。可视化页面显示关节目标。

### 配置参数

运行默认配置定义在 [session.py](agx_retargeting/inputs/webXR/session.py)。`--session <路径>` 加载会话 YAML。命令行参数覆盖会话配置。配置修改在程序重启后生效。

手部几何、关键点映射、坐标变换和运动参数位于 [revo2_params.yaml](agx_retargeting/config/revo2_params.yaml) 或 [wuji_hand_params.yaml](agx_retargeting/config/wuji_hand_params.yaml)。残差和求解器参数位于 [optimizer_params.yaml](agx_retargeting/config/optimizer_params.yaml)。

| 优化字段 | 默认值 | 含义 |
| --- | --- | --- |
| `initializer.position_sigma`、`smooth_weight`、`max_evaluations` | `0.05`、`2.5`、`10` | 骨端位置标准差（m）、关节参考权重、初始化评估次数上限 |
| `residual.pair_sigma`、`shape_sigma`、`bone_sigma` | `0.0075`、`0.02`、`0.6` | 标准差：捏合向量（m）、腕部至指尖位置（m）、末节方向（rad） |
| `residual.pinch.center_distance`、`slope` | `0.04`、`100` | 捏合与形状互补激活的中心距离（米）和斜率（1/米） |
| `residual.motion_sigma`、`motion_weight` | `0.7`、`0.1` | 世界坐标中位置误差变化的标准差（m/s）和任务权重 |
| `residual.huber_delta` | `2` | 残差归一化后，作用于残差分组的 Huber 阈值 |
| `solver.backend` | `proxqp` | 初始化与在线求解的后端，可设为 `proxqp` 或 `dogbox` |
| `solver.max_evaluations` | `20` | 在线函数评估次数上限 |
| `solver.proxqp.initial_damping` | `1.0e-4` | 缩放后的 Gauss–Newton 模型的初始阻尼 |
| `solver.proxqp.absolute_tolerance`、`max_iterations` | `1.0e-9`、`100` | 二次规划（QP）的绝对容差上限和迭代次数上限 |

残差标准差越小，对应误差的惩罚越大。`residual.wrist_sigma` 和 `residual.wrist_process_sigma` 控制腕部先验。`residual.acceleration_sigma` 控制关节运动先验。

`motion.velocity_decay_s` 控制 Ornstein–Uhlenbeck（OU）速度先验的衰减时间。手部 YAML 的默认值为 Revo2 `0.02` s、Wuji Hand `0.03` s。

会话字段 `output.arrival_time` 设置硬件到达时间，单位为 s。

### 适配输入与输出

适配器接口位于 [adapter.py](agx_retargeting/adapter.py)。共享数据类型位于 [types.py](agx_retargeting/types.py)。

| 导入路径 | 内容 |
| --- | --- |
| [agx_retargeting.core](agx_retargeting/core/__init__.py) | 手部模型和重定向 |
| [agx_retargeting.opt](agx_retargeting/opt/__init__.py) | 求解器 |
| `agx_retargeting.config.CONFIG_DIR` | 内置配置目录 |

| Skill | 所需资料 |
| --- | --- |
| [agx-adapt-input](skills/agx-adapt-input/SKILL.md) | 设备 SDK、关键点顺序、坐标约定、单位和时间戳 |
| [agx-adapt-output](skills/agx-adapt-output/SKILL.md) | 统一机器人描述格式（URDF）文件、网格文件、SDK 和执行器标定 |

```text
使用 skills/agx-adapt-input/SKILL.md，将 <设备 SDK> 适配为 HandInput。
使用 skills/agx-adapt-output/SKILL.md，接入 <灵巧手 URDF 与输出 SDK>。
```

### 算法流程

```mermaid
flowchart TD
    subgraph INPUT["1 · 输入"]
        RAW["最新采样<br/>WebXR 手部追踪"]
        IA["输入适配器<br/>25 → 21 关键点 · xyzw · 米制坐标 · 时间戳"]
        TRACK{"追踪状态"}
        RESET["重置历史状态<br/>等待恢复追踪"]
        PALM["掌面观测<br/>奇异值分解（SVD）拟合平面<br/>腕部至中指掌指关节（MCP）的方向"]
        LOCAL["局部目标<br/>以腕部为原点 · 坐标旋转 · 尺度变换"]
        RAW --> IA --> TRACK
        TRACK -->|有效| PALM --> LOCAL
        TRACK -->|丢失| RESET
    end

    subgraph INIT["2 · 初始化"]
        MODEL["手部模型<br/>URDF · mimic 关节限位<br/>掌面坐标基：配置旋转或中立姿态正运动学（FK）"]
        HISTORY{"历史状态"}
        ALIGN["q-method 配准<br/>参考 FK 骨端 ↔ 观测骨端"]
        FIT["关节初值拟合<br/>骨端位置误差 + 关节参考先验 · 关节限位"]
        PREDICT["运动预测<br/>OU 的 q、v · 输入时间戳间隔"]
        SEED["求解初值<br/>拟合或预测的 q + 腕部修正 eta"]
        MODEL --> ALIGN
        HISTORY -->|首次输入或追踪恢复| ALIGN --> FIT --> SEED
        HISTORY -->|连续追踪| PREDICT --> SEED
    end

    subgraph TASKS["3 · 残差"]
        SHAPE["形状<br/>5 组腕部至指尖（TIP）的位置误差"]
        PINCH["捏合<br/>4 组拇指至其余指尖的向量误差"]
        DIRECTION["末节方向<br/>5 组固定参考切平面中的球面 Log 误差"]
        GATE["Sigmoid 互补激活<br/>人手捏合距离 → a<br/>捏合 a · 形状 1-a · 拇指形状 1-max(a)"]
        GEOMETRY["几何残差处理<br/>sqrt_info → 分组 Huber → 捏合/形状权重平方根"]
        WRIST["腕部 SE(3) 先验<br/>观测约束 + 腕部修正的时间约束"]
        JOINT["关节运动先验<br/>OU 的 q-v 过程 · 初始速度先验"]
        WORLD["世界坐标中的运动一致性<br/>指尖位置误差的相邻帧差 / dt"]
        SHAPE & PINCH & DIRECTION & GATE --> GEOMETRY
    end

    subgraph SOLVE["4 · 求解"]
        REDUCE["速度消元<br/>候选 q、eta → 满足速度限位的条件最优 v"]
        EVALUATE["残差与 Jacobian<br/>Pinocchio FK + 解析导数"]
        STEP["默认求解器<br/>阻尼 Gauss-Newton + ProxQP 有界增量"]
        CHECK["非线性代价检验<br/>接受更新 · 调整阻尼"]
        REDUCE --> EVALUATE --> STEP --> CHECK
        CHECK -->|继续迭代| REDUCE
    end

    subgraph OUTPUT["5 · 输出"]
        RESULT["恢复完整状态<br/>关节 q · 速度 v · 腕部位姿"]
        STATE["保存历史状态<br/>q、v、eta · 世界坐标中的指尖误差"]
        OA["输出适配器<br/>关节名称与限位 → 执行器指令"]
        SDK["Revo2 SDK<br/>6 通道位置 0 至 100（含） · 到达时间单位 10 ms"]
        VIEW["可视化与记录<br/>输入骨架 · 关节目标的 FK · 拟合误差"]
        RESULT --> STATE
        RESULT -->|可选硬件输出| OA --> SDK
        RESULT --> VIEW
    end

    LOCAL --> HISTORY
    LOCAL --> SHAPE & PINCH & DIRECTION & GATE
    PALM --> WRIST & JOINT & WORLD
    MODEL -.->|坐标基| LOCAL
    MODEL -.->|关节限位| STEP
    SEED --> REDUCE
    GEOMETRY & WRIST & JOINT & WORLD --> EVALUATE
    CHECK -->|满足容差或达到评估次数上限| RESULT
    STATE -.->|下一帧数据| HISTORY
    STATE -.-> PREDICT & WRIST & JOINT & WORLD
    LOCAL --> VIEW
    SDK -.->|位置与电机状态反馈| VIEW
```

### 参考资料

| 方法 | 论文 | 官方资料 |
| --- | --- | --- |
| AKO | [Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation](https://arxiv.org/abs/2506.09384) | [项目页面](https://mingrui-yu.github.io/retargeting/) · [源代码](https://github.com/Mingrui-Yu/retargeting) |
| GeoRT | [Geometric Retargeting: A Principled, Ultrafast Neural Hand Retargeting Algorithm](https://arxiv.org/abs/2503.07541) | [项目页面](https://zhaohengyin.github.io/geort/) · [源代码](https://github.com/facebookresearch/GeoRT) |
| TopoRetarget | [TopoRetarget: Interaction-Preserving Retargeting for Dexterous Manipulation](https://arxiv.org/abs/2606.16272) | [项目页面](https://toporetarget2026.github.io/TopoRetarget/) |
