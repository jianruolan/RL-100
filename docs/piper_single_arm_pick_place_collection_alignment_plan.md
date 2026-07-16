# Piper 单臂 Soft-Block Contact 示教数据采集与 RL-100 数据对齐方案

## 1. 目标与结论

目标是在 NVIDIA Jetson Thor 上打通下面这条最小闭环：

```text
Piper 单臂 Soft-Block Contact 人工示教 + 腕部 RealSense RGB-D
  -> 时间戳化原始 episode
  -> 时间对齐、标定变换、点云裁剪与降采样
  -> RL-100 Zarr
  -> Piper Soft-Block Contact 专用 task config
  -> RL-100 3D BC 训练 smoke test
```

本方案的核心决策是：

1. **对齐 RL-100 的数据容器和训练接口，不伪造 Adroit Hand 的维度。** 现有 `adroit_door.yaml` 是仿真 Adroit Hand 契约：`state=24`、`action=28`、`point_cloud=512x3`。Piper 单臂建议新建 `piper_soft_block_contact` 契约：`state=7`、`action=7`、`point_cloud=512x3`，不应将 Piper 数据补零到 24/28 维。
2. **保留夹爪维度，但将其作为被动/冻结通道。** 数据采集和训练仍保留 6 关节 + 夹爪这 7 维接口；但由于末端夹爪当前不能开合，`action_t` 的最后一维在执行时不参与动态控制，可以固定为常数占位、默认开度或记录为被动观测值。这样既能和后续训练接口对齐，也不会把不可控通道错误解释成有效操控。
3. **采集时保存可重处理的原始 RGB-D，点云离线生成。** 不要只保存 512 个点，否则修改手眼标定、工作区裁剪或深度过滤后必须重新采集。
4. **首轮只验证 BC 和离线采样，不直接启用仿真 `AdroitRunner` 或真机 online RL。** 真机 rollout runner、安全限位和成功判定是后续独立阶段。

---

## 2. 项目现状与可复用部分

### 2.1 RL-100 现有数据契约

`RL-100/rl_100/dataset/adroit_dataset.py` 当前读取：

```text
data/
  img
  point_cloud
  state
  action
  next_img
  next_point_cloud
  next_state
  next_action
  reward
  return
  done
  timeout
meta/
  episode_ends
```

`RL-100/rl_100/config/task/adroit_door.yaml` 当前的模型输入形状为：

```text
image:       3 x 84 x 84
point_cloud: 512 x 3
agent_pos:   24
action:      28
```

Piper 版保持相同的 Zarr 键名和 `obs/next_obs/action` 数据流，但将状态和动作维度改为 7。

### 2.2 现有 RealSense 代码的限制

`tools/teleop_off2off_data/realsense.py` 可复用深度投影和 FPS 点云降采样思路，但不能直接用于本任务，因为它：

- 只 `enable_stream(depth)`，没有启用 color stream；
- 创建了 `rs.align(rs.stream.color)` 但没有调用 `align.process(frames)`；
- 点云裁剪边界和 `X_root_camera` 是另一机器人/相机的硬编码值；
- 原始输出没有 color image、实际流 profile、帧号和主机单调时间戳。

因此需新建 Piper 专用采集器，不应在旧文件中直接替换标定常量。

### 2.3 Piper SDK 可复用接口

同级目录 `../01-Piper/piper_sdk` 的 `C_PiperInterface_V2` 可提供：

```python
piper.ConnectPort()
piper.GetArmJointMsgs().joint_state
piper.GetArmGripperMsgs().gripper_state
piper.GetArmEndPoseMsgs().end_pose
piper.GetFK("feedback")
```

实际字段与 SDK 原始单位：

- `joint_1 ... joint_6`：`0.001 degree`；
- `grippers_angle`：`0.001 mm`；
- 末端 `X/Y/Z`：`0.001 mm`；
- 末端 `RX/RY/RZ`：`0.001 degree`。

写入 raw episode 时建议同时保存原始整数和 SI 制浮点数；进入训练 Zarr 时统一为 `rad` / `m` / `float32`。

---

## 3. 部分 A：示教数据采集

### 3.1 Soft-Block Contact 任务定义

首批数据建议固定 Piper base、工作台和接触区域，使用一个柔性的、小尺寸方块作为接触目标，先把任务简化成“末端碰到目标物体并形成可见接触/轻微压缩”。一条成功 episode 定义为：

1. Piper 从固定 home pose 出发，末端保持当前夹爪姿态不变。
2. 从起始区接近柔性小方块，完成末端对齐。
3. 末端与方块发生接触，并保持轻压或稳定贴触 0.3-1 s。
4. 机械臂撤离，结束 episode。

这里不再要求抓取、抬升、搬运、释放或放置；成功标志改为以下任一条满足：

```text
末端接触到方块且视觉上能看到轻微形变
末端接触到方块且方块位置发生可辨识的小幅位移
人工按键/离线标注确认接触成立
```

需要记录失败原因，但首批 BC 数据只使用成功轨迹：

```text
miss_object / overshoot / excessive_deformation / object_displaced /
collision / joint_limit / camera_invalid / operator_abort / timeout
```

成功判定优先级：视觉接触/形变 > 物体位移 > 人工标注。首轮可人工标记成功，但应保存 episode 结束前后的 RGB-D 作为复核证据，并明确“只擦边未形成接触”不算成功。

数据难度递进建议：

```text
P0: 固定方块位置 + 固定接触方向
P1: 方块在小范围内随机，接触方向固定
P2: 方块位置和朝向在安全工作区内随机
P3: 增加方块材质、背景和光照变化
```

### 3.2 状态与动作定义

首版建议保留 7 维接口，其中最后一维夹爪通道在采集和训练里保留，但执行语义冻结：

```text
state_t = [q1, q2, q3, q4, q5, q6, gripper_width]
          q: rad, gripper_width: m

action_t = [q1_target, q2_target, q3_target,
            q4_target, q5_target, q6_target, gripper_target]
           q_target: rad, gripper_target: m
```

如果当前末端夹爪不能开合，则：

- `state_t` 里仍然记录夹爪反馈/开度，作为观测维度保留；
- `action_t` 最后一维写入固定占位值、默认开度或冻结值；
- 训练时仍保留这一维，避免后续接口再改；
- 真正控制含义只落在前 6 个关节维度上。

对手拖/零力示教，没有独立 command stream 时：

```text
action_t := state at t + 1 transition
```

也就是用下一个 10 Hz 对齐时刻的关节位置作为绝对目标。这一时序定义必须保持到真机部署：policy 预测后续目标，控制端对目标做限速和插值。

原始数据额外保存，但第一版不作为模型输入：

- 末端位姿 `ee_xyz + ee_quat`；
- SDK/CAN 反馈频率、机械臂状态和错误码；
- 夹爪 effort / 开度原始值；
- 数据采集模式、操作员、场景版本、物体 ID/初始位姿和接触区域 ID/位姿。

### 3.3 采样频率与线程结构

建议起步频率：

| 数据/环路 | 频率 | 说明 |
| --- | ---: | --- |
| Piper SDK 状态 | 50-100 Hz | 按 SDK 反馈实际 `Hz` 确定 |
| RealSense RGB-D | 30 Hz | RGB/depth 同 profile，depth align to color |
| 原始日志写入 | 各源原始频率 | 不在采集环内现场下采样点云 |
| RL-100 transition | 10 Hz | 与当前 Adroit runner `fps=10` 保持一致 |

进程结构建议：

```text
RobotReader thread  -> timestamped robot ring buffer --+
                                                     +-> EpisodeWriter
CameraReader thread -> timestamped RGB-D ring buffer --+
UI/Event thread     -> start/success/failure/abort ----+
```

关键规则：

- 每个数据源同时记录设备时间戳、主机 `time.monotonic_ns()` 接收时间和序号。
- 以主机单调时间为跨设备对齐基准，不用 `time.time()` 或文件写入时间做同步。
- 采集线程不执行 FPS 点云降采样或视觉模型，防止丢帧。
- 任一源超时、相机帧停滞、时间戳逆序或 SDK 反馈异常时，当前 episode 标记 invalid，不静默补值。

### 3.4 RealSense 腕部相机实施

#### 硬件与驱动验证

在 Thor 上先单独打通相机，不与 Piper 进程同时调试：

```bash
lsusb -t
rs-enumerate-devices
```

验收点：

- 相机连在 USB 3.x，不是 480M USB 2.0 链路；
- 记录固定 serial number，程序按 serial 打开；
- 连续 10 分钟采集 RGB+depth 无掉线、无帧号回退；
- 记录 Thor 实际安装的 `librealsense` 和 `pyrealsense2` 版本，Python binding 必须与本机 native library 匹配；
- 将高带宽配置先稳定在 `640x480 @ 30 Hz`，再考虑提高分辨率。

#### 推荐流配置

```python
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
align = rs.align(rs.stream.color)

frames = pipeline.wait_for_frames()
aligned = align.process(frames)
color_frame = aligned.get_color_frame()
depth_frame = aligned.get_depth_frame()
```

每帧必须保存：

```text
color_bgr uint8 HxWx3
depth_z16 uint16 HxW
depth_scale
color intrinsics: fx fy cx cy coeffs model
device_timestamp_ms
host_monotonic_ns
color_frame_number / depth_frame_number
camera_serial
```

相机启动后先 warm-up 2-5 s；光照稳定时可锁定 exposure/white balance，并把实际参数写入 episode metadata。

#### 腕部手眼标定

腕部相机为 eye-in-hand，需求：

```text
T_ee_camera       # 相机坐标到 Piper 末端坐标的固定外参
T_base_ee(q_t)    # 由 Piper FK/反馈得到
T_base_camera(t) = T_base_ee(q_t) @ T_ee_camera
```

标定流程：

1. 将 AprilTag/ChArUco 标定板固定在 Piper base 可见区域。
2. 采集至少 20-30 个位姿，覆盖不同距离和三轴旋转，避免只有平移或共面位姿。
3. 求解 `T_ee_camera`，保存带 serial number、镜头 profile、日期和 RMS/reprojection error 的 YAML。
4. 使用未参与求解的位姿验证：同一固定物体转到 base frame 后不应随手腕运动明显漂移。

相机支架、夹爪或镜头 profile 变动后，原标定必须失效并重做。

### 3.5 Raw episode 建议格式

建议每条 episode 一个 HDF5，一旦关闭后不再修改：

```text
raw/piper_soft_block_contact/<session_id>/
  session.yaml
  calibration/
    wrist_<serial>_intrinsics.yaml
    T_ee_camera.yaml
  episodes/
    episode_000001.h5
    episode_000002.h5
```

```text
episode_xxxxxx.h5
  /robot/host_time_ns             int64 [Nr]
  /robot/device_or_sdk_time       ...   [Nr]
  /robot/joint_raw                int32 [Nr, 6]
  /robot/joint_rad                float32 [Nr, 6]
  /robot/gripper_raw              int32 [Nr, 1]
  /robot/gripper_m                float32 [Nr, 1]
  /robot/ee_pose                  float32 [Nr, 7]
  /robot/status                   ...

  /camera/host_time_ns            int64 [Nc]
  /camera/device_time_ms          float64 [Nc]
  /camera/frame_number            int64 [Nc]
  /camera/color_bgr               uint8 [Nc, H, W, 3]
  /camera/depth_z16               uint16 [Nc, H, W]
  /camera/depth_scale             float32 [Nc]

  /events/start_time_ns
  /events/end_time_ns
  /events/approach_time_ns        # 可选，人工按键或离线标注
  /events/contact_time_ns         # 可选，人工按键或离线标注
  /events/contact_end_time_ns     # 可选，人工按键或离线标注
  /labels/success                 bool
  /labels/valid                   bool
  /labels/failure_reason          string
  /attrs/schema_version
  /attrs/camera_serial
  /attrs/calibration_id
  /attrs/operator
  /attrs/task_setup_id
  /attrs/object_id
  /attrs/object_initial_pose
  /attrs/contact_zone_id
  /attrs/reset_batch_id
```

采集时先写 `.h5.tmp`，完整 flush/close 后原子改名为 `.h5`，避免异常退出留下表面上可用的半文件。

### 3.6 采集量与现场 QA

建议三阶段：

| 阶段 | 数量 | 目标 |
| --- | ---: | --- |
| P0 工程验证 | 5-10 条 | 固定起点/接触点，只验证同步、格式、标定和 Zarr 转换 |
| P1 BC smoke dataset | 30-50 条成功轨迹 | 小范围随机起点，验证小集过拟合和完整训练入口 |
| P2 首个可评估版 | 100-200 条成功轨迹 | 增加方块位置、接触方向、速度和操作风格变化 |

每采集 10 条立即运行 QA，不等全部采完再检查：

- RGB 清晰，柔性方块和接触区域在关键阶段可见；
- depth 空洞率、有效距离和深度尺度正常；
- 机器人时间戳、相机时间戳严格单调；
- robot-camera 最近帧时间差 P95 达标；
- 没有连续重复帧或大段掉帧；
- episode 开始前和结束后都有完整数据；
- 夹爪维度在原始数据和 Zarr 中都保留，但执行时不要求真实开合。
- 成功 episode 的最后帧中末端已接触方块，且满足视觉接触、轻微形变或人工确认之一。

---

## 4. 部分 B：数据对齐与后处理

### 4.1 处理流水线

```text
raw HDF5
  -> schema/integrity validation
  -> choose 10 Hz transition timeline
  -> robot interpolation + nearest RGB-D association
  -> depth filtering and deprojection
  -> eye-in-hand transform to Piper base/task frame
  -> workspace/table/robot crop
  -> 512-point FPS sampling
  -> state/action construction
  -> next_* / reward / done / timeout / return
  -> RL-100 Zarr
  -> Zarr validator + visualization + loader smoke test
```

后处理不覆盖 raw 文件；每次输出一个带版本的新 Zarr，并在 root attrs 中写入源 episode 清单、Git commit、标定 ID 和处理参数哈希。

### 4.2 时间对齐规则

对每个 episode 建立 10 Hz 时间轴：

```text
t_k = episode_start + k * 0.1 s
```

对齐策略：

- 关节角在相邻 robot sample 之间做线性插值；
- 末端旋转仅用于标定/调试时使用 quaternion SLERP；
- 夹爪通道保留 7 维语义；若采集期间不可控或只记录被动开度，则用 zero-order hold / 常数占位，不把它当作真实控制插值；
- RGB-D 取距 `t_k` 最近的同步帧，不在图像上做时间插值；
- 起步阈值建议：相机时间差 `<=20 ms`，关节插值两侧样本距离 `<=20 ms`。超限 transition 删除；如连续超限则整条 episode invalid。

阈值最终应由 P0 实测时间差分布确定，并在报告中输出 mean/P50/P95/max，不只输出“对齐成功”。

### 4.3 图像与点云后处理

#### RGB

- raw 保留 BGR/RGB 约定和原始分辨率；
- Zarr `img` 缩放到 84x84，使用 `uint8`；
- 转换时明确 BGR -> RGB；
- 不要在转换脚本中混用 HWC/CHW。

现有 Adroit 生成器会在写 Zarr 前将 CHW 转成 HWC，但 task `shape_meta` 声明为 CHW。由于当前 3D 路线主要使用点云+状态，P0 实施前应将你已在 Thor 上跑通的 `adroit_door_expert.zarr` 作为 golden sample，打印它的实际 image layout，然后在 Piper dataset/loader 中只保留一次显式 transpose。

#### 点云

1. 用当前 color profile 对应的 intrinsics 将 aligned depth 反投影到 camera frame，深度单位转为 m。
2. 使用对齐后的 `q_t` 计算 `T_base_camera(t)`，将点云变换到 Piper base frame。
3. 先按有效深度范围过滤，再按 base/task frame 中的固定 3D ROI 裁剪。
4. 如有需要，利用机器人模型去除夹爪/手腕自身点；首轮可先保留，但必须视化检查它是否遮挡柔性方块。
5. 体素降采样后用 FPS 生成固定 `512x3 float32`。
6. ROI 中有效点少于 512 时不建议简单重复单一点：应标记该帧 invalid；调试期如需 padding，必须同时输出 invalid/padding 比例。

每批数据随机输出至少 20 帧 base-frame 点云叠加可视化，检查方块、接触区域、桌面和末端的尺度/方向。ROI 应覆盖起始区与接触区域的完整联合工作空间，不能只围绕固定接触点裁剪。

### 4.4 RL-100 Zarr 目标结构

建议输出：

```text
RL-100/data/piper_soft_block_contact_v001.zarr
```

| 键 | 形状 | dtype | 语义 |
| --- | --- | --- | --- |
| `data/img` | `[N,84,84,3]` 或 golden sample 确认的 layout | uint8 | `obs_t` 腕部 RGB |
| `data/point_cloud` | `[N,512,3]` | float32 | `obs_t` base-frame XYZ, m |
| `data/state` | `[N,7]` | float32 | `q_t` rad + gripper m / frozen gripper channel |
| `data/action` | `[N,7]` | float32 | 可下发的下一绝对关节目标 + 冻结夹爪通道 |
| `data/next_img` | 同 `img` | uint8 | episode 内 `obs_(t+1)` |
| `data/next_point_cloud` | `[N,512,3]` | float32 | episode 内下一 observation |
| `data/next_state` | `[N,7]` | float32 | episode 内下一 state |
| `data/next_action` | `[N,7]` | float32 | episode 内下一 action |
| `data/reward` | `[N,1]` | float32 | 首轮成功终止帧为 1，其余为 0 |
| `data/done` | `[N,1]` | bool | 每条 episode 最后一帧为 True |
| `data/timeout` | `[N,1]` | bool | 因时间上限结束时 True |
| `data/return` | `[N,1]` | float32 | 按 episode 倒序计算，`gamma=0.99` |
| `meta/episode_ends` | `[E]` | int64 | 累计 exclusive end index |

终止帧的 `next_*` 使用本帧复制，绝对不能跨 episode 取下一帧。

虽然 `timeout` 另存，首版转换建议所有 episode 终止点都设 `done=True`，因为当前 `AdroitDataset` 的 batch `not_done` 只由 `done` 生成。`return` 计算仍以 `done | timeout` 作为边界。

### 4.5 reward 与数据筛选

首轮只测 BC，reward 保持简单：

```text
success terminal: 1.0
other step:       0.0
```

不建议立即复制现有 `data_prepare.py` 中的轨迹长度/动作平滑惩罚，因为 Piper 的 7D 动作单位和 Adroit 不同，未归一化的 L2 penalty 会把关节和夹爪量纲混合。进入 offline RL 前再设计基于接触成功、接触持续时间、目标位姿偏差、压缩量和安全事件的 reward。

数据集划分应按 `session_id` 或 `reset_batch_id` 分组，不要将同一次物体/目标摆放下连续采集的相邻 episode 拆到 train 和 validation 两侧。P2 建议额外留出未见过的起点-目标组合用于测试空间泛化。

### 4.6 新 task 契约

建议新建：

```text
RL-100/rl_100/config/task/piper_soft_block_contact.yaml
```

核心形状：

```yaml
name: piper_soft_block_contact
task_name: piper_soft_block_contact

shape_meta:
  obs:
    image:
      shape: [3, 84, 84]
      type: rgb
    point_cloud:
      shape: [512, 3]
      type: point_cloud
    agent_pos:
      shape: [7]
      type: low_dim
  action:
    shape: [7]

env_runner: null

dataset:
  _target_: rl_100.dataset.adroit_dataset.AdroitDataset
  zarr_path: data/piper_soft_block_contact_v001.zarr
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  seed: 42
  val_ratio: 0.1
```

P0 可先复用 `AdroitDataset`，因为它本身不写死 24/28 维；但建议在正式实施时新建轻量 `PiperSoftBlockContactDataset`，用于：

- 明确 image layout 转换；
- 对 7D 状态/动作做 assert；
- 检查单位、NaN/Inf 和 episode 边界；
- 不将真机 Piper 语义继续挂在 `AdroitDataset` 名下。

`env_runner: null` 用于首轮纯离线 BC，避免训练过程误调用仿真 Adroit Hand 评估。后续完成真机推理和安全控制后，再实现 `PiperSoftBlockContactRunner`。

### 4.7 转换后验收

转换脚本必须输出一份机器可读 JSON 和一份人可读 Markdown 报告，至少包含：

```text
episode count / transition count
success / invalid / failure count
episode length min / mean / P95 / max
camera-robot skew mean / P95 / max
RGB duplicate/drop count
depth valid ratio
point count before/after crop
state/action per-dimension min/max/mean/std
NaN/Inf count
terminal and episode_ends consistency
source episode manifest
```

强制检查：

```text
N == episode_ends[-1]
all data arrays have first dimension N
episode_ends strictly increasing
each episode has exactly one terminal at its last index
no next_* crosses an episode boundary
point_cloud.shape == (N, 512, 3)
state.shape == action.shape == (N, 7)
all training float arrays are finite
```

---

## 5. 训练流程测试

### 5.1 三级 smoke test

#### Level 1：Dataset loader

直接实例化 dataset，检查：

- `len(dataset) > 0`；
- `get_shape_info()` 与 `piper_soft_block_contact.yaml` 一致；
- 一个 batch 的 `obs/next_obs/action` 形状正确；
- normalizer 每维有限且范围合理。

#### Level 2：小集过拟合

取 3-5 条 episode，固定 seed，运行几百到几千个 gradient step。验收标准是 train loss 和 train action MSE 显著下降，并能在固定 observation 上重复输出接近标注的 7D action。

#### Level 3：完整 BC

从现有 3D recipe 复制一份 Piper 专用脚本，而不直接修改通用 `train_policy.sh`。起步参数保持与已在 Thor 上验证过的 Adroit 3D BC 经验接近：

```text
task=piper_soft_block_contact
policy=RL1003D
only_bc=True
offline=True
n_obs_steps=3
n_action_steps=1
horizon=3
env_runner=null
wandb offline
```

不建议第一次就运行现有 recipe 中的全部 lr / rollout_length / clip_std_max 多重循环。先固定一组参数打通单次 BC，确认数据维度、checkpoint 和离线 action MSE 后再做 sweep。

### 5.2 首轮不包含的范围

下列内容不应与首轮数据对齐同时开发：

- 真机 policy rollout 和自动下发；
- offline RL reward shaping；
- online RL；
- 多相机融合；
- 相对末端 action/IK 控制；
- 失败恢复数据的混合训练。

这些内容应在“P0/P1 数据 -> Zarr -> BC”闭环通过后分阶段加入。

---

## 6. 建议代码交付物

实施时建议增加：

```text
tools/piper_soft_block_contact/
  collect_episode.py              # Piper + RealSense 多线程原始采集
  piper_reader.py                 # SDK 反馈、单位转换、状态检查
  realsense_reader.py             # RGB-D、align、serial、timestamp
  raw_schema.py                   # HDF5 schema/version
  align_and_build_zarr.py         # 时间对齐与 Zarr 生成
  validate_raw.py
  validate_zarr.py
  visualize_episode.py
  configs/
    collect_piper_soft_block_contact.yaml
    build_piper_soft_block_contact_zarr.yaml

RL-100/rl_100/dataset/
  piper_soft_block_contact_dataset.py

RL-100/rl_100/config/task/
  piper_soft_block_contact.yaml

scripts/PiperSoftBlockContact/
  train_bc_smoke.sh
```

建议不直接扩展当前 `tools/teleop_off2off_data/data_prepare.py`：它的 raw loader 假设了另一种 `demo_*.npy` 字段和硬编码相机外参，而且当前 Zarr writer 不写 `img/next_img`。可复用它的 `next_*`、`return`、`episode_ends` 和 source manifest 思路，但 Piper Soft-Block Contact 应使用独立 schema 和转换器。

---

## 7. 里程碑与验收门槛

### M0：相机单独打通

- [ ] Thor 可稳定读取腕部 RGB+depth 30 Hz。
- [ ] depth 已 align to color，并保存实际 intrinsics/depth scale。
- [ ] 10 分钟无掉线，帧号/时间戳单调。

### M1：标定和联合 raw 采集

- [ ] 完成 `T_ee_camera` 标定及独立位姿验证。
- [ ] Piper 与 RGB-D 在同一 episode 中分别以原始频率保存。
- [ ] 用急停/中止测试验证 `.tmp` 文件和 invalid episode 处理。

### M2：P0 Zarr

- [ ] 5-10 条 episode 转成 `piper_soft_block_contact_v001.zarr`。
- [ ] 时间差统计、数组形状和 episode 边界全部通过。
- [ ] 随机回放 RGB/点云/关节轨迹能相互对应。
- [ ] RL-100 dataset loader 可读取一个 batch。

### M3：P1 BC smoke test

- [ ] 30-50 条成功轨迹通过 QA。
- [ ] 3-5 条小集可过拟合，train action MSE 明显下降。
- [ ] 完整 BC 训练可产生 checkpoint，不调用仿真 `AdroitRunner`。
- [ ] 固定验证集的 action MSE 和逐维误差已记录。

### M4：进入真机评估的前置条件

- [ ] 独立 `PiperSoftBlockContactRunner` 和 policy-to-SDK 单位转换通过静态测试。
- [ ] 关节限位、夹爪冻结通道检查、每步变化限制、速度限制、watchdog 和独立急停可用。
- [ ] 先使用不接触物体和桌面的悬空轨迹验证动作方向。
- [ ] 用低速、单步或人工确认方式逐步开放闭环。

---

## 8. 实施顺序建议

```text
1. 在 Thor 上读取已跑通的 Adroit 基准 Zarr，固化 golden schema（保留原有文件路径作为参考）
2. 单独打通腕部 RealSense RGB-D
3. 完成手眼标定和点云 base-frame 可视化
4. 实现 Piper + RealSense raw episode 采集
5. 采 5-10 条 P0 数据，完成对齐、Zarr 和 validator
6. 新建 piper_soft_block_contact task/dataset，跑 loader 和小集过拟合
7. 采 30-50 条 P1 成功数据，跑完整 BC smoke test
8. 根据离线误差和数据可视化决定是否扩到 100-200 条
9. 最后再开发真机 runner 和受限速的 policy rollout
```

这个顺序优先消除三个最高风险：RealSense 在 ARM 主机上的稳定性、eye-in-hand 点云外参，以及 Piper 7D 契约与现有 Adroit 24/28D 契约的混淆。
