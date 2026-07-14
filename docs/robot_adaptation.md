# Robot Adaptation Checklist

本文档整理 RL-100 接入不同机器人本体时的潜在适配点。RL-100 的训练主流程相对统一，但真实换机器人时，必须保证新机器人的控制接口、观测格式、数据 schema、环境 runner、任务配置、奖励和安全约束都与项目训练代码的约定一致。

核心原则：

```text
new robot hardware
  -> robot env / wrapper
  -> observation + action contract
  -> zarr dataset schema
  -> task yaml shape_meta
  -> env_runner evaluation / rollout
  -> train.py / train_real.py common pipeline
```

## 1. 控制接口适配

相关位置：

```text
RL-100/rl_100/env/
tools/teleop_off2off_data/
```

已有参考：

```text
RL-100/rl_100/env/franka/
RL-100/rl_100/env/ur5/
RL-100/rl_100/env/ur51/
RL-100/rl_100/env/r1/
tools/teleop_off2off_data/franka_wrapper.py
tools/teleop_off2off_data/xarm_wrapper.py
tools/teleop_off2off_data/robotiq_wrapper.py
```

需要明确和实现：

| 适配项 | 说明 |
| --- | --- |
| 机器人 SDK / driver | 如何连接机器人、读状态、发控制命令 |
| 控制模式 | joint position、joint velocity、Cartesian delta pose、Cartesian impedance、gripper command 等 |
| 动作单位 | 米、弧度、关节角、归一化动作，必须和训练数据一致 |
| 动作维度 | 不同机器人/夹爪/灵巧手维度不同，例如 MetaWorld 4 维、Adroit 28 维 |
| 控制频率 | `fps`、底层控制周期、`n_action_steps` 要匹配 |
| action clipping | 限制最大位移、旋转、关节速度、夹爪开合量 |
| gripper / hand | 二指夹爪、Robotiq、Allegro、DexHand 等要单独封装 |
| reset 流程 | 每个 episode 如何回初始位姿，是否需要人工摆放物体 |
| watchdog | 控制循环卡住、通信断开或动作异常时立即停机 |

建议输出一个机器人环境封装，至少支持：

```python
obs = env.reset()
obs, reward, done, info = env.step(action)
```

## 2. Observation 观测适配

项目里的 policy 和 dataset 默认围绕这些 key 工作：

```python
obs = {
    "point_cloud": ...,
    "agent_pos": ...,
    "image": ...
}
```

新机器人需要保证在线环境和离线数据都能提供相同结构。

| 观测项 | 适配内容 |
| --- | --- |
| `agent_pos` | 机器人 proprioception，例如关节角、关节速度、末端位姿、夹爪状态、手指关节 |
| `point_cloud` | 点云来源、点数、坐标系、是否裁剪、是否带颜色 |
| `image` | RGB 图像尺寸、通道顺序、resize/crop、归一化方式 |
| `next_obs` | 离线 zarr 中要有下一帧观测，用于 critic/IQL/dynamics |
| 历史观测 | `n_obs_steps` 决定输入几帧历史 |
| shape 配置 | `shape_meta` 必须和真实 tensor shape 对齐 |

task yaml 中通常要修改：

```yaml
shape_meta:
  obs:
    image:
      shape: [3, 84, 84]
      type: rgb
    point_cloud:
      shape: [512, 3]
      type: point_cloud
    agent_pos:
      shape: [24]
      type: low_dim
  action:
    shape: [28]
```

最常变的是：

```text
shape_meta.obs.agent_pos.shape
shape_meta.obs.point_cloud.shape
shape_meta.obs.image.shape
shape_meta.action.shape
```

## 3. 相机和坐标系适配

真实机器人尤其需要单独校准。

相关参考：

```text
tools/teleop_off2off_data/realsense.py
RL-100/rl_100/env/franka/realsense.py
RL-100/rl_100/env/ur5/realsense.py
RL-100/rl_100/env/flipping/realsense.py
```

需要适配：

| 适配项 | 说明 |
| --- | --- |
| 相机内参 | depth 转 point cloud 的 `fx/fy/cx/cy` |
| 相机外参 | camera frame 到 robot base/world frame |
| 点云裁剪区域 | 只保留操作区，过滤桌面、背景和机器人无关部分 |
| 点云下采样 | 项目常用 512 或 1024 点 |
| RGB/depth 对齐 | 确认彩色图和深度图是否对齐 |
| 坐标系方向 | x/y/z 方向、右手系/左手系、单位米/毫米 |
| 时间同步 | robot state、RGB、depth 的 timestamp 是否对齐 |
| 延迟补偿 | 相机延迟和动作执行延迟是否需要补偿 |

上线前要检查：同一个物体在点云坐标下的位置是否和机器人 base 坐标下的位置一致，否则 policy 学到的空间关系会错位。

## 4. Action 空间适配

这是换本体时最容易出问题的部分。

需要定义清楚：

| 项 | 说明 |
| --- | --- |
| action 类型 | 绝对目标、增量目标、关节命令、末端位姿命令 |
| action 坐标系 | base frame、tool frame、camera frame |
| action 维度 | 例如 `[dx, dy, dz, droll, dpitch, dyaw, gripper]` |
| action scale | 模型输出对应多少米、多少弧度、多少关节变化 |
| action normalization | `action_norm=True` 时 normalizer 会根据数据统计缩放动作 |
| action chunk | 一次预测一个动作还是 `n_action_steps` 个动作 |
| latency | 是否需要设置 `n_latency_steps` 或在 wrapper 中补偿 |
| 安全裁剪 | workspace bounds、关节限位、速度限位 |

相关配置：

```yaml
horizon: 3
n_obs_steps: 3
n_action_steps: 1
n_latency_steps: 0
chunk_as_single_action: true
action_norm: true
shape_meta:
  action:
    shape: [...]
```

离线数据中的：

```text
data/action
data/next_action
```

必须和真实执行时传给 `env.step(action)` 的动作定义完全一致。

## 5. Dataset 适配

训练代码期望 zarr 满足统一 schema：

```text
data/
  point_cloud
  next_point_cloud
  state
  next_state
  action
  next_action
  reward
  return
  done
  timeout

meta/
  episode_ends
```

已有数据准备入口：

```text
tools/teleop_off2off_data/data_prepare.py
tools/teleop_off2off_data/configs/data_prepare.yaml
```

需要适配：

| 适配项 | 说明 |
| --- | --- |
| raw 采集格式 | teleop 或 policy rollout 保存哪些字段 |
| raw 到 npy/zarr | 把机器人原始数据转换成项目 schema |
| `state` 映射 | 一般对应 policy 输入里的 `agent_pos` |
| `action` 映射 | 控制命令必须和 policy 输出空间一致 |
| `reward` | offline RL/IQL 依赖 reward |
| `return` | 通常由 reward 和 done/timeout 反向累计 |
| `done` | 任务成功/失败或 episode 真结束 |
| `timeout` | 到最大步数终止 |
| `episode_ends` | 每条轨迹的累计结束 index |
| `next_*` 字段 | 每个 transition 的下一帧数据 |

dataset 类要输出统一 batch contract：

```python
{
  "obs": {
    "point_cloud": ...,
    "agent_pos": ...,
    "image": ...,
  },
  "next_obs": {...},
  "action": ...,
  "next_action": ...,
  "reward": ...,
  "not_done": ...,
  "return": ...,
}
```

如果新机器人数据和已有类一致，可以复用 `AdroitDataset` 或类似 dataset；否则新增：

```text
RL-100/rl_100/dataset/<robot>_dataset.py
```

## 6. Env Runner 适配

训练和评估通过 `cfg.task.env_runner` 实例化 runner：

```text
train.py
  -> hydra.utils.instantiate(cfg.task.env_runner, output_dir=self.output_dir)
  -> env_runner.run(policy)
```

新机器人建议新增：

```text
RL-100/rl_100/env_runner/<robot>_runner.py
RL-100/rl_100/env/<robot>/<robot>_env.py
```

runner 至少实现：

```python
class XxxRunner(BaseRunner):
    def run(self, policy):
        ...

    def make_env(self, record_video=True):
        ...
```

`run()` 需要完成：

```text
env.reset()
  -> policy.reset()
  -> obs 转 torch tensor
  -> policy.predict_action(...)
  -> env.step(action)
  -> 统计 return / success / video
  -> 返回 log_data
```

推荐返回字段：

```python
{
    "mean_returns": ...,
    "mean_success_rates": ...,
    "test_mean_score": ...,
    "SR_test_L3": ...,
    "SR_test_L5": ...,
}
```

`test_mean_score` 很重要，因为 top-k checkpoint 和 best policy 选择依赖它。

## 7. Task YAML 适配

每个机器人/任务建议新增一个配置：

```text
RL-100/rl_100/config/task/<robot_task>.yaml
```

模板：

```yaml
name: new_robot_task
task_name: ${name}

shape_meta:
  obs:
    image:
      shape: [3, 84, 84]
      type: rgb
    point_cloud:
      shape: [512, 3]
      type: point_cloud
    agent_pos:
      shape: [D_STATE]
      type: low_dim
  action:
    shape: [D_ACTION]

env_runner:
  _target_: rl_100.env_runner.new_robot_runner.NewRobotRunner
  eval_episodes: 20
  max_steps: 200
  n_obs_steps: ${n_obs_steps}
  n_action_steps: ${n_action_steps}
  fps: 10
  env_num: 1

dataset:
  _target_: rl_100.dataset.new_robot_dataset.NewRobotDataset
  zarr_path: data/new_robot_task.zarr
  horizon: ${horizon}
  pad_before: ${eval:'${n_obs_steps}-1'}
  pad_after: ${eval:'${n_action_steps}-1'}
  seed: 42
  val_ratio: 0.02
```

如果使用 offline RL，最好也配置：

```yaml
critic_dataset:
  ...

finetune_dataset:
  ...

scale_dataset:
  ...
  scale_strategy: dynamic
```

## 8. Reward 和 Success 适配

offline RL、online RL、评估和 best checkpoint 都依赖 reward/success。

需要定义：

| 项 | 说明 |
| --- | --- |
| dense reward | 每步 reward，用于 value/Q/IQL |
| sparse success reward | 成功时给 1 或额外奖励 |
| success 判定 | 视觉检测、状态估计、人工标注或规则函数 |
| failure 判定 | 碰撞、掉落、超出 workspace、超时 |
| done | 成功/失败等真实 episode 结束 |
| timeout | 到最大步数结束 |
| return | zarr 中 `return` 的折扣累计规则 |

真实机器人任务中的 reward 来源可能是：

```text
人工标注
视觉检测器
物体状态估计
末端/物体几何规则
二阶段数据筛选
任务完成按钮
```

注意：如果 reward 逻辑改变，旧 zarr 的 `reward/return` 通常需要重新生成。

## 9. 模型配置适配

换机器人时模型主体通常不用重写，但配置要和新数据对齐。

| 情况 | 需要改 |
| --- | --- |
| action 维度变化 | `shape_meta.action.shape` |
| proprioception 维度变化 | `shape_meta.obs.agent_pos.shape` |
| 点云点数变化 | `shape_meta.obs.point_cloud.shape`、encoder `num_points` |
| 点云通道变化 | encoder `in_channels` |
| RGB 尺寸变化 | `shape_meta.obs.image.shape`、crop/resize |
| 使用 2D 图像 policy | `policy._target_=rl_100.policy.rl100_2d.RL1002D` |
| 使用 3D 点云 policy | `policy._target_=rl_100.policy.rl100_3d.RL1003D` |
| 控制频率变化 | `fps`、`horizon`、`n_action_steps` |
| chunk 控制变化 | `chunk_as_single_action`、`n_action_steps` |
| 想用 flow | `policy.scheduler_type='flow'` 和 flow scheduler 配置 |
| 想用 diffusion/DDIM | `policy.scheduler_type='ddim'` 和 DDIM scheduler 配置 |

## 10. 真实机器人安全适配

真实机器人不要只依赖 policy 输出，必须在 env/wrapper 层加保护。

建议至少实现：

| 安全项 | 说明 |
| --- | --- |
| workspace bounds | 限制末端 xyz 范围 |
| joint limits | 防止超过关节限位 |
| velocity limits | 限制每步最大动作变化 |
| acceleration limits | 防止动作抖动和冲击 |
| collision guard | 桌面、夹具、自碰撞、人体安全区 |
| emergency stop | 外部急停必须随时可用 |
| action smoothing | 平滑 policy 输出 |
| reset safety | reset 走安全轨迹，而不是瞬移或直接大动作 |
| gripper force limit | 防止夹坏物体或夹具 |
| watchdog | 通信异常、循环超时立即停止 |
| dry-run mode | 只打印/可视化动作，不真正执行 |

建议所有安全限制放在机器人 env 或低层 wrapper 中，而不是散落在 policy 或训练代码里。

## 11. 最小接入步骤

接入一个新机器人时，可以按这个顺序做：

1. 定义 action 空间：维度、单位、坐标系、scale、是否 chunk。
2. 定义 observation 空间：`agent_pos`、`image`、`point_cloud`。
3. 做相机内外参标定和点云处理。
4. 实现 robot env：`reset()`、`step(action)`、`get_obs()`。
5. 实现安全 wrapper：bounds、速度限制、急停、watchdog。
6. 实现 teleop 或 policy rollout 数据采集。
7. 转换数据到 zarr schema。
8. 新增或复用 dataset 类，确认 batch contract。
9. 新增 task yaml，配置 `shape_meta`、dataset、env_runner。
10. 新增 env_runner，返回 `test_mean_score`。
11. 用少量数据跑 BC smoke test，确认 loss 能下降。
12. 跑短评估，确认 `policy.predict_action()` 输出能被机器人安全执行。
13. 再进入 offline RL、distill、online RL 或 off2off 数据飞轮。

## 12. Smoke Test 建议

正式训练前建议做几类小测试：

| 测试 | 目标 |
| --- | --- |
| zarr 读取测试 | dataset 能读出 batch，shape 和 dtype 正确 |
| normalizer 测试 | action/state 不出现异常极值或 NaN |
| policy forward 测试 | `compute_loss()` 能跑通 |
| predict_action 测试 | 输出 action shape 和范围正确 |
| env dry-run 测试 | action 映射到机器人命令前先检查 clipping |
| 真实机器人慢速执行 | 小动作、低频率、短 episode |
| reset 测试 | 多次 reset 后状态稳定 |
| success/reward 测试 | 成功和失败轨迹能正确写 reward/done/timeout |

## 13. 快速定位表

| 想改的东西 | 优先看哪里 |
| --- | --- |
| 数据路径 | `RL-100/rl_100/config/task/<task>.yaml` 的 `dataset.zarr_path` |
| obs/action shape | `shape_meta` |
| 机器人动作执行 | `RL-100/rl_100/env/<robot>/` |
| 评估逻辑 | `RL-100/rl_100/env_runner/<robot>_runner.py` |
| zarr 写入 | `tools/teleop_off2off_data/data_prepare.py` |
| batch 格式 | `RL-100/rl_100/dataset/*_dataset.py::_sample_to_data()` |
| 模型输入 | `RL-100/rl_100/policy/rl100_3d.py` 或 `rl100_2d.py` |
| 训练阶段分支 | `RL-100/train.py::TrainDP3Workspace.run()` |
| 真实机器人采集 | `RL-100/train_real.py` 和 `data_collect=True` |

一句话总结：换机器人本体时，模型通常可以复用；真正要适配的是机器人 I/O、action/observation 定义、数据 schema、runner 评估、task yaml、reward/success 和安全层。
