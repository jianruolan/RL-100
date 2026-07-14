# Dataset Guide

本文档整理 RL-100 中数据集的构成、不同训练阶段使用的数据来源，以及当你要新增一个任务时需要补充哪些数据集。

一句话先说结论：RL-100 的训练代码最终都吃统一的 zarr replay dataset；区别在于这个 zarr 是公开/benchmark 数据、仿真 expert 生成数据，还是真实机器人针对任务采集的数据。

## 1. 数据集在项目里的位置

训练入口不会直接读 raw 文件，而是通过 task yaml 指向一个 zarr：

```text
RL-100/rl_100/config/task/<task>.yaml
  -> dataset.zarr_path
  -> critic_dataset.zarr_path
  -> finetune_dataset.zarr_path
  -> scale_dataset.zarr_path
```

典型路径：

```text
RL-100/data/*.zarr
```

代码流：

```text
task yaml
  -> cfg.task.dataset
  -> hydra.utils.instantiate(...)
  -> BaseDataset subclass
  -> ReplayBuffer.copy_from_path(zarr_path)
  -> SequenceSampler
  -> DataLoader
  -> policy / critic / PPO / IQL
```

常见 dataset 类：

| 类 | 主要任务 |
| --- | --- |
| `AdroitDataset` | Adroit door/hammer/pen |
| `MetaworldDataset` | MetaWorld 单视角任务 |
| `MetaworldMultiViewDataset` | MetaWorld 多视角任务 |
| `DexArtDataset` | DexArt laptop/faucet/bucket/toilet |
| `DMCDataset` | DMC cheetah/walker/cartpole 等 |
| `RealDexDataset` | realdex roll/drill/pour/dumpling 等真实任务 |
| `RotateDataset`, `PourDataset`, `JuicingDataset`, `FoldingDataset`, `Cloth` | 项目内真实或特定任务采集数据 |

## 2. zarr 数据集标准结构

训练代码期望 zarr 至少包含：

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

部分 2D 或仿真数据还会包含：

```text
data/
  img
  next_img
  full_state
```

dataset 输出给模型的 batch 通常是：

```python
{
  "obs": {
    "point_cloud": ...,
    "agent_pos": ...,
    "image": ...,
  },
  "next_obs": {
    "point_cloud": ...,
    "agent_pos": ...,
    "image": ...,
  },
  "action": ...,
  "next_action": ...,
  "reward": ...,
  "not_done": ...,
  "return": ...,
}
```

注意：zarr 里叫 `state`，进入 policy 后通常映射成 `agent_pos`。

## 3. 按训练阶段看数据来源

### Stage 0: 数据准备

来源可能有三类：

| 来源 | 例子 | 是否针对任务采集 |
| --- | --- | --- |
| 公开/benchmark 数据 | D4RL/Adroit medium、DMC replay 等 | 通常不是你现场采集，但仍是任务相关数据 |
| 仿真 expert 生成 | MetaWorld、DexArt、Adroit expert rollout | 针对仿真任务生成 |
| 真实机器人采集 | teleop、真实机器人 policy rollout、off2off 数据 | 必须针对具体机器人和任务采集 |

项目里的仿真生成脚本：

```text
scripts/gen_demonstration_metaworld.sh
scripts/gen_demonstration_adroit.sh
scripts/gen_demonstration_adroit_medium.sh
scripts/gen_demonstration_adroit_medium_expert.sh
scripts/gen_demonstration_dexart.sh
scripts/gen_demonstration_metaworld_multiview.sh
```

真实机器人数据准备入口：

```text
tools/teleop_off2off_data/data_prepare.py
tools/teleop_off2off_data/configs/data_prepare.yaml
```

### Stage 1: BC 初始化

使用数据：

```text
cfg.task.dataset
```

数据性质：

| 场景 | 数据来源 |
| --- | --- |
| Adroit/MetaWorld/DexArt/DMC smoke test | 公开或仿真 expert zarr |
| 真实机器人任务 | 人类 teleop demonstration zarr，或已有成功轨迹 zarr |

需要包含：

```text
obs / action
```

最好也包含：

```text
next_obs / next_action / reward / done / timeout / return
```

因为后续 offline RL 会用到。

### Stage 2: Offline RL post-training

使用数据：

```text
cfg.task.dataset
cfg.task.critic_dataset
cfg.task.finetune_dataset
cfg.task.scale_dataset
```

这些可以指向同一个 zarr，也可以拆开：

| 数据配置 | 用途 |
| --- | --- |
| `dataset` | BC 和主训练数据 |
| `critic_dataset` | IQL/Q/V/critic 训练 |
| `finetune_dataset` | BPPO/offline actor 更新 |
| `scale_dataset` | reward scaling 或 value/reward 归一化 |

数据性质：

| 场景 | 数据来源 |
| --- | --- |
| benchmark offline RL | 公开或生成好的 offline replay |
| 仿真任务 | expert rollout、medium replay、medium-expert mixture |
| 真实机器人 | teleop + 上一轮 policy rollout 合并后的 zarr |

offline RL 比 BC 更依赖 reward 质量，需要确保：

```text
reward
return
done
timeout
next_state
next_action
next_point_cloud
```

都正确。

### Stage 3: One-step distillation

使用数据：

```text
train_dataloader / val_dataloader
```

也就是和当前训练阶段相同的 zarr batch。

数据性质：

| distill_phase | 数据来源 |
| --- | --- |
| `after_dp` | BC 阶段同一个 dataset |
| `after_offline` | offline RL 训练时同一个 dataset |
| `online` | 在线 rollout + 当前 policy/teacher 输出，仍会用离线数据作辅助 |

distillation 的核心 teacher 是训练出来的 diffusion/flow policy，不是新的外部数据集。

### Stage 4: Online RL fine-tuning

使用数据：

```text
env_runner live rollout
online replay buffer
optional offline dataloader mixed batch
```

数据性质：

| 场景 | 数据来源 |
| --- | --- |
| 仿真 online | 在 MetaWorld/Adroit/DexArt/DMC 环境中即时 rollout |
| 真实机器人 online | 在真实机器人上即时 rollout |
| online IQL 混合 | online buffer + offline dataset 混合 |

在线数据不一定马上写成 zarr。训练时通常先进入 online replay buffer；如果是 off2off 数据飞轮，会把真实机器人 rollout 保存成 `.h5`，再用 `data_prepare.py` 合并成下一轮 zarr。

### Stage 5: Off2Off 数据飞轮

这是面向真实机器人的迭代离线训练流程：

```text
teleop zarr
  -> offline RL policy
  -> real robot rollout .h5
  -> data_prepare.py extend_zarr / build_zarr
  -> next-round zarr
  -> next offline RL
```

使用数据：

| 数据 | 来源 | 作用 |
| --- | --- | --- |
| base zarr | teleop 或上一轮 zarr | 作为历史数据 |
| rollout `.h5` | 当前 policy 在真实机器人上跑出来 | 增量数据 |
| new zarr | base + rollout 合并 | 下一轮训练 |

工具入口：

```bash
cd tools/teleop_off2off_data

python data_prepare.py \
  --config configs/data_prepare.yaml \
  --mode extend_zarr \
  --rollout-source off2off_004 \
  --base-zarr-path /path/to/base.zarr \
  --zarr-output-path /path/to/new.zarr
```

## 4. 按任务类型看数据来源

### 4.1 Adroit

配置例子：

```text
adroit_door.yaml
adroit_door_medium.yaml
adroit_door_medium_expert.yaml
adroit_hammer*.yaml
adroit_pen*.yaml
```

数据路径例子：

```text
data/adroit_door_expert.zarr
data/adroit_door_medium.zarr
data/adroit_door_medium_expert.zarr
data/adroit_hammer_medium.zarr
data/adroit_pen_medium.zarr
```

来源：

| 数据名特征 | 含义 |
| --- | --- |
| `*_expert.zarr` | expert policy 在仿真/benchmark 环境生成 |
| `*_medium.zarr` | medium 质量 offline 数据，类似 benchmark/offline RL 数据 |
| `*_medium_expert.zarr` | medium + expert 混合 |

生成参考：

```bash
bash scripts/gen_demonstration_adroit.sh door
bash scripts/gen_demonstration_adroit.sh hammer
bash scripts/gen_demonstration_adroit.sh pen
```

README 里也提供了 `adroit_door_medium.zarr` 的下载示例。

### 4.2 MetaWorld

配置例子：

```text
metaworld_assembly.yaml
metaworld_push.yaml
metaworld_pick-place.yaml
metaworld_button-press.yaml
metaworld_*_medium-expert.yaml
metaworld_*_multiview.yaml
```

数据路径例子：

```text
data/metaworld_assembly_expert.zarr
data/metaworld_push_expert.zarr
data/metaworld_pick-place_medium-expert.zarr
```

来源：

| 数据名特征 | 含义 |
| --- | --- |
| `*_expert.zarr` | MetaWorld 仿真 expert 轨迹 |
| `*_medium-expert.zarr` | 中等质量和 expert 混合数据 |
| `*_multiview` 配置 | 多视角图像/点云数据路径 |

生成参考：

```bash
bash scripts/gen_demonstration_metaworld.sh basketball
```

### 4.3 DexArt

配置例子：

```text
dexart_laptop.yaml
dexart_faucet.yaml
dexart_bucket.yaml
dexart_toilet.yaml
```

数据路径例子：

```text
data/dexart_laptop_expert.zarr
data/dexart_faucet_expert.zarr
data/dexart_bucket_expert.zarr
data/dexart_toilet_expert.zarr
```

来源：DexArt 仿真环境 expert policy rollout。

生成参考：

```bash
bash scripts/gen_demonstration_dexart.sh laptop
bash scripts/gen_demonstration_dexart.sh faucet
bash scripts/gen_demonstration_dexart.sh bucket
bash scripts/gen_demonstration_dexart.sh toilet
```

### 4.4 DMC

配置例子：

```text
dmc_cheetah_run.yaml
dmc_cheetah_run_medium.yaml
dmc_cheetah_run_replay.yaml
dmc_walker_run.yaml
dmc_cartpole_swingup.yaml
```

数据路径例子：

```text
data/dmc_cheetah_run_medium_replay_expert.zarr
```

来源：DMC 仿真环境 replay 或 expert/medium replay 数据。不是机器人本体采集数据，主要用于算法验证和仿真 benchmark。

### 4.5 RealDex / 真实机器人任务

配置例子：

```text
realdex_roll.yaml
realdex_drill.yaml
realdex_pour.yaml
realdex_dumpling.yaml
```

数据路径例子：

```text
data/realdex_roll.zarr
data/realdex_drill.zarr
```

来源：真实机器人/真实任务采集数据。一般需要你针对具体任务采集 teleop demonstration 或 policy rollout。

### 4.6 项目内真实或特定任务数据

配置例子：

```text
rotate.yaml
rotate_512_arm.yaml
rotate_1024.yaml
pour.yaml
juicing.yaml
folding.yaml
cloth.yaml
flipping.yaml
bowling.yaml
push_t.yaml
```

数据路径例子：

```text
data/data_rotate_100_512.zarr
data/data_rotate_100_arm_512.zarr
data/data_pour_100_512.zarr
data/data_juicing_100_1024.zarr
data/cloth_rgb_240_320_action.zarr
```

来源通常是项目作者针对真实或特定任务采集/处理后的数据。若你换任务，不能直接复用这些数据，只能复用 schema、dataset 写法和处理流程。

## 5. 如果新增一个任务，需要补哪些数据集

新增任务时，最小需要补一套离线 demonstration zarr。更完整的 RL-100 流程建议准备三类数据。

### 5.1 必需：任务 demonstration dataset

用途：

```text
BC 初始化
offline RL 初始数据
normalizer 统计
```

来源：

| 场景 | 采集方式 |
| --- | --- |
| 仿真任务 | expert policy、scripted policy、planner、human teleop in sim |
| 真实机器人任务 | human teleop、kinesthetic teaching、VR/keyboard/spacemouse 控制 |

必须包含：

```text
state / point_cloud / image
action
done / timeout
episode_ends
```

强烈建议包含：

```text
next_state
next_point_cloud
next_image
next_action
reward
return
```

否则 BC 也许能跑，但 offline RL/IQL/dynamics 会受影响。

### 5.2 推荐：任务 reward / success labeled dataset

用途：

```text
offline RL value/Q/IQL
best checkpoint selection
后续 rollout 过滤
```

你需要为每条 trajectory 或 transition 标好：

```text
reward
done
timeout
is_success 或 success info
```

如果是 sparse reward，至少要在成功终点给正奖励。如果是 dense reward，要保证尺度稳定，不要出现异常尖峰。

### 5.3 推荐：policy rollout dataset

用途：

```text
off2off 数据飞轮
扩展 offline dataset
覆盖 teleop 没覆盖到的状态分布
```

来源：

```text
用 BC/offline policy 在仿真或真实机器人上 rollout
```

真实机器人中一般先保存 `.h5` rollout，再转 zarr：

```text
online_ft/.../*.h5
  -> data_prepare.py
  -> new_round.zarr
```

### 5.4 可选：validation / held-out dataset

用途：

```text
验证 BC loss
检查 overfit
比较不同任务设置
```

项目里很多 dataset 通过 `val_ratio` 从同一个 zarr 切 validation set，不一定要单独准备文件。但真实任务如果场景变化明显，建议单独留一批 held-out episodes。

### 5.5 可选：scale dataset

用途：

```text
reward scaling
return normalization
critic/value 稳定训练
```

如果 reward 分布和 demonstration 数据差异很大，可以把 `scale_dataset` 单独指向更合适的数据源。

## 6. 新任务数据采集清单

如果你要针对某个具体任务采集数据，建议至少记录这些字段。

### 每一帧 transition

```text
timestamp
state
point_cloud
image
action
reward
done
timeout
is_success
```

其中：

| 字段 | 说明 |
| --- | --- |
| `state` | 机器人 proprioception，最终映射成 `agent_pos` |
| `point_cloud` | 任务空间点云，最好在机器人 base/world 坐标系 |
| `image` | RGB 图像，形状要和 `shape_meta` 一致 |
| `action` | 实际下发的控制命令，不是 UI 原始输入 |
| `reward` | 当前步奖励 |
| `done` | 任务完成/失败等真实终止 |
| `timeout` | 达到最大步数 |
| `is_success` | episode 是否成功，便于过滤和统计 |

### 每个 episode

```text
episode_id
task_id / task_name
initial_state
camera_calibration_version
robot_config_version
operator_id or policy_id
success
failure_reason
```

这些元信息不一定进入训练 batch，但对排查数据质量很有用。

## 7. 新任务数据量建议

项目没有硬编码必须多少条数据。实际需要量取决于任务难度、状态分布、观测噪声和动作维度。可以按阶段推进：

| 阶段 | 建议数据 |
| --- | --- |
| smoke test | 5-10 条成功 episode，先验证 zarr、shape、BC forward |
| BC baseline | 50-200 条高质量 teleop/expert episode |
| offline RL 初版 | BC 数据 + reward/success 完整标注 |
| off2off 第 1 轮 | 新增 20-100 条 policy rollout，失败也可保留但 reward/done 要准确 |
| 稳定提升 | 多轮 rollout，每轮合并新的成功/失败覆盖 |

高质量数据比盲目堆量更重要。尤其真实机器人任务中，action 定义、相机坐标和 reward 标注错误会比数据少更致命。

## 8. 新任务接入步骤

### 8.1 仿真任务

需要补：

1. 仿真 env 或 wrapper。
2. expert/scripted policy 或 teleop 采集方式。
3. 生成 zarr 的脚本。
4. task yaml，指向 `dataset.zarr_path` 和 `env_runner`。
5. dataset 类，如果现有 schema 不兼容。

推荐流程：

```text
generate expert zarr
  -> run BC
  -> evaluate in sim env_runner
  -> offline RL
  -> optional online RL in sim
```

### 8.2 真实机器人任务

需要补：

1. robot env/wrapper。
2. teleop 采集脚本。
3. 相机标定和点云处理。
4. reward/success 标注方式。
5. raw -> npy/zarr 转换逻辑。
6. task yaml。
7. env_runner。
8. 安全 wrapper。

推荐流程：

```text
teleop raw
  -> raw_to_npy
  -> build_zarr
  -> BC
  -> offline RL
  -> real rollout .h5
  -> extend_zarr
  -> next offline RL
  -> optional online RL
```

## 9. task yaml 中多数据集字段怎么填

最简单版本：全部指向同一个 zarr。

```yaml
dataset:
  _target_: rl_100.dataset.my_task_dataset.MyTaskDataset
  zarr_path: data/my_task.zarr

critic_dataset:
  _target_: rl_100.dataset.my_task_dataset.MyTaskDataset
  zarr_path: data/my_task.zarr

finetune_dataset:
  _target_: rl_100.dataset.my_task_dataset.MyTaskDataset
  zarr_path: data/my_task.zarr

scale_dataset:
  _target_: rl_100.dataset.my_task_dataset.MyTaskDataset
  zarr_path: data/my_task.zarr
  scale_strategy: dynamic
```

什么时候拆开：

| 情况 | 建议 |
| --- | --- |
| critic 需要更多失败/低质量数据 | `critic_dataset` 指向更大混合数据 |
| actor finetune 只想用高质量数据 | `finetune_dataset` 指向过滤后的成功数据 |
| reward scaling 想覆盖更广 | `scale_dataset` 指向包含多轮 rollout 的数据 |
| chunk action 需要不同 stride | `critic_dataset` / `finetune_dataset` 配不同 `sequence_stride` |

## 10. 数据质量检查

新任务数据生成后，训练前建议检查：

| 检查项 | 目标 |
| --- | --- |
| shape | 和 `shape_meta` 完全一致 |
| dtype | 通常 float32 / bool / int64 |
| episode_ends | 单调递增，最后等于 transition 总数 |
| action 范围 | 没有异常大值，单位正确 |
| state 范围 | 没有 NaN/Inf，维度稳定 |
| point_cloud | 点数固定，坐标系正确 |
| image | 通道顺序和大小正确 |
| reward/return | return 能和 reward/done 对上 |
| done/timeout | episode 结束逻辑正确 |
| next_* | 最后一帧处理合理，其余帧和下一步对齐 |

## 11. 常见问题

### 只做 BC 是否必须有 reward？

严格说，BC 主要需要 `obs` 和 `action`。但项目的 dataset 类和后续 offline RL 通常会读取 `reward/done/return/next_*`，所以建议从一开始就按完整 schema 写。

### 可以直接用公开数据训练真实机器人吗？

通常不够。公开/仿真数据可以用于算法 smoke test 或预训练，但真实机器人任务需要至少补充本体对应的观测、动作和相机坐标数据。否则 action 维度、尺度、坐标系和视觉分布都会不匹配。

### 不同任务之间能共用一个数据集吗？

只有在任务定义、机器人本体、action 空间、obs 空间、奖励和成功判定一致时才建议共用。不同任务通常至少要单独采集 demonstration 和 reward/success 标注。

### policy rollout 失败数据要不要保留？

offline RL 可以利用失败数据，但前提是 reward/done/timeout 标注准确。如果失败来自安全异常、传感器错误或 reset 错误，建议过滤或单独标记。

### 什么时候重新 build zarr，什么时候 extend zarr？

`build_zarr` 适合 reward、点云处理、action/state schema 或过滤规则改变时；`extend_zarr` 适合只新增一批 policy rollout，且预处理逻辑没变时。

## 12. 最小结论

如果你要做一个新任务，至少要补：

```text
1. 任务 demonstration zarr
2. reward / done / timeout / return
3. next_state / next_action / next_point_cloud / next_img
4. task yaml 中的 shape_meta、dataset、env_runner
5. 如果是真实机器人，还要补 teleop/raw 数据采集和 rollout-to-zarr 数据飞轮
```

仿真任务可以用 expert policy 或 benchmark replay 生成数据；真实机器人任务必须针对具体机器人、具体任务、具体相机和动作空间采集数据。
