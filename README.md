<h1 align="center">RL-100</h1>

<h3 align="center">面向真实世界强化学习的高性能机器人操作框架</h3>

<p align="center">
  <b>一个统一支持 diffusion / flow policy 强化学习后训练、迭代离线数据飞轮和真实机器人强化学习的代码库。</b>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2510.14830"><img src="https://img.shields.io/badge/Paper-arXiv%202510.14830-b31b1b?style=for-the-badge" alt="Paper"></a>
  <a href="https://lei-kun.github.io/RL-100/"><img src="https://img.shields.io/badge/Project-Page-2563eb?style=for-the-badge" alt="Project Page"></a>
  <a href="https://youtu.be/OrnHUgMGmlU"><img src="https://img.shields.io/badge/Video-YouTube-dc2626?style=for-the-badge" alt="YouTube"></a>
  <a href="https://x.com/kunlei15/status/1978840280297255124"><img src="https://img.shields.io/badge/Social-Twitter%20%2F%20X-111827?style=for-the-badge" alt="Twitter / X"></a>
</p>

<p align="center">
  <a href="https://lei-kun.github.io/blogs/RL100.html"><img src="https://img.shields.io/badge/Blog-RL--100-7c3aed?style=flat-square" alt="Blog RL-100"></a>
  <a href="https://lei-kun.github.io/blogs/rl.html"><img src="https://img.shields.io/badge/Blog-RL-7c3aed?style=flat-square" alt="Blog RL"></a>
  <a href="https://zhuanlan.zhihu.com/p/1961019618517836122"><img src="https://img.shields.io/badge/Article-Zhihu-2563eb?style=flat-square" alt="Zhihu"></a>
  <a href="https://lei-kun.github.io/my_files/RL-100.pdf"><img src="https://img.shields.io/badge/Talk-Slides-f59e0b?style=flat-square" alt="Talk Slides"></a>
  <a href="https://www.bilibili.com/video/BV1UurLBoEJK/"><img src="https://img.shields.io/badge/Talk-Bilibili-ec4899?style=flat-square" alt="Talk Bilibili"></a>
  <a href="https://twitter.com/RoboPapers/status/2011453253314134277"><img src="https://img.shields.io/badge/Talk-RoboPapers-111827?style=flat-square" alt="Talk RoboPapers"></a>
</p>

---

<h4 align="center">作者</h4>

<p align="center">
  <a href="https://lei-kun.github.io/">Kun Lei</a><sup>*†</sup> ·
  <a href="https://li-huanyu.github.io/">Huanyu Li</a><sup>*</sup> ·
  <a href="https://manutdmoon.github.io/">Dongjie Yu</a><sup>*</sup> ·
  <a href="https://zhenyuwei2003.github.io/">Zhenyu Wei</a><sup>*</sup>
  <br>
  <a href="https://lingxiao-guo.github.io/">Lingxiao Guo</a> ·
  <a href="https://jzndd.github.io/">Zhennan Jiang</a> ·
  <a href="https://wadiuvatzy.github.io">Ziyu Wang</a> ·
  <a href="https://www.liang-shiyu.com/">Shiyu Liang</a> ·
  <a href="http://hxu.rocks/">Huazhe Xu</a>
</p>

<p align="center"><b>
  <sup>*</sup> 共同一作 &nbsp;·&nbsp; <sup>†</sup> 项目负责人
</b>
</p>

![RL-100 repository overview](media/overview.jpg)

---

## 概览

**RL-100** 是一个基于 diffusion 和 flow 视觉运动策略的真实世界机器人操作强化学习框架。它通过统一的 imitation-to-reinforcement learning 流水线，面向部署级别的**可靠性**、**效率**和**鲁棒性**。

RL-100 旨在提供一个较完整的 diffusion policy RL 和真实机器人 RL 后训练代码库。该框架在一个统一仓库中覆盖以下组合。

> **下方支持项可以跨轴自由组合，用于不同机器人后训练设置。**
>
> <span style="color:#dc2626"><strong>所有组合都通过紧凑的 offline RL / online RL 训练接口实现。</strong></span>

<table width="100%">
<thead>
<tr>
  <th align="left">维度</th>
  <th align="left">支持选项</th>
</tr>
</thead>
<tbody>
<tr>
  <td><b>策略骨干</b></td>
  <td>
    <span style="color:#16a34a">✓</span> Diffusion policy<br>
    <span style="color:#16a34a">✓</span> Flow policy
  </td>
</tr>
<tr>
  <td><b>一步部署</b></td>
  <td>
    <span style="color:#2563eb">✓</span> Diffusion-to-CM 一步蒸馏<br>
    <span style="color:#2563eb">✓</span> Flow-to-flow 一步 on-policy 蒸馏
  </td>
</tr>
<tr>
  <td><b>观测模态</b></td>
  <td>
    <span style="color:#f59e0b">✓</span> 3D 点云<br>
    <span style="color:#f59e0b">✓</span> 2D RGB 图像
  </td>
</tr>
<tr>
  <td><b>控制模式</b></td>
  <td>
    <span style="color:#dc2626">✓</span> Action chunking<br>
    <span style="color:#dc2626">✓</span> 单动作高频控制
  </td>
</tr>
<tr>
  <td><b>策略提取策略</b></td>
  <td>
    <span style="color:#7c3aed">✓</span> Policy gradient<br>
    <span style="color:#7c3aed">✓</span> 带拒绝采样的 IDQL 风格提取<br>
    <span style="color:#7c3aed">✓</span> 混合变体
  </td>
</tr>
<tr>
  <td><b>RL 阶段</b></td>
  <td>
    <span style="color:#0891b2">✓</span> Offline policy gradient<br>
    <span style="color:#0891b2">✓</span> Online policy gradient
  </td>
</tr>
<tr>
  <td><b>训练数据机制</b></td>
  <td>
    <span style="color:#db2777">✓</span> 遥操作数据集<br>
    <span style="color:#db2777">✓</span> 迭代离线 rollout 数据集<br>
    <span style="color:#db2777">✓</span> Online rollouts<br>
    <span style="color:#db2777">✓</span> Offline-to-online 混合训练
  </td>
</tr>
</tbody>
</table>

> 从人类遥操作 demonstration 出发，RL-100 会用 offline RL 迭代改进策略，把策略部署到真实机器人上采集新的 rollout，再把这些 rollout 合并回离线数据集，并可选进行轻量级 online RL 微调。

---

## 目录

- [0. 快速安装指引](#0-快速安装指引)
  - [下载 Smoke-Test 数据集](#下载-smoke-test-数据集)
- [1. 项目概览](#1-项目概览)
- [2. RL 后训练框架](#2-rl-后训练框架)
- [3. 真实机器人训练和数据飞轮](#3-真实机器人训练和数据飞轮)
- [安装](#安装)
- [引用](#引用)
- [致谢](#致谢)
- [许可证](#许可证)

---

## 0. 快速安装指引

环境搭建和依赖安装请参见[安装](#安装)章节和 [`INSTALL.md`](INSTALL.md)。我们建议用 `adroit_door_medium` 作为第一个 smoke-test 任务。

### 下载 Smoke-Test 数据集

推荐的 `adroit_door_medium` zarr 数据集托管在 Hugging Face Datasets：[`leokk/RL-100-adroit-door-medium`](https://huggingface.co/datasets/leokk/RL-100-adroit-door-medium)。在仓库根目录下，可以用 Hugging Face CLI 下载：

```bash
python -m pip install -U huggingface_hub
mkdir -p RL-100/data
hf download leokk/RL-100-adroit-door-medium \
  adroit_door_medium.zarr.tar.gz \
  --repo-type dataset \
  --local-dir /tmp
tar -xzf /tmp/adroit_door_medium.zarr.tar.gz -C RL-100/data
```

如果你的环境更适合直接 URL 下载，可以使用：

```bash
mkdir -p RL-100/data
wget -O /tmp/adroit_door_medium.zarr.tar.gz \
  https://huggingface.co/datasets/leokk/RL-100-adroit-door-medium/resolve/main/adroit_door_medium.zarr.tar.gz
tar -xzf /tmp/adroit_door_medium.zarr.tar.gz -C RL-100/data
```

可选的校验和验证：

```bash
echo "73a7cd510a0715492e8f8041843ac51edf1a0feb6d30706a95a5e5857addef37  /tmp/adroit_door_medium.zarr.tar.gz" | sha256sum -c -
```

解压后，数据集应位于：

```text
Repo/RL-100/data/adroit_door_medium.zarr
```

该数据集遵循 RL-100 仓库/数据发布条款。正式数据集许可证会和公开仓库许可证一起最终确定。

## 1. 项目概览

### 为什么是 RL-100

RL-100 被设计为一个**全栈真实世界机器人学习系统**，而不只是一组训练脚本。

它在**一个代码库**中结合了：

- **行为克隆**
- **离线 policy-gradient RL**
- **迭代离线 rollout 数据扩展**
- **在线 policy-gradient 微调**
- **一步策略蒸馏**

目标是把真实机器人策略后训练变成一个**可重复流水线**，而不是一组互不相连的脚本。

### 仓库结构

```text
RL-100/
├── rl_100/
│   ├── policy/              # 2D 和 3D 视觉运动策略实现
│   ├── unidpg/              # offline / online RL 训练逻辑
│   └── config/              # policy、task 和训练配置
├── scripts/
│   ├── Diffusion/           # diffusion policy 训练和 online 脚本
│   └── Flow/                # flow policy 训练和 online 脚本
├── tools/
│   └── teleop_off2off_data/ # 遥操作和迭代离线数据飞轮工具
├── third_party/             # 外部环境和依赖
└── visualizer/              # 轻量级点云可视化器
```

---

## 2. RL 后训练框架

RL-100 把上面的能力矩阵落实为一条实用的后训练流水线，用于从日志数据和 online 交互中适配大型视觉运动策略。本节关注算法阶段和启动器；真实机器人迭代离线数据飞轮见[第 3 节](#3-真实机器人训练和数据飞轮)。

### 命令选择器

可以使用交互式选择器，根据几个选项生成启动命令：

```bash
python scripts/select_recipe.py
```

它会询问：

<table width="100%">
<thead>
<tr>
  <th align="center" width="80">#</th>
  <th align="left">选项</th>
</tr>
</thead>
<tbody>
<tr><td align="center">1</td><td>DDIM &nbsp;/&nbsp; Flow</td></tr>
<tr><td align="center">2</td><td>Offline &nbsp;/&nbsp; Online</td></tr>
<tr><td align="center">3</td><td>3D &nbsp;/&nbsp; 2D</td></tr>
<tr><td align="center">4</td><td>Chunk action &nbsp;/&nbsp; Single action</td></tr>
<tr><td align="center">5</td><td>PG &nbsp;/&nbsp; PG + IDQL</td></tr>
<tr><td align="center">6</td><td>一步蒸馏：是 &nbsp;/&nbsp; 否</td></tr>
</tbody>
</table>

前四个选项决定 launcher 路径。PG/IDQL 和蒸馏选项会打印 launcher 级别的建议，因为这些设置通常在 bash sweep 内控制。例如，选择器会提醒你使用 launcher flag 来启用 **PG** 或 **PG + IDQL 风格提取**；对于 offline distillation，应先完成 offline sweep 和参数选择，再设置 `distill_phase='after_offline'`；对于 online distillation，应设置 `distill_phase='online'`。

<details>
<summary><b>示例输出</b></summary>

```text
Command:
bash scripts/Flow/Online/3D/train_policy_online_flow_distill_online.sh rl100 adroit_door_medium 0112 100 8
```

</details>

### 训练阶段

完整算法流水线围绕以下阶段组织。

#### Stage 1 · 行为克隆初始化

从 demonstrations 训练初始视觉运动策略。该阶段在 RL 后训练前提供稳定的 policy prior。

#### Stage 2 · Offline RL 后训练

在日志轨迹上使用 offline policy-gradient 更新来改进初始化策略。该阶段支持 diffusion 和 flow policy、2D 和 3D 观测骨干、chunk-action 和 single-action 控制，以及通过 launcher sweep 配置的可选 IDQL 风格提取。

**Offline RL 启动器示例：**

```bash
# 单卡 3D diffusion offline RL。
bash scripts/Diffusion/Offline/3D/train_policy.sh rl100 adroit_door_medium 0112 100

# DDP 两阶段 3D diffusion offline RL。
# 最后一个参数是 GPU 数量。
bash scripts/Diffusion/Offline/3D/train_policy_two_stage.sh rl100 adroit_door_medium 0112 100 4
```

2D 和 chunk-action offline recipe 也提供 DDP 风格启动器，包括：

- `scripts/Diffusion/Offline/2D/train_policy_image_unet_two_stage.sh`
- `scripts/Diffusion/Offline/2D/train_policy_image_unet_chunk_two_stage.sh`
- `scripts/Diffusion/Offline/3D/train_policy_chunk_two_stage.sh`

#### Stage 3 · Online RL 微调

使用真实或仿真的 online rollouts 继续改进策略。当前 rollout policy 收集 transitions，PPO 风格更新复用已有 rollout 和 log-probability 流程。

该阶段支持：

- Online policy gradient。
- Diffusion 和 flow online 微调。
- 2D 和 3D online runners。
- Chunked 和高频 single-action 控制。
- Offline RL 之后的轻量 online adaptation。

#### Stage 4 · 一步策略蒸馏

为了提升部署效率，RL-100 包含一步策略提取和蒸馏路径：

- **Diffusion-to-CM**：把多步 diffusion policy 蒸馏成 consistency-model 风格的一步策略。
- **Flow-to-flow**：通过 on-policy training 蒸馏或提取一步 flow policy。

这些路径面向快速真实机器人推理，同时保留 RL 后训练带来的行为改进。

#### Stage 5 · 策略提取

RL-100 支持机器人策略后训练中使用的策略提取方式：

- 从日志或 online transitions 中进行 policy-gradient 提取。
- 使用来自高质量采样轨迹的拒绝采样进行 IDQL 风格提取。
- 结合 policy-gradient 更新和过滤/选择数据的混合策略。

### 配置入口

主要实现和配置入口是：

```text
RL-100/rl_100/policy/rl100_3d.py
RL-100/rl_100/policy/rl100_2d.py
RL-100/rl_100/unidpg/
RL-100/rl_100/config/
scripts/Diffusion/
scripts/Flow/
```

详细命令行和推荐配置围绕这些入口组织，并会随着 release 脚本最终确定继续扩展。

---

## 3. 真实机器人训练和数据飞轮

RL-100 包含一个真实机器人数据飞轮，用于迭代离线强化学习和最终 online improvement。

<table width="100%">
<thead>
<tr>
  <th align="left">阶段</th>
  <th align="left">输入</th>
  <th align="left">输出</th>
</tr>
</thead>
<tbody>
<tr><td><b>遥操作</b></td><td>人类 demonstrations</td><td>原始机器人 episodes</td></tr>
<tr><td><b>数据准备</b></td><td>原始 teleop 数据和 rollout 数据</td><td>训练 zarr 数据集</td></tr>
<tr><td><b>Offline RL</b></td><td>合并后的离线数据集</td><td>改进后的 policy checkpoint</td></tr>
<tr><td><b>真实机器人 rollout</b></td><td>Offline RL checkpoint</td><td>新的 policy rollout 数据集</td></tr>
<tr><td><b>迭代离线数据集合并</b></td><td>Base zarr 和新的 rollout h5 文件</td><td>下一轮 zarr 数据集</td></tr>
<tr><td><b>Online RL</b></td><td>Offline RL checkpoint 和 live rollouts</td><td>最终部署 checkpoint</td></tr>
</tbody>
</table>

```text
human teleoperation
   └─▶ raw teleop data
        └─▶ processed offline dataset
             └─▶ BC / offline RL training
                  └─▶ real-robot policy rollout
                       └─▶ rollout dataset
                            └─▶ merge with previous offline data
                                 └─▶ next offline RL round
                                      └─▶ online RL fine-tuning
```

### 真实机器人设置

本节记录真实机器人部署所需的硬件和运行时假设：

- 机器人平台和控制器要求。
- 相机和点云设置。
- 末端执行器 / 灵巧手 / 夹爪设置。
- 工作空间限制和 reset protocol。
- 运行学习策略前的安全检查。

### 遥操作数据采集

遥操作和数据处理工具位于：

```text
tools/teleop_off2off_data/
```

计划工作流为：

```bash
cd tools/teleop_off2off_data

# 采集遥操作数据
python teleop.py

# 如果机器人设置需要，也可以使用 swapped teleoperation 入口
python teleop_swapped.py
```

> 硬件相关命令行参数应记录在对应机器人设置旁边。

### 数据准备

数据准备入口为：

```bash
cd tools/teleop_off2off_data
python data_prepare.py --config configs/data_prepare.yaml
```

完整遥操作和迭代离线数据集工作流见 [`tools/teleop_off2off_data/DATA_PREPARE.md`](tools/teleop_off2off_data/DATA_PREPARE.md)。

该工具支持三种模式：

<table width="100%">
<thead>
<tr>
  <th align="left">模式</th>
  <th align="left">用途</th>
</tr>
</thead>
<tbody>
<tr><td><code>raw_to_npy</code></td><td>把原始遥操作 episodes 转成处理后的 <code>.npy</code> 数据。</td></tr>
<tr><td><code>build_zarr</code></td><td>从处理后的 teleop 数据和选定 rollout sources 构建新的训练 zarr。</td></tr>
<tr><td><code>extend_zarr</code></td><td>向 base zarr 追加新的 rollout sources，并写出一个新的 zarr，不原地修改 base 数据集。</td></tr>
</tbody>
</table>

### 迭代 Offline RL 数据飞轮

迭代离线训练是真实机器人流水线使用的主要数据飞轮。由 offline RL 训练出的策略会在机器人上 rollout，生成的轨迹会作为 rollout 数据集保存，下一轮 offline RL 则在合并后的数据集上训练。

真实机器人上采集的 policy rollouts 在 YAML 中作为命名数据源：

```yaml
rollout_sources:
  - name: off2off_004
    round: "004"
    enabled: true
    type: h5_rollout_dir
    path: /path/to/policy_rollouts/004/online_ft
```

**典型用法：**

```bash
# 从 teleop 数据和所有启用的 rollout sources 构建数据集。
python data_prepare.py --config configs/data_prepare.yaml --mode build_zarr --include-rollouts

# 用一个新的 rollout source 扩展已有 zarr。
python data_prepare.py \
  --config configs/data_prepare.yaml \
  --mode extend_zarr \
  --rollout-source off2off_004 \
  --base-zarr-path /path/to/base.zarr \
  --zarr-output-path /path/to/new.zarr
```

合并输出会保持 RL-100 训练代码期望的 schema：

```text
data/
├── point_cloud, next_point_cloud
├── state, next_state
├── action, next_action
└── reward, return, done, timeout
meta/
└── episode_ends
```

### 在真实机器人数据集上训练

生成 zarr 数据集后，将 RL-100 task config 指向该数据集路径，然后运行所选 offline RL 脚本。

本节预留给具体真实机器人训练 recipe：

- 真实机器人 task config 示例。
- 3D 点云 offline 训练命令。
- 2D 图像 offline 训练命令。
- 推荐 checkpoint 选择和评估流程。

### 真实机器人 Rollout 采集

迭代离线 rollout 采集会复用 online training launcher 作为真实机器人数据采集 wrapper。实践中，加载 offline-RL 微调后的 policy checkpoint，进入 online 脚本，并在 online 微调前或过程中调用 evaluation 来让策略在机器人上 rollout，同时保存 `.h5` 轨迹。

真实机器人 offline 和 online 训练 bash 脚本与[第 2 节](#2-rl-后训练框架)介绍的 launcher 相同。迭代离线 rollout 采集唯一额外设置是，在采集运行中启用 `data_collect=True`。

Offline 和 online launchers 必须描述同一个 policy network，checkpoint 权重才能正确加载。需要保持 offline 训练 bash 和 rollout-collection bash 中的架构相关选项一致，包括 policy family、观测模态、chunk/single-action 设置、`policy.model`、`policy.encoder_type`、`policy.use_vib`、`policy.use_recon`、`horizon`、`n_action_steps`、`n_obs_steps`、encoder 输出维度，以及 diffusion/flow scheduler 设置。如果 online bash 修改这些字段，offline checkpoint 可能只会部分加载，或者在静默的网络不匹配状态下运行。

对于 RL 数据采集，rollout policy 应该是随机的，而不是确定性的 evaluation policy。在用于采集的 bash/config 路径中设置 `data_collect=True`。这会启用两个噪声来源：

- **Denoising/SDE action noise：** 当 `data_collect=True` 时，真实机器人 runner 会切换到 `deterministic=False`，因此 `policy.predict_action(...)` 会采样随机 denoising transitions。
- **VIB latent noise：** 当 `policy.use_vib=True` 且 `ppo.force_stochastic_online=True` 时，`train_real.py` 会在 `eval(..., data_collect=True)` 中保持 observation encoder 随机。

典型真实机器人采集设置：

```bash
data_collect=True
policy.use_vib=True
ppo.force_stochastic_online=True
```

使用 `offline_cp_timestamp` / `offline_cp_timestep` 或 online launcher 中对应的 checkpoint-loading 字段，指向所选 offline RL checkpoint。生成的 rollout `.h5` 文件随后应添加为 `tools/teleop_off2off_data/configs/data_prepare.yaml` 中的新 `rollout_sources` 条目。

### Online 真实机器人微调

本节覆盖最终 online RL 阶段：

- 启动真实 online RL 微调前设置 `data_collect=False`。`data_collect=True` 只用于迭代离线 rollout 数据集采集；如果 online 微调时忘记关闭，周期性 eval 会以采集模式运行并写出 rollout `.h5` 文件。
- 从 BC 或 offline RL checkpoint 开始。
- 在真实机器人上采集短 online rollouts。
- 执行 online policy-gradient 更新。
- 评估并保存部署 checkpoint。

### 安全注意事项

> 在硬件上运行策略前：

- 验证工作空间边界和 action scaling。
- 测试控制器延迟和急停。
- 从低速和保守 action 限制开始。
- 在自主长时间运行前，先进行短时有人监督 rollout。
- **不要**在验证 observation normalization、action 维度和 reset 行为前，把 checkpoint 部署到硬件上。

---

## 安装

完整环境配置见 [`INSTALL.md`](INSTALL.md)。我们建议使用 `adroit_door_medium` 作为第一个 smoke-test 任务。

**已验证服务器设置使用：**

<table width="100%">
<thead>
<tr>
  <th align="left">组件</th>
  <th align="left">版本</th>
</tr>
</thead>
<tbody>
<tr><td>NVIDIA driver</td><td><code>550.54.15</code></td></tr>
<tr><td>System CUDA</td><td><code>12.4</code></td></tr>
<tr><td>Python</td><td><code>3.8.20</code></td></tr>
<tr><td>PyTorch</td><td><code>2.4.0+cu121</code></td></tr>
</tbody>
</table>

**最小工作流：**

```bash
conda create -n rl100 --clone dp3 -y
conda activate rl100

export REPO_ROOT=$(pwd)
export PYTHONPATH=${REPO_ROOT}/RL-100:${PYTHONPATH}
export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python -m pip install -e third_party/dexart-release
python -m pip install -e third_party/gym-0.21.0
python -m pip install -e third_party/Metaworld
python -m pip install -e third_party/rrl-dependencies/mj_envs/.
python -m pip install -e third_party/rrl-dependencies/mjrl/.
python -m pip install -e third_party/mujoco-py-2.1.2.14
python -m pip install -e third_party/pytorch3d_simplified
python -m pip install -e visualizer
```

**Sanity check：**

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import rl_100, zarr, hydra, einops, metaworld, mujoco_py, open3d as o3d; print('imports ok', o3d.__version__)"
```

**已验证的 flow online distillation 入口：**

```bash
./scripts/Flow/Online/3D/train_policy_online_flow_distill_online.sh rl100 adroit_door_medium 0112 100 8
```

---

## 引用

如果你觉得这个工作有用，请引用：

```bibtex
@article{rl100,
  title  = {RL-100: Performant Robotic Manipulation with Real-World Reinforcement Learning},
  author = {Lei, Kun and Li, Huanyu and Yu, Dongjie and Wei, Zhenyu and Guo, Lingxiao and Jiang, Zhennan and Wang, Ziyu and Liang, Shiyu and Xu, Huazhe},
  journal = {arXiv preprint arXiv:2510.14830},
  year   = {2025}
}
```

<details>
<summary><b>也请考虑引用本项目使用的相关 RL 后训练方法</b></summary>

```bibtex
@inproceedings{lei2024unio,
  title     = {Uni-O4: Unifying Online and Offline Deep Reinforcement Learning with Multi-Step On-Policy Optimization},
  author    = {Kun LEI and Zhengmao He and Chenhao Lu and Kaizhe Hu and Yang Gao and Huazhe Xu},
  booktitle = {The Twelfth International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=tbFBh3LMKi}
}

@inproceedings{zhuang2023behavior,
  title     = {Behavior Proximal Policy Optimization},
  author    = {Zifeng Zhuang and Kun LEI and Jinxin Liu and Donglin Wang and Yilang Guo},
  booktitle = {The Eleventh International Conference on Learning Representations},
  year      = {2023},
  url       = {https://openreview.net/forum?id=3c13LptpIph}
}
```

</details>

---

## 致谢

RL-100 建立在 **Uni-O4**、**BPPO**、**DP3**、**Diffusion Policy**、机器人操作环境和强化学习基础设施等前序工作的基础上。详细的第三方致谢和许可证说明会随对应代码和依赖维护。

---

## 许可证

本项目基于 [Apache License 2.0](LICENSE) 发布。

<div align="center">
<sub>由 RL-100 团队用心制作。</sub>
</div>
