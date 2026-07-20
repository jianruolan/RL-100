# WandB 使用与排障指南

本文整理 RL-100 使用 WandB 记录 BC、critic、dynamics、offline RL 和 online RL 指标时的初始化、查看、续跑与常见问题。

## 1. 账号和终端登录

服务器终端凭据与浏览器登录状态彼此独立。终端登录成功不代表浏览器已经登录同一账号。

首次接手他人电脑时，应重新登录自己的账号：

```bash
conda activate rl100_test
wandb login --relogin
```

在浏览器登录自己的 WandB 账号后，从下面页面复制 API key：

```text
https://wandb.ai/authorize
```

终端凭据通常保存在：

```text
~/.netrc
```

检查当前账号：

```bash
wandb login
wandb status
```

项目配置中的 `logging.entity` 当前为 `null`，表示使用当前登录账号的默认 entity，不再绑定前一个使用者的 team。

## 2. Online 与 Offline 模式

实时上传：

```yaml
use_wandb: true
logging.mode: online
```

本地记录、稍后上传：

```yaml
use_wandb: true
logging.mode: offline
```

Offline run 上传示例：

```bash
wandb sync /path/to/wandb/offline-run-xxxx
```

Online 模式不需要启动类似 TensorBoard 的独立服务。先登录终端账号，再启动训练，浏览器直接打开 run URL。

## 3. Project、Group、Run 和 ID

RL-100 默认配置：

```yaml
logging:
  entity: null
  project: ${name}+${task_name}
  group: ${exp_name}
  name: ${training.seed}
  id: null
  resume: false
```

含义：

| 字段 | 含义 |
|---|---|
| entity | 用户或 team workspace |
| project | 项目容器 |
| group | 一组相关实验 |
| name | 网页显示的 run 名称 |
| id | WandB run 的唯一 ID |
| resume | 是否追加到已有 run |

训练代码设置了 `WANDB_SILENT=true` 以减少输出，但现在会显式打印：

```text
[WandB] run: ... (id=...)
[WandB] url: https://wandb.ai/...
```

应优先打开终端打印的完整 URL，不要手动猜 entity 或 project。

## 4. 模型 resume 与 WandB resume 不是一回事

```yaml
training.resume: true
```

只恢复模型、optimizer 和训练状态，不会自动恢复 WandB run。

如果仍为：

```yaml
logging.id: null
logging.resume: false
```

每次重启都会创建新的 WandB run。于是常见现象是：

```text
旧 run：BC、validation、Q/Value
新 run：dynamics、BPPO/offline RL
```

旧数据没有删除，只是位于另一个 run。

若希望追加到同一个 WandB run，需要使用原 run ID：

```yaml
logging.id: 原run_id
logging.resume: allow
```

Hydra 覆盖示例：

```bash
logging.id=qkk5sx8f logging.resume=allow
```

前提是 entity、project 相同，并且当前账号对旧 run 有权限。接手他人电脑后，通常不能把自己的数据追加到他人私有 run，此时保留多个 run 更安全。

## 5. 不同训练阶段的主要指标

### BC/Policy

```text
train_loss
val_loss
bc_loss
kl_loss
recon_loss
train_action_mse_error
lr
epoch
global_step
```

### Critic

```text
Q_loss
value_loss
```

### Dynamics

```text
dynamics_train_loss
dynamics_val_loss
dynamics_epoch
loss/dynamics_train_loss
loss/dynamics_holdout_loss
```

### Offline RL

```text
dpg_loss
current_mean_qs
current_bppo_scores
normal_eval_scores
ema_eval_scores
idql_eval_scores
```

没有真实/仿真 runner 时，环境成功率类指标不可用；`current_mean_qs` 只是模型内 OPE 信号。

### Online RL

根据配置可能出现：

```text
online ppo success rates
online ppo returns
online iql Q_loss
online iql value value_loss
online ema success rates
online ema returns
```

## 6. 网页显示为空的排查顺序

如果出现：

```text
This team has no public content
Looks like you stumbled on an empty page
```

依次检查：

1. 浏览器是否已登录；
2. 浏览器账号是否与 `wandb login` 显示的账号一致；
3. 是否打开了具体 project/run，而不是 team 的公开主页；
4. 终端打印的 entity、project、run ID 是否一致；
5. 训练是否在 WandB 初始化前已经报错；
6. 本地是否生成 `wandb/run-.../run-*.wandb`；
7. 该 run 是否属于前一个电脑使用者的私有 workspace。

查找本地 run：

```bash
find RL-100/data/outputs -type d \
  \( -name 'run-*' -o -name 'offline-run-*' \)
```

强制同步一个本地 online run 的示例：

```bash
wandb sync \
  --include-online \
  --entity <自己的entity> \
  --project '<项目名>' \
  --id <run-id> \
  /path/to/run-<id>.wandb
```

不要在文档、脚本或 Git 中保存 API key。

## 7. 多阶段实验的推荐组织方式

推荐二选一：

### 单 run 续写

适合希望在一张图上连续查看 BC、critic、dynamics、offline RL 的情况。续跑时固定 `logging.id` 并设置 `logging.resume=allow`。

### 分 run 记录

分别建立：

```text
阶段1-BC
阶段2-Critic
阶段3-Dynamics
阶段4-OfflineRL
```

再在 WandB 项目页选择多个 run 比较。该方式更容易避免 step 冲突，也更适合不同账号或不同机器迁移。

