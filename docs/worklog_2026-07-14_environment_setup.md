# RL-100 ARM 环境配置与问题排查记录（2026-07-14）

## 1. 当前环境概况

- 系统架构：`aarch64`
- Conda 环境：`rl100_test`
- Python：`3.10`
- 项目目录：`/home/mtarch/Desktop/zyf/RL-100`
- 主要复现任务：`adroit_door_medium`
- 训练入口：

```bash
bash scripts/Diffusion/Offline/3D/train_policy.sh \
  rl100 adroit_door_medium 0112 100
```

## 2. MuJoCo：现代 API 与 legacy API 的区别

环境中已经安装并验证了现代 MuJoCo：

```bash
python -c "import mujoco; print('mujoco import ok')"
python -c "import dm_control; print('dm_control import ok')"
```

但 RL-100 的 Adroit 路径依赖旧版：

```text
Gym 0.21
  -> mj_envs / mjrl
  -> mujoco_py==2.1.2.14
  -> legacy MuJoCo native library
```

`pip install mujoco` 提供的是现代 Python binding，不能代替 `mujoco_py` 需要的 native MuJoCo 目录。

## 3. ARM native MuJoCo 安装

### 3.1 版本选择

官方 MuJoCo 2.1.0 没有合适的 Linux aarch64 release。测试过：

- MuJoCo 2.2.1 aarch64：可以下载，但 API 已与 `mujoco-py` 2.1 不兼容。
- MuJoCo 2.1.1 aarch64：头文件和接口更接近 `mujoco-py`，最终采用。

下载文件：

```text
/tmp/mujoco-2.1.1-linux-aarch64.tar.gz
```

安装目录：

```text
/home/mtarch/.mujoco/mujoco210
```

2.2.1 备份目录：

```text
/home/mtarch/.mujoco/mujoco221_backup
```

### 3.2 `mujoco-py` 兼容链接

MuJoCo 2.1.1 的 ARM release 使用：

```text
lib/libmujoco.so.2.1.1
lib/libglewegl.so
```

而 `mujoco-py` 会从 `mujoco210/bin` 查找旧库名。因此增加了兼容链接：

```text
bin/libmujoco210.so -> ../lib/libmujoco.so.2.1.1
bin/libmujoco.so    -> ../lib/libmujoco.so.2.1.1
bin/libglew.so      -> ../lib/libglew.so
bin/libglewegl.so   -> ../lib/libglewegl.so
bin/libglewosmesa.so -> ../lib/libglewosmesa.so
```

### 3.3 构建工具

`mujoco-py` 与 Cython 3 不兼容，已将 Cython 固定为：

```text
Cython==0.29.37
```

同时安装：

```text
patchelf==0.17.2
```

最终验证：

```bash
export MUJOCO_PY_MUJOCO_PATH="$HOME/.mujoco/mujoco210"
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:$HOME/.mujoco/mujoco210/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"

python -c "import mujoco_py; print('mujoco_py ok')"
```

## 4. 训练脚本中的 MuJoCo 环境变量

已在 `scripts/Diffusion/Offline/3D/train_policy.sh` 中加入：

```bash
export MUJOCO_PY_MUJOCO_PATH="${MUJOCO_PY_MUJOCO_PATH:-$HOME/.mujoco/mujoco210}"
export LD_LIBRARY_PATH="${MUJOCO_PY_MUJOCO_PATH}/bin:${MUJOCO_PY_MUJOCO_PATH}/lib:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
```

这避免了每次打开终端后忘记设置 native library 路径。

## 5. 数据集路径问题

任务配置声明：

```yaml
zarr_path: data/adroit_door_medium.zarr
```

真实数据位置：

```text
/home/mtarch/Desktop/zyf/RL-100/RL-100/data/adroit_door_medium.zarr
```

但 `RL-100/train.py` 使用三级 `parent` 并执行 `os.chdir(ROOT_DIR)`，运行时工作目录变成：

```text
/home/mtarch/Desktop/zyf
```

因此相对路径被解析为：

```text
/home/mtarch/Desktop/zyf/data/adroit_door_medium.zarr
```

当前为了先跑通训练，在外层 `data` 中创建了数据集软链接。这个方式能工作，但不是长期最佳方案。后续应统一项目根目录计算或将数据路径改为基于配置文件/项目根目录解析的绝对路径。

Zarr 已验证可读取，包含：

```text
state, action, point_cloud, img,
next_state, next_action, next_point_cloud, next_img,
reward, done, timeout, return
```

## 6. 非当前任务依赖导致的导入失败

### 6.1 DexArt/SAPIEN

运行 Adroit 时曾出现：

```text
ModuleNotFoundError: No module named 'dexart'
```

原因不是 Adroit 需要 DexArt，而是 `rl_100.env.__init__` 在包导入时无条件加载 DexArt、UR5、Franka 等所有环境。ARM 下 `sapien==2.2.1` 没有可用 wheel，不能简单安装解决。

已将 `RL-100/rl_100/env/__init__.py` 改为 PEP 562 风格延迟导入：只有真正访问 `DexArtEnv`、`UR5Env` 等名称时才加载对应模块。已验证：

```text
AdroitEnv lazy import ok
```

### 6.2 Stable-Baselines3 混用

曾出现：

```text
cannot import name 'StackedDictObservations'
```

原因是项目 vendored 的 `rl_100.stable_baselines3` 进入 `vec_env` 后，又通过绝对导入跳到环境里的 `stable_baselines3==2.9.0`。项目环境仍使用旧 Gym 四返回值协议，不能直接改用 SB3 2.9 的 Gymnasium vector env。

已处理：

- `vec_env/__init__.py` 只导出项目需要的本地类。
- `subproc_vec_env.py` 对 `base_vec_env` 使用相对导入。

验证结果：

```text
AdroitRunner import ok
rl_100.stable_baselines3.common.vec_env.subproc_vec_env
```

## 7. 点云段错误定位与解决

### 7.1 原始现象

训练在首次评估时出现：

```text
Eval in Adroit door Pointcloud Env: 0/30
Found 5 GPUs for rendering. Using device 0.
Segmentation fault (core dumped)
```

### 7.2 分阶段测试

依次测试结果：

| 阶段 | 结果 |
|---|---|
| `import mujoco_py` | 通过 |
| `gym.make('door-v0')` | 通过 |
| 原始环境 `reset()` | 通过 |
| MuJoCo dynamics `sim.step()` | 通过 |
| EGL RGB render | 通过 |
| EGL RGB + depth render | 通过 |
| `AdroitEnv.reset()` | 通过 |
| Open3D `Image` | 通过 |
| Open3D 相机内参 | 通过 |
| `PointCloud.create_from_depth_image()` | 段错误 |

用纯合成深度图、不经过 MuJoCo 也能复现同一个段错误。因此根因不是 EGL 或 MuJoCo，而是 Open3D ARM native 扩展与 NumPy ABI 的组合。

### 7.3 最终版本组合

PyPI 对当前 aarch64/Python 3.10 没有 Open3D 0.19.0 wheel，可用最高版本为 0.18.0。最终使用：

```text
open3d==0.18.0 (aarch64 wheel)
numpy==1.26.4
opencv-python==4.11.0.86
```

原来的组合：

```text
open3d==0.17.0/0.18.0
numpy==2.2.6
```

会在 `create_from_depth_image()` 中直接段错误。

最终回归：

```text
Open3D synthetic cloud: (7056, 3)
Adroit full reset point_cloud: (512, 6)
RGB image: (3, 84, 84)
depth: (84, 84)
```

## 8. GPU 检测遗留问题

`scripts/find_gpu.sh` 使用：

```bash
nvidia-smi --query-gpu=index,memory.used
```

当前 ARM 平台返回 `[N/A]`，导致 shell 算术比较报错：

```text
((: [N/A]: syntax error
```

脚本随后默认使用 GPU 0，因此目前不是致命问题。后续应让脚本过滤非数字值，或在单 GPU ARM 机器上直接允许通过环境变量指定 GPU。

## 9. WandB 监控

训练脚本当前固定：

```bash
wandb_mode=offline
```

本地日志目录：

```text
/home/mtarch/Desktop/zyf/data/outputs_vib_ablation/
adroit_door_medium-rl100-0112_seed100/relu/dp3vib/skipnet/wandb/
```

上传某个完成的 run：

```bash
wandb login
wandb sync /path/to/offline-run-xxxxxxxx_xxxxxx-xxxxxxxx
```

不要随意使用 `wandb sync --sync-all`，因为目录中存在多次失败和超参数循环产生的 run。若要实时网页监控，应在启动训练前将 `wandb_mode` 改为 `online`。

## 10. 当前已知依赖风险

`rl100_test` 中还混有 LeRobot、Rerun、CMEEL 等现代包，它们部分要求 NumPy 2.x；当前 RL-100 Adroit 路径需要 NumPy 1.26.4 以兼容 Open3D ARM wheel。

这些冲突目前不影响已经验证的 Adroit 训练路径，但说明该环境不适合同时承担所有机器人项目。后续更稳妥的做法是为 RL-100 单独维护一份明确锁定版本的 conda 环境。

## 11. 当前验证状态

已经验证：

- `mujoco_py` 导入成功。
- Adroit `door-v0` 创建、reset 和 dynamics step 成功。
- EGL RGB/depth 渲染成功。
- Open3D 深度转点云成功。
- PyTorch3D FPS 下采样成功。
- `AdroitRunner` 导入成功。
- `adroit_door_medium` 数据集读取成功。
- BC 训练和仿真评估已经可以正常启动。

