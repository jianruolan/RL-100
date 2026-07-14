# RL-100 安装指南

本指南用于安装 RL-100 在仿真训练、迭代 offline RL、online RL 以及 flow/diffusion policy distillation 中使用的环境。

下方设置已在当前服务器上验证：

```text
NVIDIA driver: 550.54.15
System CUDA shown by nvidia-smi: 12.4
Python: 3.8.20
PyTorch: 2.4.0+cu121
torch.version.cuda: 12.1
```

驱动支持 CUDA 12.4，而已验证的 PyTorch wheel 是 CUDA 12.1 构建。这是正常情况：CUDA 12.1 的 PyTorch wheel 可以在 CUDA 12.4 驱动上正确运行。

## 1. 创建环境

### 当前服务器推荐方式

已有的 `dp3` 环境已知可以运行本仓库。已验证的 `rl100` 环境是通过克隆该环境，并把 editable packages 重新安装到当前仓库路径来创建的：

```bash
conda create -n rl100 --clone dp3 -y
conda activate rl100
```

### 从零安装

在一台干净机器上，先使用 Python 3.8 和已验证的 PyTorch 版本：

```bash
conda create -n rl100 python=3.8 -y
conda activate rl100

python -m pip install "setuptools==59.5.0" wheel
python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

然后安装 RL-100 使用的 Python 包集合：

```bash
python -m pip install \
  zarr==2.12.0 wandb==0.20.1 ipdb==0.13.13 gpustat==1.1.1 \
  dm_control==1.0.23 omegaconf==2.3.0 hydra-core==1.2.0 dill==0.3.5.1 \
  einops==0.8.1 diffusers==0.33.1 huggingface_hub==0.33.1 numba==0.56.4 \
  moviepy==1.0.3 imageio==2.35.1 av==12.3.0 matplotlib==3.7.5 termcolor==2.4.0 \
  open3d==0.19.0 opencv-python==4.11.0.86 scipy==1.10.1 scikit-learn==1.3.2 \
  scikit-image==0.21.0 pandas==2.0.3 h5py==3.11.0 trimesh==4.6.12 \
  pybullet==3.2.7 plyfile==1.0.3 transforms3d==0.4.2 shapely==2.0.7 rtree==1.3.0 \
  tensorboardx==2.6.2.2 tqdm==4.67.1 colorlog==6.9.0 tabulate==0.9.0 \
  gdown==5.2.0 ftfy==6.2.3 regex==2024.11.6 yacs==0.1.8 \
  fvcore==0.1.5.post20221221 iopath==0.1.10 blessed==1.21.0 wcwidth==0.2.13 \
  timm==1.0.26 transformers==4.40.0
```

如果只跑仿真，真实机器人和相机工具是可选的：

```bash
python -m pip install \
  ur_rtde==1.6.1 atomics==1.0.3 viser==0.2.23 pyrealsense2 \
  dt_apriltags fpsample zerorpc yapf==0.43.0 littleutils==0.2.4 \
  sorcery==0.2.2 wrapt==1.17.2 diffusers==0.33.1 \
  "timm>=0.9.0" "torchvision>=0.15.0" "einops>=0.6.0"
```

可选包集合为：

```text
ur_rtde==1.6.1
atomics==1.0.3
viser==0.2.23
pyrealsense2
dt_apriltags
fpsample
zerorpc
yapf
# ViT encoder dependencies
timm>=0.9.0
torchvision>=0.15.0
einops>=0.6.0
yapf==0.43.0
littleutils==0.2.4
sorcery==0.2.2
wrapt==1.17.2
diffusers==0.33.1
```

部分真实机器人包，例如 `pyrealsense2`、`ur_rtde` 和硬件 wrapper，可能要求匹配的硬件/操作系统支持。如果你只运行模拟的 Adroit/DexArt/MetaWorld 训练，无法安装的硬件包可以单独处理。

## 2. 设置运行时路径

本仓库中的 `RL-100/` 不是一个可通过 pip 安装的 Python 包，因为它没有 `setup.py` 或 `pyproject.toml`。请使用 `PYTHONPATH`：

```bash
export REPO_ROOT=/path/to/RL-100-repo
export PYTHONPATH=${REPO_ROOT}/RL-100:${PYTHONPATH}
```

对于 MuJoCo/EGL 渲染：

```bash
export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Online 脚本会基于选中的 GPU 设置 `CUDA_VISIBLE_DEVICES`、`MUJOCO_EGL_DEVICE_ID` 和 `EGL_DEVICE_ID`。

## 3. 安装 MuJoCo

将 MuJoCo 2.1 安装到 `~/.mujoco`：

```bash
mkdir -p ~/.mujoco
cd ~/.mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz -O mujoco210.tar.gz --no-check-certificate
tar -xvzf mujoco210.tar.gz
```

在干净的 Ubuntu 机器上，`mujoco-py` 的 EGL 构建通常还需要：

```bash
sudo apt-get update
sudo apt-get install -y libglew-dev libgl1-mesa-dev libosmesa6-dev libglfw3 libglfw3-dev patchelf ninja-build
```

## 4. 安装仓库内 editable 依赖

从仓库根目录运行：

```bash
conda activate rl100

python -m pip install -e third_party/dexart-release
python -m pip install -e third_party/gym-0.21.0
python -m pip install -e third_party/Metaworld
python -m pip install -e third_party/rrl-dependencies/mj_envs/.
python -m pip install -e third_party/rrl-dependencies/mjrl/.
python -m pip install -e third_party/mujoco-py-2.1.2.14
python -m pip install -e third_party/pytorch3d_simplified
python -m pip install -e visualizer
```

重要说明：

- **不要**运行 `pip install -e RL-100`；`RL-100/` 通过 `PYTHONPATH` 导入。
- 当前 `third_party/r3m` 目录不包含可安装包。已验证服务器环境目前从外部 editable install 导入 `r3m`。对于干净 release，要么把 R3M 源码放入 `third_party/r3m`，要么在运行 image/R3M encoder 代码路径前安装等价的 `r3m` 包。
- `gym` 应解析到本仓库的 `third_party/gym-0.21.0`。

检查 editable 位置：

```bash
python -m pip show dexart gym metaworld mj-envs mjrl mujoco-py pytorch3d visualizer r3m
```

Editable project locations 应指向当前仓库，除非你有意从外部安装 `r3m`。

## 5. Assets 和数据

训练前，请确认所选任务需要的路径存在：

```text
third_party/dexart-release/assets       # DexArt assets
third_party/VRL3/ckpts                  # 如果生成 demos，需要 Adroit expert checkpoints
RL-100/data/*.zarr                      # offline datasets
```

对于 `adroit_door_medium` 等 Adroit medium 任务，数据集路径配置在：

```text
RL-100/rl_100/config/task/adroit_door_medium.yaml
```

## 6. Sanity Checks

从仓库根目录运行：

```bash
conda activate rl100
export REPO_ROOT=$(pwd)
export PYTHONPATH=${REPO_ROOT}/RL-100:${PYTHONPATH}
export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import rl_100, zarr, hydra, einops, metaworld, mujoco_py, open3d as o3d; print('imports ok', o3d.__version__)"
python -c "import gym; print('gym version:', gym.__version__)"
```

预期输出：

```text
torch 2.4.0+cu121, torch.version.cuda 12.1, cuda available True
imports ok 0.19.0
gym version: 0.21.0
```

## 7. 已验证的 Flow Online Distillation 运行

Flow online distillation launcher 是：

```text
scripts/Flow/Online/3D/train_policy_online_flow_distill_online.sh
```

在 `rl100` 环境中使用这个标准示例命令：

```bash
conda activate rl100
export REPO_ROOT=$(pwd)
export PYTHONPATH=${REPO_ROOT}/RL-100:${PYTHONPATH}
export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

./scripts/Flow/Online/3D/train_policy_online_flow_distill_online.sh rl100 adroit_door_medium 0112 100 8
```

它会展开为带有以下关键覆盖项的 `train.py --config-name=rl100_3d_flow.yaml`：

```text
task=adroit_door_medium
training.seed=100
++ppo.train_env_num=8
++ppo.eval_env_num=8
policy.scheduler_type=flow
distill_phase=online
flow_distill_inference_steps=1
flow_distill_teacher_steps=10
```

PG / PG + IDQL 风格提取和一步蒸馏是 launcher 级别的 sweep 设置。对于 offline distillation，先完成 offline sweep，再设置 `distill_phase='after_offline'`；对于 online distillation，设置 `distill_phase='online'`。

该 launcher 路径已在 `rl100` 环境中验证：它可以进入 `python train.py` 训练进程，并成功初始化 MuJoCo/EGL。由于它是较长的 online 训练任务（`ppo.max_train_steps=1000000`，`training.num_epochs=200`），验证运行在确认训练入口进程已启动后停止。

## 8. 常见问题

### 混合 editable installs

如果同一台机器上有多个本仓库副本，过期的 editable installs 可能会静默地从错误路径导入代码。

使用：

```bash
python -m pip show dexart gym metaworld mj-envs mjrl mujoco-py pytorch3d visualizer r3m
```

如有需要，从当前仓库重新安装 editable packages。

### `ModuleNotFoundError: rl_100`

设置：

```bash
export PYTHONPATH=$(pwd)/RL-100:${PYTHONPATH}
```

### `r3m` 导入问题

当前仓库不包含可安装的 `third_party/r3m` 包。使用基于 R3M 的图像 encoder 前，请先安装或补全 R3M。

### MuJoCo/OpenGL 错误

检查：

```bash
export LD_LIBRARY_PATH=${HOME}/.mujoco/mujoco210/bin:/usr/lib/nvidia:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```
