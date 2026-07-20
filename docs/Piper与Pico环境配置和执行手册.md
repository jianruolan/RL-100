# Piper + Pico + Isaac Teleop 环境配置与执行 Runbook

本文是一份按顺序执行的操作手册，目标是在当前 NVIDIA Jetson AGX Thor 主机上：

```text
安装 Isaac Teleop
  -> Pico 连接 CloudXR
  -> 验证左右控制器数据
  -> 安装并验证 Piper SDK
  -> 配置双 CAN
  -> 单臂低速运动
  -> Pico 单臂直控
  -> Pico 双臂直控
  -> 采集真实机器人数据
```

本阶段不启动 Isaac Sim，不配置 Isaac Lab，也暂时不处理三相机内外参和点云标定。相机只在最后保留采集接口位置。

## 0. 当前环境和原则

本机已确认：

```text
Host: NVIDIA Jetson AGX Thor Developer Kit
Architecture: aarch64
OS: Ubuntu 24.04.4 LTS
Python: 3.12.12
CUDA: 13.0
L4T: R38.2.2
Isaac Teleop checkout: /home/mtarch/Desktop/zyf/IsaacTeleop
Piper SDK checkout: /home/mtarch/Desktop/zyf/piper_sdk
RL-100 checkout: /home/mtarch/Desktop/zyf/RL-100
```

当前尚未安装：

```text
isaacteleop
rclpy
```

本方案第一版不使用 ROS2，因此不要求安装 `rclpy`。

### 0.1 安全原则

在开始前确认：

- 两台 Piper 都有可触达的物理急停。
- 机械臂下方和周围没有人或易损设备。
- 机械臂失电下落方向没有障碍物。
- 首次运动只启用一条手臂。
- 首次运动速度比例不超过 10%-20%。
- 未验证坐标轴前不同时启用双臂。
- 不直接运行 `piper_ctrl_go_zero_dual.py`。
- 不把 `ResetPiper()` 当作普通停止；该调用会使机械臂立即失电下落。

### 0.2 设置路径变量

每个新终端先执行：

```bash
export WS=/home/mtarch/Desktop/zyf
export ISAAC_TELEOP_ROOT=${WS}/IsaacTeleop
export PIPER_SDK_ROOT=${WS}/piper_sdk
export RL100_ROOT=${WS}/RL-100
```

检查：

```bash
test -d "$ISAAC_TELEOP_ROOT" && echo "IsaacTeleop found"
test -d "$PIPER_SDK_ROOT" && echo "piper_sdk found"
test -d "$RL100_ROOT" && echo "RL-100 found"
```

## 1. 主机基础检查

### Step 1.1：检查系统

```bash
uname -m
lsb_release -ds
python3 --version
nvcc --version
cat /proc/device-tree/model
cat /etc/nv_tegra_release
```

预期关键结果：

```text
aarch64
Ubuntu 24.04
Python 3.12
CUDA 13.0
Jetson AGX Thor
```

Jetson 上 `nvidia-smi` 失败不能单独说明 GPU 不可用，主要检查 CUDA、L4T 和后续 CloudXR 实际启动结果。

### Step 1.2：检查存储和网络

```bash
df -h /
ip -brief address
ip route
```

建议至少预留：

```text
Isaac Teleop 构建和依赖：20 GB
CloudXR 缓存和日志：5 GB
首轮真实数据：100 GB 以上
```

Pico 和 Thor 建议连接同一个 5 GHz/6 GHz 局域网。记录 Thor 的局域网 IP：

```bash
hostname -I
```

后面记为：

```text
<THOR_IP>
```

## 2. 安装系统依赖

### Step 2.1：安装 Isaac Teleop 构建依赖

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git pkg-config swig \
  libx11-dev clang-format-14 ccache patchelf \
  libvulkan1 libgl1-mesa-dev libegl1-mesa-dev \
  android-tools-adb coturn curl
```

检查 CMake：

```bash
cmake --version
```

要求 CMake 3.20 或更高。

### Step 2.2：安装 Piper/CAN 系统依赖

```bash
sudo apt-get install -y can-utils ethtool iproute2
```

检查：

```bash
ip -brief link
which candump
which ethtool
```

### Step 2.3：安装 uv

Isaac Teleop 本地构建系统使用 `uv`：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
uv --version
```

如果 `source ~/.local/bin/env` 不存在，则执行：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## 3. 创建独立 Python 环境

不要把 Isaac Teleop 安装进 `rl100_test`。两个项目依赖差异较大，应隔离环境。

### Step 3.1：创建环境

```bash
uv python install 3.12
uv venv --python 3.12 "$HOME/venvs/piper_isaac_teleop"
source "$HOME/venvs/piper_isaac_teleop/bin/activate"
```

检查：

```bash
which python
python --version
```

预期：

```text
/home/mtarch/venvs/piper_isaac_teleop/bin/python
Python 3.12.x
```

### Step 3.2：更新基础工具

```bash
python -m pip install -U pip setuptools wheel
```

后续每个 Isaac Teleop/Piper 遥操作终端都先执行：

```bash
source "$HOME/venvs/piper_isaac_teleop/bin/activate"
```

## 4. 构建并安装 Isaac Teleop

当前本地 checkout 是 `1.4.x`，ARM 主机优先使用本地源码构建，避免将 checkout 代码和另一个 pip 版本混合。

### Step 4.1：确认源码版本和工作区

```bash
cd "$ISAAC_TELEOP_ROOT"
cat VERSION
git log -1 --oneline
```

不要在构建前执行 `git clean` 或删除本地修改。

### Step 4.2：配置构建

第一版不需要 Isaac Sim、ROS2、Televiz 或额外硬件插件，使用最小构建：

```bash
cd "$ISAAC_TELEOP_ROOT"

cmake -B build -S . \
  -DCMAKE_BUILD_TYPE=Release \
  -DISAAC_TELEOP_PYTHON_VERSION=3.12 \
  -DBUILD_TESTING=OFF \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_VIZ=OFF \
  -DBUILD_PLUGIN_OAK_CAMERA=OFF \
  -DENABLE_CLANG_FORMAT_CHECK=OFF
```

这一步需要网络，用于 CMake FetchContent 和 CloudXR SDK 获取。

若配置成功，应生成：

```text
IsaacTeleop/build/
```

### Step 4.3：编译和安装

Thor 核数较多，但首次建议限制并行度，避免内存峰值：

```bash
cmake --build build --parallel 8
cmake --install build
```

预期生成：

```text
IsaacTeleop/install/
IsaacTeleop/install/wheels/isaacteleop-*.whl
```

检查：

```bash
find install/wheels -maxdepth 1 -name 'isaacteleop-*.whl' -print
```

### Step 4.4：安装本地 wheel 和运行依赖

仍在 `piper_isaac_teleop` 环境中：

```bash
cd "$ISAAC_TELEOP_ROOT"

python -m pip install \
  -r src/core/python/requirements.txt \
  -r src/core/python/requirements-cloudxr.txt \
  -r src/core/python/requirements-retargeters-lite.txt

python -m pip install --force-reinstall install/wheels/isaacteleop-*.whl
```

这里使用 `retargeters-lite`，不安装完整 dex-hand retargeting，因此不需要先构建 ARM `nlopt` wheel。Piper 二指夹爪和 controller pose 路线不需要完整 dex-hand 依赖。

### Step 4.5：验证导入

```bash
python -c "import isaacteleop; print('isaacteleop import ok')"
python -c "from isaacteleop.cloudxr import CloudXRLauncher; print('cloudxr import ok')"
python -c "from isaacteleop.retargeting_engine.deviceio_source_nodes import ControllersSource; print('controller source ok')"
```

Gate 4：三个命令必须全部成功。

### Step 4.6：pip 安装备选路线

只有本地源码构建失败、且 NVIDIA index 提供当前 ARM wheel 时才使用：

```bash
python -m pip install \
  "isaacteleop[cloudxr,retargeters-lite]~=1.3.131" \
  --extra-index-url https://pypi.nvidia.com
```

采用备选路线后，不要再通过 `PYTHONPATH` 导入本地 1.4.x checkout；一个环境只保留一套 Isaac Teleop 实现。

## 5. 启动 CloudXR 并连接 Pico

### Step 5.1：确认 Pico 条件

官方本地文档明确支持：

```text
Pico 4 Ultra
Pico OS 15.4.4U 或更新版本
Pico Browser 4.0.40 或更新版本（部分企业功能）
```

如果是其他 Pico 型号，也可以测试 WebXR，但应把兼容性视为未确认。

### Step 5.2：开放防火墙端口

```bash
sudo ufw allow 47998/udp
sudo ufw allow 49100,48322/tcp
sudo ufw status
```

### Step 5.3：启动 CloudXR

终端 A：

```bash
source "$HOME/venvs/piper_isaac_teleop/bin/activate"
cd "$ISAAC_TELEOP_ROOT"

python -m isaacteleop.cloudxr --accept-eula --host-client
```

第一次启动可能下载 runtime/Web Client 并写入：

```text
~/.cloudxr/
```

保持终端 A 运行。

### Step 5.4：Pico 浏览器连接

在 Pico 浏览器打开：

```text
https://<THOR_IP>:48322/client/
```

按页面流程：

1. 输入 `<THOR_IP>`。
2. 打开证书链接。
3. 接受自签名证书。
4. 返回 Web Client。
5. 点击 Connect。
6. 允许 VR/控制器权限。

建议先使用默认 Pico 配置：

```text
90 FPS
100 Mbps
auto-webrtc
```

### Step 5.5：加载 CloudXR 环境

终端 B：

```bash
source "$HOME/venvs/piper_isaac_teleop/bin/activate"
source "$HOME/.cloudxr/run/cloudxr.env"
cd "$ISAAC_TELEOP_ROOT"
```

检查：

```bash
cat "$HOME/.cloudxr/run/cloudxr.env"
```

### Step 5.6：测试 trigger

终端 B：

```bash
python examples/teleop/python/gripper_retargeting_example_simple.py
```

按左右 trigger，检查输出是否在 `0.0-1.0` 间连续变化。

### Step 5.7：测试 controller pose

```bash
python examples/teleop/python/se3_retargeting_example.py
```

交互选择：

```text
4. Relative Positioning (Controller Delta -> Delta)
```

移动右控制器，检查 `dPos` 和旋转量连续变化。

Gate 5：

- Pico 可以稳定连接至少 10 分钟。
- 左右控制器能被区分。
- trigger、squeeze、position、orientation 都更新。
- 控制器静止时数据没有明显跳变。
- 关闭 Pico 或遮挡 tracking 后，程序能观察到 invalid/disconnect。

未通过 Gate 5 时不要连接 Piper 控制代码。

## 6. 安装 Piper SDK

### Step 6.1：安装本地 Piper SDK

终端 C：

```bash
source "$HOME/venvs/piper_isaac_teleop/bin/activate"
cd "$PIPER_SDK_ROOT"

python -m pip install -e .
```

检查：

```bash
python -c "import piper_sdk; print('piper_sdk import ok')"
python -m pip show piper-sdk
```

本地 SDK 版本应为：

```text
0.6.1
```

### Step 6.2：仅检查 CAN 设备

先不要给机械臂发命令：

```bash
ip -brief link
ip -details link show type can
```

当前主机能看到 `can0-can3`，但不能仅凭编号判断哪两个连接 Piper。

逐个查看物理 USB 位置：

```bash
for iface in can0 can1 can2 can3; do
  echo "=== $iface ==="
  sudo ethtool -i "$iface" | grep bus-info
done
```

逐个拔插 Piper CAN 模块，记录：

```text
left Piper USB bus-info  -> <LEFT_USB_PORT>
right Piper USB bus-info -> <RIGHT_USB_PORT>
```

### Step 6.3：配置固定 CAN 名称

本地脚本：

```text
../piper_sdk/piper_sdk/can_muti_activate.sh
```

目前脚本内写的是：

```text
1-4.1:1.0 -> can_left
1-4.2:1.0 -> can_right
```

这只是当前文件里的示例/旧机器映射。必须替换为 Step 6.2 实际测得的 USB bus-info 后才能执行。

修改映射为：

```bash
USB_PORTS["<LEFT_USB_PORT>"]="can_left:1000000"
USB_PORTS["<RIGHT_USB_PORT>"]="can_right:1000000"
```

然后执行：

```bash
cd "$PIPER_SDK_ROOT"
bash piper_sdk/can_muti_activate.sh
```

检查：

```bash
ip -brief link show can_left
ip -brief link show can_right
ip -details link show can_left
ip -details link show can_right
```

两侧都应为：

```text
UP
bitrate 1000000
```

### Step 6.4：CAN 只读检查

机械臂上电后，分别观察 CAN 数据：

```bash
timeout 5 candump can_left
timeout 5 candump can_right
```

两侧都应有持续数据。若没有数据，先检查接线、供电、接口名和 bitrate，不运行控制 demo。

## 7. Piper 只读状态测试

现有 `piper_read_joint_state.py` 默认使用 `can0`，不适合直接验证已经重命名的双臂。建议下一步实现统一的只读脚本：

```text
tools/piper_isaac_teleop/read_piper_state.py
```

预期命令：

```bash
cd "$RL100_ROOT"
python tools/piper_isaac_teleop/read_piper_state.py --can can_left --duration 30
python tools/piper_isaac_teleop/read_piper_state.py --can can_right --duration 30
```

脚本应读取但不使能：

```text
GetArmStatus()
GetArmJointMsgs()
GetArmEndPoseMsgs()
GetArmGripperMsgs()
GetCanFps()
```

Gate 7：

- 左右臂反馈 Hz 持续非零。
- joint 单位从 `0.001 degree` 转为 rad 后合理。
- TCP 位置从 `0.001 mm` 转为 meter 后合理。
- gripper 从 `0.001 mm` 转为 meter 后合理。
- 两个接口不会读到同一条机械臂。
- 状态中没有 collision、limit、communication error。

## 8. 单臂安全运动测试

这一步需要先实现：

```text
tools/piper_isaac_teleop/test_cartesian_step.py
```

不要使用原始 `piper_ctrl_end_pose.py` 做首次运动，因为它包含无限循环、100% 速度和固定绝对目标。

### Step 8.1：dry-run

```bash
cd "$RL100_ROOT"

python tools/piper_isaac_teleop/test_cartesian_step.py \
  --can can_left \
  --axis x \
  --delta-mm 1 \
  --speed-percent 10 \
  --dry-run
```

dry-run 只打印：

```text
当前反馈 pose
目标 pose
单位转换结果
workspace/step limit 检查
```

### Step 8.2：实际执行 1 mm

旁站人员握住物理急停：

```bash
python tools/piper_isaac_teleop/test_cartesian_step.py \
  --can can_left \
  --axis x \
  --delta-mm 1 \
  --speed-percent 10 \
  --execute
```

依次验证：

```text
x +1 mm
x -1 mm
y +1 mm
y -1 mm
z +1 mm
z -1 mm
```

位置通过后再测试每次不超过 0.5-1 度的旋转。

### Step 8.3：右臂重复

```bash
python tools/piper_isaac_teleop/test_cartesian_step.py \
  --can can_right \
  --axis x \
  --delta-mm 1 \
  --speed-percent 10 \
  --dry-run
```

确认右臂 base frame 的每个方向，不要假设和左臂镜像。

Gate 8：

- 六个移动方向与预期一致。
- 单步没有突然跳跃。
- Ctrl+C 后不继续产生新目标。
- 超过单步、workspace 或 joint limit 的目标会被拒绝。
- `EmergencyStop(0x01)` 已在受控条件下验证。

## 9. 实现 Pico 到 Piper 的 bridge

在 Gate 5、7、8 全部通过后实现：

```text
tools/piper_isaac_teleop/
  isaac_controller_source.py
  piper_arm.py
  vr_mapper.py
  safety_filter.py
  teleoperate.py
  configs/pick_place.yaml
```

### Step 9.1：先做无机器人 replay/dry-run

预期命令：

```bash
source "$HOME/venvs/piper_isaac_teleop/bin/activate"
source "$HOME/.cloudxr/run/cloudxr.env"
cd "$RL100_ROOT"

python tools/piper_isaac_teleop/teleoperate.py \
  --arm left \
  --dry-run
```

验证输出：

```text
squeeze 松开 -> no command
squeeze 按下 -> anchor
控制器移动 -> 相对 TCP target
trigger -> gripper target
tracking invalid -> HOLD
```

### Step 9.2：Pico 单臂低速控制

CloudXR 保持在终端 A 运行。控制终端执行：

```bash
source "$HOME/venvs/piper_isaac_teleop/bin/activate"
source "$HOME/.cloudxr/run/cloudxr.env"
cd "$RL100_ROOT"

python tools/piper_isaac_teleop/teleoperate.py \
  --arm left \
  --can-left can_left \
  --control-hz 20 \
  --translation-scale 0.3 \
  --rotation-scale 0.3 \
  --speed-percent 10 \
  --execute
```

首次运行顺序：

1. 只启用平移。
2. 验证 clutch 松开和重新按下。
3. 开启旋转。
4. 最后开启 trigger 到 gripper。

Gate 9：

- 松开 squeeze 时保持不动。
- 重新 squeeze 不跳变。
- tracking loss 或 100 ms 无新数据时 hold。
- 单臂连续运行 20 个短 episode 无异常。

## 10. Pico 双臂控制

双臂模式需要先配置：

```text
左右独立 workspace
中央禁入区
TCP 最小距离
任一臂故障时双臂 hold
左右 controller active 检查
```

### Step 10.1：双臂 dry-run

```bash
python tools/piper_isaac_teleop/teleoperate.py \
  --arm dual \
  --can-left can_left \
  --can-right can_right \
  --dry-run
```

### Step 10.2：双臂外侧空间低速执行

```bash
python tools/piper_isaac_teleop/teleoperate.py \
  --arm dual \
  --can-left can_left \
  --can-right can_right \
  --control-hz 20 \
  --translation-scale 0.3 \
  --rotation-scale 0.3 \
  --speed-percent 10 \
  --execute
```

首先让两臂只在各自外侧区域移动，不抓物体、不进入中央协作区。

Gate 10：

- 左右手柄与左右臂没有交换。
- 单侧 clutch 不会驱动另一侧。
- 同时 clutch 可以稳定控制双臂。
- 任一侧 CAN/tracking/arm status 异常会触发配置的停止策略。
- 中央禁入区有效。

## 11. 安装数据记录依赖

仍使用 `piper_isaac_teleop` 环境：

```bash
python -m pip install \
  h5py zarr numcodecs pyyaml msgpack msgpack-numpy \
  numpy scipy opencv-python
```

如果三台相机是 RealSense，再安装：

```bash
python -m pip install pyrealsense2
```

本阶段只要求能打开设备、读取 RGB/depth 和时间戳，不进行标定和点云融合。

## 12. 数据采集脚本执行

需要实现：

```text
tools/piper_isaac_teleop/record.py
tools/piper_isaac_teleop/episode_recorder.py
```

### Step 12.1：无机器人记录 smoke test

```bash
python tools/piper_isaac_teleop/record.py \
  --arm left \
  --dry-run \
  --num-episodes 2 \
  --episode-time-s 10 \
  --output-dir data/piper_pick_place/raw_smoke
```

检查每条 episode 是否包含：

```text
Pico controller pose/buttons/valid/timestamp
unfiltered target
safety-filtered target
host monotonic timestamp
episode result
```

### Step 12.2：单臂真实采集 smoke test

```bash
python tools/piper_isaac_teleop/record.py \
  --arm left \
  --can-left can_left \
  --control-hz 20 \
  --transition-hz 10 \
  --num-episodes 5 \
  --episode-time-s 30 \
  --output-dir data/piper_pick_place/raw_left_v0 \
  --execute
```

控制键建议：

```text
n: 当前 episode 成功并保存
f: 当前 episode 失败并保存
r: 当前 episode 作废并重采
q: 完成当前安全停止后退出
```

### Step 12.3：验证原始数据

```bash
python tools/piper_data/validate_raw.py \
  --input data/piper_pick_place/raw_left_v0
```

必须检查：

```text
episode 可解析
无 NaN/Inf
时间戳严格递增
机器人反馈 Hz 和 XR Hz 合理
action 是实际下发命令
tracking invalid 区间没有继续生成动作
左右设备没有串号
```

### Step 12.4：导出 LeRobot

```bash
python tools/piper_data/export_lerobot.py \
  --input data/piper_pick_place/raw_left_v0 \
  --output data/piper_pick_place/lerobot_left_v0
```

### Step 12.5：导出 RL-100 Zarr

相机标定和点云流程暂时未完成时，可以先导出 state/action smoke 数据；正式 DP3 训练前再加入融合点云：

```bash
python tools/piper_data/export_rl100_zarr.py \
  --input data/piper_pick_place/raw_left_v0 \
  --output RL-100/data/piper_left_pick_place_v0.zarr \
  --without-point-cloud
```

注意：当前 RL-100 3D policy 仍要求 `point_cloud`。`--without-point-cloud` 只用于验证 episode、state/action 和转换程序，不是正式 DP3 训练数据。

## 13. 每次正式采集的终端布局

### 终端 A：CloudXR

```bash
source "$HOME/venvs/piper_isaac_teleop/bin/activate"
python -m isaacteleop.cloudxr --accept-eula --host-client
```

### 终端 B：CAN 和状态监控

```bash
source "$HOME/venvs/piper_isaac_teleop/bin/activate"
cd "$RL100_ROOT"
python tools/piper_isaac_teleop/read_piper_state.py --can can_left --watch
```

双臂阶段再开终端 C 监控 `can_right`。

### 终端 D：遥操作/记录

```bash
source "$HOME/venvs/piper_isaac_teleop/bin/activate"
source "$HOME/.cloudxr/run/cloudxr.env"
cd "$RL100_ROOT"

python tools/piper_isaac_teleop/record.py \
  --arm left \
  --can-left can_left \
  --output-dir data/piper_pick_place/raw_left_v0 \
  --execute
```

### 旁站人员

- 持有物理急停。
- 观察机械臂、夹爪和线缆。
- 负责成功/失败/作废标签。
- 发生异常时优先物理停止，不等待软件界面。

## 14. 故障分支

### CloudXR 无法在 Thor 启动

1. 保存 CloudXR 日志。
2. 检查 ARM wheel、Vulkan、L4T 和 runtime SDK 架构。
3. 尝试 `controller-only` 最小示例，不构建 Televiz/ROS2。
4. 仍失败则在 x86_64 RTX 工作站运行 Isaac Teleop。
5. 通过局域网把 controller pose 发送到 Thor。
6. Piper CAN、安全控制和数据记录继续留在 Thor。

### Pico 页面能打开但控制器无数据

检查：

```text
Pico 型号和 OS
浏览器 WebXR 权限
控制器是否激活
证书是否接受
47998/udp、49100/tcp、48322/tcp
Pico 和 Thor 是否同网段
```

### Piper 有 CAN 帧但 SDK Hz 为 0

检查：

```text
can_left/can_right 是否映射正确
bitrate 是否为 1000000
固件与 C_PiperInterface_V2 是否匹配
是否误处于 master/slave 硬件跟随模式
```

### Piper 突然朝错误方向移动

立即急停，不通过修改比例继续试。重新执行：

```text
单臂
1 mm
单轴
dry-run
坐标和单位审计
```

## 15. 完成标准

环境和直控链路完成需要满足：

- Isaac Teleop 在独立 Python 3.12 环境中可导入。
- CloudXR 在 Thor 上可稳定运行。
- Pico 左右控制器 pose/trigger/squeeze 可读取。
- `can_left`、`can_right` 与物理手臂固定对应。
- Piper 双臂只读状态连续稳定。
- 左右臂分别通过 1 mm/1 度低速测试。
- clutch、tracking timeout、CAN timeout 和急停有效。
- Pico 单臂连续运行 20 个短 episode。
- 双臂在隔离 workspace 中稳定运行。
- raw episode 可保存、回放和校验。
- 后续能从同一 raw 数据导出 LeRobot 和 RL-100 数据。

## 16. 尚未实现的脚本

以下命令在本 runbook 中给出了最终接口，但对应脚本目前尚未创建：

```text
tools/piper_isaac_teleop/read_piper_state.py
tools/piper_isaac_teleop/test_cartesian_step.py
tools/piper_isaac_teleop/isaac_controller_source.py
tools/piper_isaac_teleop/piper_arm.py
tools/piper_isaac_teleop/vr_mapper.py
tools/piper_isaac_teleop/safety_filter.py
tools/piper_isaac_teleop/teleoperate.py
tools/piper_isaac_teleop/record.py
tools/piper_isaac_teleop/episode_recorder.py
tools/piper_data/validate_raw.py
tools/piper_data/export_lerobot.py
tools/piper_data/export_rl100_zarr.py
```

因此实际执行应分成两段：

```text
现在可执行：Step 1-6，以及 Isaac/Pico controller smoke test
完成 bridge 实现后：Step 7-12
```

下一项代码工作应先实现三个最小脚本：

```text
read_piper_state.py
test_cartesian_step.py
teleoperate.py --dry-run
```

这三个脚本通过后，再实现真实运动和数据记录。
