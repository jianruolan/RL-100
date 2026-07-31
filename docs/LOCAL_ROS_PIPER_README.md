# 本地 ROS 2 / Piper 真机端环境

本机端连接真实设备，负责采集观测、接收服务器动作、执行最终安全检查并下发
Piper。模型和 checkpoint 在服务器 GPU 容器中运行；本机不需要加载 DP3 模型。

当前目标架构：

```text
RealSense ROS driver ─┐
                      ├─> 观测/推理 ROS 节点 ──gRPC──> 服务器 DP3
Piper ROS driver ─────┘                                      │
                                                             ↓
                                               安全规划 ROS 节点
                                                             ↓
                                                    Piper ROS driver
```

Piper ROS driver 是唯一的 CAN 所有者。观测节点和规划节点不连接 `piper_sdk`；
服务器只运行 gRPC 推理服务，不运行 ROS。

## 1. 本机职责

```text
RealSense RGB-D + Piper feedback
              ↓
      本地 observation bridge
              ↓  observation
       服务器 DP3 推理服务
              ↓  action chunk
      本地安全控制/ROS 节点
              ↓
       Piper CAN / JointCtrl
```

本机必须始终保留以下控制权：

- 关节和夹爪反馈读取；
- 训练范围、物理范围和单步变化检查；
- 速度/加速度限制；
- 相机和状态 watchdog；
- 网络超时、乱序动作和旧动作丢弃；
- Piper 使能、保持当前位置和急停。

服务器断开时，本机不能继续执行缓存动作。

## 2. 已有本地工作区

本机已有 ROS/Piper 工作区：

```text
/home/mtuser/文档/zyf/piper_teleop_v1.1
```

Piper SDK 位于（由 Piper ROS driver 内部调用）：

```text
/home/mtuser/文档/zyf/01-Piper/piper_sdk
```

首次打开终端：

```bash
conda activate rl100_local
cd /home/mtuser/文档/zyf/piper_teleop_v1.1
source tools/project_env.sh
command -v python3
python3 --version
```

`project_env.sh` 会设置 ROS 2 环境、工作区 overlay、`ROS_LOG_DIR` 和本地日志路径。
不要直接使用复制工作区中残留的旧 `build/`、`install/`、`log/` 路径。

本机已创建 `rl100_local` Conda 环境，使用 Python 3.10，并通过环境内 `.pth` 复用
全局 Python 3.10 已安装的 ROS、NumPy、OpenCV、Piper SDK 和 Pinocchio 包。
`command -v python3` 应输出
`/home/mtuser/miniconda3/envs/rl100_local/bin/python3`。不要使用自动激活的 Conda
base（当前为 Python 3.14）运行 ROS 节点，否则 `rclpy` 和已编译 ROS 扩展会出现
ABI/导入错误。服务器 DP3 推理环境与本机 ROS 环境必须分开。

`project_env.sh` 会设置 ROS 的 `PYTHONPATH`。在 source 之后如需执行 Conda 管理命令，
应新开终端或先执行 `unset PYTHONPATH`；正常运行 ROS/Piper 节点时不要清除它。

当前本机状态：Piper SDK 和 Piper ROS 源码已存在，RealSense ROS 包已安装；Piper
控制包需要构建后再使用。

构建 Piper ROS 消息和控制包（不启动 CAN）：

```bash
source /home/mtuser/miniconda3/etc/profile.d/conda.sh
conda activate rl100_local
source /home/mtuser/文档/zyf/piper_teleop_v1.1/tools/project_env.sh
cd /home/mtuser/文档/zyf/piper_teleop_v1.1/third_party/piper_ros
colcon --log-base /tmp/piper_ros_log build \
  --symlink-install --packages-select piper_msgs piper \
  --build-base /tmp/piper_ros_build \
  --install-base /home/mtuser/文档/zyf/piper_teleop_v1.1/third_party/piper_ros/install \
  --cmake-args -DBUILD_TESTING=OFF
source install/setup.bash
ros2 pkg prefix piper
ros2 pkg prefix piper_msgs
ros2 pkg prefix realsense2_camera
```

若 `piper_msgs` 提示缺少接口生成器，只安装最小依赖：

```bash
sudo apt-get install ros-humble-rosidl-default-generators
```

## 3. 软件环境检查

先做不连接真机的检查：

```bash
conda activate rl100_local
cd /home/mtuser/文档/zyf/piper_teleop_v1.1
source tools/project_env.sh
./tools/check_environment.sh
```

检查 ROS 基础命令：

```bash
source /opt/ros/humble/setup.bash
ros2 doctor --report
ros2 pkg list | rg 'piper|realsense|teleop|data_collection'
```

如果修改了 ROS 包，重新构建：

```bash
conda activate rl100_local
cd /home/mtuser/文档/zyf/piper_teleop_v1.1
source tools/project_env.sh
colcon build --symlink-install --cmake-args -DBUILD_TESTING=OFF
source tools/project_env.sh
```

软件测试（不启动 CAN 和相机）：

```bash
conda activate rl100_local
cd /home/mtuser/文档/zyf/piper_teleop_v1.1
source tools/project_env.sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
  ws_piper/src/piper_teleop/test \
  ws_piper/src/piper_data_collection/test
```

## 4. 硬件检查顺序

先确认 CAN 接口状态。仅看到 `can0` 不足以证明可以控制 Piper：

```bash
ip -details -statistics link show can0
```

需要关注接口是否为 `UP`、是否 `STOPPED`/`BUS-OFF`，以及 RX/TX error、重启次数。
如果没有 `can0`、状态异常或没有 Piper 接入，不要启动真机 launch。

RealSense 只先做预览：

```bash
cd /home/mtuser/文档/zyf/piper_teleop_v1.1
source tools/project_env.sh
ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  align_depth.enable:=false
```

确认图像和深度 topic 正常后再退出。DP3 训练输入是 RGB `[3,84,84]`、点云 `[512,3]`，
深度/点云坐标系和训练时必须一致；当前 RL-100 入口使用未对齐深度和
`d435i_depth_optical_frame`。

## 5. 本地 bridge

本地客户端已经实现：

```text
tools/local/piper_policy_bridge.py
tools/local/piper_remote_runtime.py
tools/local/ros_observation_inference_node.py
tools/local/ros_action_planner_node.py
tools/local/ros_piper_bridge_node.py
tools/local/remote_server_smoke.py
tools/local/requirements.txt
server/policy.proto
server/policy_pb2.py
server/policy_pb2_grpc.py
```

它完成：

1. 读取 RealSense 最新完整 RGB-D 帧。
2. 读取 Piper 六关节位置和夹爪宽度。
3. 按训练预处理生成 `image[3,84,84]`、`point_cloud[512,3]`、`agent_pos[7]`。
4. 添加 `episode_id`、`sequence_id` 和时间戳后发送给服务器。
5. 接收 `action_chunk[4,7]`，只接受对应序号且未过期的响应。
6. 对动作执行本地 `safe_action`、速度/加速度限制和夹爪限制。
7. 由独立 planner 节点执行最终动作限幅和 100Hz 轨迹规划。
8. Piper ROS driver 负责唯一的 CAN 下发；相机、反馈或网络超时时进入保持。

`ros_observation_inference_node.py` 和 `ros_action_planner_node.py` 是新的目标架构。
原来的 `piper_policy_bridge.py` 保留作直连 SDK 的兼容/调试原型；正式真机运行时不要
与 Piper ROS driver 同时启动它。

重要：正式架构只允许 Piper ROS driver 连接 `can0`。不要同时启动旧直连 bridge、
`piper_ctrl_single_node` 的第二个实例或其他 Piper SDK 进程。

旧版直连 bridge 提供以下 telemetry 接口（仅调试用）：

```text
发布 /remote_dp3/joint_states_feedback  sensor_msgs/msg/JointState
发布 /remote_dp3/joint_target           sensor_msgs/msg/JointState
发布 /remote_dp3/status                 std_msgs/msg/String（JSON）
发布 /remote_dp3/executing              std_msgs/msg/Bool
订阅 /remote_dp3/stop                   std_msgs/msg/Bool（True：保持并退出）
订阅 /remote_dp3/estop                  std_msgs/msg/Bool（True：Piper 急停并退出）
```

`JointState.position` 前六维为弧度，第七维 `gripper_width` 为米。

新架构 topic：

```text
订阅 /joint_states_feedback             Piper driver feedback
订阅 /camera/color/image_raw            RealSense RGB
订阅 /camera/depth/image_rect_raw       RealSense 未对齐 depth
订阅 /camera/depth/camera_info          depth 内参
发布 /remote_dp3/action_chunk           std_msgs/msg/String（JSON）
订阅 /remote_dp3/action_chunk           planner 输入
发布 /remote_dp3/joint_cmd              Piper driver command remap
```

`action_chunk` JSON 包含 `episode_id`、`sequence_id`、`action_chunk[4,7]`、模型版本、
本地 monotonic 提交时间和完整训练安全统计；这些统计由 planner 用于执行
与 `infer_piper_control_clean_dp3.py` 一致的 `safe_action`。后续稳定后可将
它替换为正式的 ROS interface package。

协议接口：

```text
请求：episode_id, sequence_id, timestamp, agent_pos[7], image[3,84,84], point_cloud[512,3]
响应：episode_id, sequence_id, action_chunk[4,7], inference_time_ms, model_version
```

第一阶段固定使用 `chunk_exec_steps=1` 的 receding-horizon 逻辑，降低网络延迟和
动作堆积风险。

安装和离线自测：

```bash
source /home/mtuser/miniconda3/etc/profile.d/conda.sh
conda activate rl100_local
cd /home/mtuser/文档/zyf/RL-100

python -m pip install -r tools/local/requirements.txt
python tools/local/piper_policy_bridge.py --offline-smoke
```

服务器启动后，先只检查协议和模型元数据，不访问本地硬件：

```bash
python tools/local/piper_policy_bridge.py \
  --server 127.0.0.1:50051 \
  --check-server
```

服务器必须通过 `GetServerInfo` 返回训练安全统计；本地会检查 `policy_action`、
`n_obs_steps=3`、`n_action_steps=4` 和所有输入输出形状。缺少或不匹配时拒绝启动。

如果还没有连接真机，使用下面的纯网络测试发送一次合成观测。它不会导入 ROS、
RealSense 或 Piper SDK，也不会访问 `can0`：

```bash
python tools/local/remote_server_smoke.py \
  --server 127.0.0.1:50051
```

看到 `[network-smoke] PASS`，才说明本机确实完成了 `GetServerInfo -> ResetEpisode ->
Infer -> action_chunk` 的完整通信。合成观测只验证协议和服务链路，不代表真实相机
分布或真机动作已经验证。

## 6. Shadow 到真机的启动顺序

正式架构的启动顺序是：

```text
1. RealSense ROS driver
2. Piper ROS driver（auto_enable=false，唯一 CAN 所有者）
3. ros_observation_inference_node.py
4. ros_action_planner_node.py
```

在没有真机时，只运行纯网络 smoke，不启动下面三个硬件相关节点：

```bash
source /home/mtuser/miniconda3/etc/profile.d/conda.sh
conda activate rl100_local
cd /home/mtuser/文档/zyf/RL-100
python tools/local/remote_server_smoke.py --server 127.0.0.1:50051
```

确认 CAN 和工作区安全后，Piper driver 命令如下（会连接 CAN）：

```bash
source /home/mtuser/文档/zyf/piper_teleop_v1.1/tools/project_env.sh
source /home/mtuser/文档/zyf/piper_teleop_v1.1/third_party/piper_ros/install/setup.bash
ros2 run piper piper_single_ctrl --ros-args \
  -p can_port:=can0 -p auto_enable:=false \
  -r joint_ctrl_single:=/remote_dp3/joint_cmd
```

观测/推理节点和规划节点分别运行：

```bash
cd /home/mtuser/文档/zyf/RL-100
python tools/local/ros_observation_inference_node.py \
  --ros-args \
  -p server:=127.0.0.1:50051 \
  -p rpc_timeout:=1.0 \
  -p fps:=5.0
```

```bash
cd /home/mtuser/文档/zyf/RL-100
python tools/local/ros_action_planner_node.py
```

规划节点默认发布 `/remote_dp3/joint_cmd`，与上面的 Piper driver remap 对接。
本地 planner 的处理顺序固定为：

```text
action_chunk[0]
  -> safe_action（物理限位 + 训练分布 margin + 夹爪命令解码）
  -> JointTrajectoryPlanner（速度/加速度/跟踪误差限制）
  -> 100 Hz JointState 指令
```

下列 ROS 参数名和默认值与
`tools/teleop_off2off_data/infer_piper_control_clean_dp3.py` 保持一致：

```text
rate=100.0
max_joint_speed_rad_s=0.20
max_joint_accel_rad_s2=0.50
max_gripper_speed_m_s=0.02
planner_tracking_error_rad=0.10
dataset_margin_rad=0.03
gripper_margin_m=0.005
gripper_command_threshold=0.5
clip_actions=false
skip_current_joint_range_check=false
skip_gripper_safety_check=false
```

`clip_actions=false` 时，超出训练分布的目标会被拒绝，这与原脚本不传
`--clip-actions` 时一致。如果明确需要原脚本的 `--clip-actions` 行为，启动
planner 时显式添加：

```bash
python tools/local/ros_action_planner_node.py --ros-args \
  -p clip_actions:=true
```

除上述原脚本规划逻辑外，ROS 节点保留 `max_action_age_ms`、序号单调性和
反馈超时检查；这些只是远程通信架构的额外安全层，不会在服务器侧重复规划。
CPU 推理服务建议先使用 `rpc_timeout=1.0s`；RPC 超时只控制网络等待，planner 仍按
`max_action_age_ms` 独立拒绝旧动作。当前 CPU 实测推理约 142ms，建议先以 5Hz 请求，
避免用 15Hz 持续压满 CPU；切换 GPU 后再提高观测请求频率。

### 旧版直连 SDK Shadow（兼容调试，不用于正式架构）

```bash
source /home/mtuser/miniconda3/etc/profile.d/conda.sh
conda activate rl100_local
cd /home/mtuser/文档/zyf/piper_teleop_v1.1
source tools/project_env.sh

cd /home/mtuser/文档/zyf/RL-100
python tools/local/piper_policy_bridge.py \
  --server 127.0.0.1:50051 \
  --can can0 \
  --max-steps 1000 \
  --clip-actions
```

默认就是 shadow：读取真机反馈和相机、请求服务器并打印/记录动作，但绝不使能或
下发 Piper。正式架构应使用前面列出的 Piper ROS driver、观测节点和 planner 节点。

在另一个已经 source ROS 环境的终端可观察节点：

```bash
ros2 node list | rg remote_dp3
ros2 topic echo /remote_dp3/status
ros2 topic hz /remote_dp3/joint_states_feedback
```

要求安全停止并保持：

```bash
ros2 topic pub --once /remote_dp3/stop std_msgs/msg/Bool '{data: true}'
```

只有发生真实危险时才发送急停：

```bash
ros2 topic pub --once /remote_dp3/estop std_msgs/msg/Bool '{data: true}'
```

### 真机阶段

只有上述 shadow 连续稳定后，才允许：

```text
1. 确认机械臂周围无障碍物，急停可用
2. 使用极低 speed-percent
3. max_steps 设置为很小的值
4. chunk 每次只执行 1 步
5. 人工确认后才 EnablePiper
6. Ctrl+C、网络断开、反馈超时都必须进入保持/停止逻辑
```

已有真机脚本的 `--execute`、确认短语、关节限幅、夹爪检查和停止保持逻辑应继续
复用，不要在网络 bridge 中绕过这些安全层。

真机命令必须在 shadow 稳定后运行；建议第一次只执行很短时间：

```bash
python tools/local/piper_policy_bridge.py \
  --server 127.0.0.1:50051 \
  --can can0 \
  --max-steps 200 \
  --speed-percent 5 \
  --max-joint-speed-rad-s 0.10 \
  --max-joint-accel-rad-s2 0.20 \
  --clip-actions \
  --execute
```

程序还会要求输入：

```text
EXECUTE REMOTE DP3 PIPER
```

## 7. 本机验收清单

- [ ] `/opt/ros/humble` 可 source，`ros2` 可用。
- [ ] `tools/project_env.sh` 和 `tools/check_environment.sh` 通过。
- [ ] `colcon build` 完成，测试通过。
- [ ] `ip -details -statistics link show can0` 显示正常 CAN 状态。
- [ ] RealSense RGB、Depth 和时间戳稳定。
- [ ] 本地 bridge 能生成 3 帧观测历史。
- [ ] `ros2 node list` 能看到 `/remote_dp3_piper_bridge`，反馈 topic 持续更新。
- [ ] 没有其他进程同时连接 `can0`。
- [ ] server/client 的 `sequence_id` 和时间戳检查通过。
- [ ] 网络超时不会执行旧动作。
- [ ] shadow 阶段通过后，才进行低速、短时真机测试。
