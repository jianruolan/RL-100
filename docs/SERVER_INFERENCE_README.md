# 服务器端 DP3 推理环境

本文说明如何在服务器的 VS Code 容器中加载并验证 control-clean DP3 模型。
服务器只做 GPU 推理，不连接本地 Piper、CAN、RealSense 或 ROS。RealSense、Piper
反馈读取、动作安全检查和最终下发应运行在本地真机电脑。

## 1. 部署边界

```text
本地真机电脑                         服务器 GPU 容器
RealSense + ROS/Piper                RL-100 + PyTorch + checkpoint
读取 joint/gripper 状态       --->   组装观测、模型推理
安全限幅、watchdog、急停       <---   返回 action chunk
JointCtrl/ROS 下发
```

服务器容器不需要映射 `/dev/can0`、Piper USB 或 RealSense USB，也不需要使用
`--privileged`。第一阶段可以只在服务器运行离线 smoke test；远程真机运行时，
应将模型加载/`predict_policy` 抽成常驻推理服务，再由本地控制节点发送观测并接收动作。

## 2. 服务器目录和权重

以下命令假设仓库位于 `/workspace/RL-100`。如果容器内路径不同，所有命令中的
`/workspace/RL-100` 替换为实际路径。

本次使用的模型输出目录是：

```text
data/outputs/piper_pick_and_place_augmented_chunk4_control_clean_dp3_medium_episode10_bs64_epoch4000_seed42
```

配置中的关键约定为：

```text
image       = [3, 84, 84]
point_cloud = [512, 3]
agent_pos   = [7]
action      = [7]
n_obs_steps = 3
n_action_steps = 4
action_key  = policy_action
```

`bc/` 和 `best_val/` 都包含 `model.pt`、`encoder.pt`。默认脚本使用 `bc`，也可以
通过 `--policy-subdir best_val` 切换。

## 3. 容器内依赖

容器中至少需要：

```text
Python 3
PyTorch（与服务器 GPU 驱动匹配）
NumPy、OpenCV、OmegaConf/Hydra
仓库内 third_party/pytorch3d_simplified
RL-100 的 Python 包
```

如果服务器使用 CUDA，先确认 GPU：

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

如果项目环境使用 MUSA，应在容器启动前设置对应的 `MUSA_HOME`、`PATH`、
`LD_LIBRARY_PATH` 和可见设备变量，并把 `--device` 改成该环境支持的设备名。

## 4. 指定权重的服务器端验证

先进入仓库根目录。不要从 `tools/teleop_off2off_data` 目录执行，否则相对的
`data/` 和仓库模块可能无法解析。

```bash
cd /workspace/RL-100

python tools/teleop_off2off_data/infer_piper_control_clean_dp3.py \
  --output-dir data/outputs/piper_pick_and_place_augmented_chunk4_control_clean_dp3_medium_episode10_bs64_epoch4000_seed42 \
  --policy-subdir bc \
  --device cuda:0 \
  --offline-smoke
```

该命令只加载模型、构造一条离线观测并执行推理，不会打开 RealSense，不会连接
Piper，也不会下发动作。看到 smoke test 完成且没有 `FileNotFoundError`、维度错误
或 CUDA 错误，才说明服务器端模型环境基本可用。

使用 `best_val` 的验证命令：

```bash
python tools/teleop_off2off_data/infer_piper_control_clean_dp3.py \
  --output-dir data/outputs/piper_pick_and_place_augmented_chunk4_control_clean_dp3_medium_episode10_bs64_epoch4000_seed42 \
  --policy-subdir best_val \
  --device cuda:0 \
  --offline-smoke
```

## 5. 现有脚本的边界

`tools/teleop_off2off_data/infer_piper_control_clean_dp3.py` 当前是“本地相机 + 本地
Piper + 推理 + 下发”的真机执行器，不是网络推理 server。它的 `--offline-smoke`
模式适合在服务器验证权重；直接在没有硬件的服务器上运行普通模式会尝试打开
RealSense 和 Piper SDK，不能作为远程服务使用。

远程部署时建议保留该脚本的以下部分到服务器：

```text
contact.load_policy_and_dataset()
base.build_obs()
base.predict_policy()
AsyncPolicyWorker
training_stats / action_key 处理
```

并把以下部分留在本地控制端：

```text
RealSense()
PiperProcessProxy / Piper SDK
GetArmJointMsgs()
enable_robot_for_position_control()
safe_action() 的最终硬件限幅
send_action()、watchdog、急停和保持当前位置
```

## 6. 服务器端必须实现的文件

`--offline-smoke` 只能验证模型，不能接收本地真机观测。要进行“服务器推理、
本地下发”，服务器代码至少需要补齐以下文件：

```text
server/
├── __init__.py
├── policy.proto               # 已在仓库中提供，服务器必须直接复用
├── policy_pb2.py              # 已按共享proto生成
├── policy_pb2_grpc.py         # 已按共享proto生成
├── dp3_runtime.py             # 只负责模型加载、预热和推理
├── inference_server.py        # gRPC 服务入口
└── smoke_client.py            # 不连接真机的服务自测客户端
```

### 6.1 `server/policy.proto`

定义本地客户端与服务器之间唯一的数据协议。需要包含：

```text
InferenceRequest
  protocol_version
  episode_id
  sequence_id
  capture_timestamp_ns
  agent_pos        # 7 个 float32
  image            # float32 bytes，形状 [3,84,84]
  point_cloud      # float32 bytes，形状 [512,3]

InferenceResponse
  protocol_version
  episode_id
  sequence_id
  capture_timestamp_ns
  action_chunk     # float32 bytes，形状 [4,7]
  inference_time_ms
  model_version
  ready
  status_message

GetServerInfo
  模型版本和所有固定shape
  n_obs_steps / n_action_steps
  state_min / state_max
  action_min / action_max
  point_cloud_low / point_cloud_high
  gripper_action_mode
  action_key
```

如修改了共享 proto，必须从仓库根目录重新生成：

```bash
cd /workspace/RL-100

python -m grpc_tools.protoc \
  -I . \
  --python_out=. \
  --grpc_python_out=. \
  server/policy.proto
```

### 6.2 `server/dp3_runtime.py`

这个文件只处理模型，不导入 RealSense、ROS 或 Piper SDK。启动时完成：

```text
解析 output_dir 和 policy_subdir
读取 .hydra/config.yaml
检查 image/point_cloud/agent_pos/action 形状
调用 contact.load_policy_and_dataset()
加载 bc/model.pt 和 bc/encoder.pt
把模型移动到 cuda:0
执行 GPU warmup
保存训练统计量和 action_key=policy_action
```

对外提供最小接口：

```python
class DP3Runtime:
    def __init__(self, output_dir, policy_subdir, device): ...
    def predict(self, observations) -> np.ndarray: ...
```

`predict()` 接收连续 3 帧观测，返回 `[4,7]` action chunk。服务端不能在这里调用
`safe_action()` 或 `send_action()`；最终硬件安全检查必须由本地执行。

不要直接 import `infer_piper_control_clean_dp3.py` 作为 runtime。该脚本当前会导入
RealSense/Piper 相关模块，而且包含 `/home/mtarch/Desktop/zyf/RL-100` 的硬编码
路径。服务器 runtime 应使用 `Path(__file__)` 推导仓库根目录，或由
`--repo-root`/`RL100_REPO_ROOT` 显式传入。

### 6.3 `server/inference_server.py`

这是容器内常驻进程。职责：

```text
创建 DP3Runtime 并完成预热
监听 0.0.0.0:50051
实现GetServerInfo并返回训练安全统计
实现ResetEpisode并清空该episode历史
校验 protocol_version、shape、episode_id 和 sequence_id
拒绝重复、乱序和过期请求
按 episode 维护 3 帧 observation history
调用 DP3Runtime.predict()
返回 [4,7] action chunk
记录请求序号、推理耗时、模型版本和错误
客户端断开时清空对应 episode 的历史
```

默认一次只允许一个真机 episode 操作模型；不要把来自不同机械臂的历史混在一起。

启动命令：

```bash
cd /workspace/RL-100

python server/inference_server.py \
  --host 0.0.0.0 \
  --port 50051 \
  --output-dir data/outputs/piper_pick_and_place_augmented_chunk4_control_clean_dp3_medium_episode10_bs64_epoch4000_seed42 \
  --policy-subdir bc \
  --device cuda:0 \
  --max-observation-age-ms 200
```

### 6.4 `server/smoke_client.py`

这个客户端只用于服务器容器内自测，不连接 RealSense 和 Piper。它读取一条离线
样本或构造合法张量，调用 `127.0.0.1:50051`，并检查：

```text
response.sequence_id 与 request 一致
action_chunk 形状为 [4,7]
所有动作都是有限数值
inference_time_ms 有记录
错误 shape 和旧 sequence_id 会被拒绝
```

启动自测：

```bash
python server/smoke_client.py \
  --server 127.0.0.1:50051 \
  --output-dir data/outputs/piper_pick_and_place_augmented_chunk4_control_clean_dp3_medium_episode10_bs64_epoch4000_seed42
```

以上四部分是服务器端交付范围。本地的 RealSense/Piper 客户端应单独实现为：

```text
tools/local/piper_policy_bridge.py
```

它不放在服务器容器中，具体职责见 `docs/LOCAL_ROS_PIPER_README.md`。

## 7. 容器启动和端口

容器需要挂载仓库、checkpoint，并获得 GPU，但不需要 `--privileged`、CAN 或 USB：

```bash
docker run --rm -it \
  --gpus all \
  -v /服务器实际路径/RL-100:/workspace/RL-100 \
  -p 127.0.0.1:50051:50051 \
  --name rl100-dp3-server \
  <镜像名>
```

端口只绑定服务器的 `127.0.0.1`，本地通过 SSH 隧道访问：

```bash
ssh -N \
  -L 50051:127.0.0.1:50051 \
  用户名@服务器地址
```

容器内 `inference_server.py` 监听 `0.0.0.0:50051`；本地 bridge 连接
`127.0.0.1:50051`。

## 8. 远程推理接口约定

服务器推理服务收到一条请求：

```text
episode_id
sequence_id
timestamp
agent_pos[7]
image[3,84,84]
point_cloud[512,3]
```

返回：

```text
episode_id
sequence_id
action_chunk[4,7]
inference_time_ms
model_version
```

服务器必须检查请求版本、数据维度和序号，并记录推理耗时。客户端应设置请求
deadline；超时、乱序或网络断开时，本地不得执行旧动作，而应保持当前目标或进入
安全停止状态。ROS 2 DDS 不建议直接跨公网，优先使用 gRPC/ZeroMQ，并通过 VPN 或
SSH 隧道传输。

## 9. 服务器端启动顺序

```text
1. 容器内确认 GPU 和 checkpoint 文件。
2. 执行 infer_piper_control_clean_dp3.py --offline-smoke。
3. 启动 server/inference_server.py，等待模型预热完成。
4. 在容器内运行 server/smoke_client.py。
5. 本地建立 SSH 隧道。
6. 本地 bridge 先以 shadow 模式请求推理，不下发 Piper。
7. 验证延迟、序号、断网和超时行为后，才进入低速真机阶段。
```

## 10. 服务器端验收清单

- [ ] 容器能看到目标 GPU。
- [ ] 目标输出目录存在 `.hydra/config.yaml`、`bc/model.pt`、`bc/encoder.pt`。
- [ ] `--offline-smoke` 成功。
- [ ] `policy.proto` 已生成 `policy_pb2.py` 和 `policy_pb2_grpc.py`。
- [ ] `dp3_runtime.py` 不导入 RealSense、ROS 或 Piper SDK。
- [ ] `inference_server.py` 能监听 `50051` 并在启动时完成 GPU 预热。
- [ ] `smoke_client.py` 能获得 `[4,7]` action chunk。
- [ ] 错误形状、过期时间戳、重复和乱序序号都会被拒绝。
- [ ] 日志能记录模型目录、`policy_subdir`、设备和推理耗时。
- [ ] 服务端没有访问 CAN、USB 或 ROS 的权限需求。
- [ ] 网络超时不会让服务器继续缓存并发送旧动作。
- [ ] 真机首次执行仍从本地 shadow 模式、低速和人工确认开始。
