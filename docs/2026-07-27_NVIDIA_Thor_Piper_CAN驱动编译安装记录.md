# 2026-07-27 NVIDIA Thor 上 Piper USB-CAN 驱动编译安装记录

## 1. 问题与结论

本机为 NVIDIA Thor/aarch64，当前内核：

```text
Linux 6.8.12-tegra aarch64
```

Piper 使用的 USB-CAN 适配器遵循 `gs_usb`/candleLight 协议。当天遇到的核心问题是：USB 设备可以被系统枚举，但当前 `6.8.12-tegra` 的模块目录中没有可加载的 `gs_usb` 驱动，因此不会生成可供 Piper SDK 使用的 SocketCAN 接口。

最终解决方法是：

1. 使用与内核基础版本一致的 Linux `v6.8.12` 源码取得 `drivers/net/can/usb/gs_usb.c`；
2. 使用 Thor 当前内核的构建头文件单独编译外部模块；
3. 把 `gs_usb.ko` 安装到当前内核模块目录；
4. 执行 `depmod` 和 `modprobe`；
5. 根据 USB 物理端口把生成的 CAN 接口固定命名为 `can_piper`，bitrate 设置为 1 Mbps。

当前已验证的模块信息：

```text
filename: /lib/modules/6.8.12-tegra/kernel/drivers/net/can/usb/gs_usb.ko
vermagic: 6.8.12-tegra SMP preempt mod_unload modversions aarch64
depends:  can-dev
```

`lsusb -t` 中 USB-CAN 接口显示：

```text
Driver=gs_usb
```

当天使用的 USB bus-info 为：

```text
1-2.3:1.0
```

最终 SocketCAN 名称为：

```text
can_piper
```

## 2. 先判断是不是驱动缺失

### 2.1 确认内核和架构

```bash
uname -a
uname -r
uname -m
```

本机应看到：

```text
6.8.12-tegra
aarch64
```

### 2.2 确认 USB 设备是否存在

```bash
lsusb
lsusb -t
```

如果 `lsusb` 能看到 USB-CAN 设备，但 `lsusb -t` 对应接口为 `Driver=[none]`，同时下面的命令看不到新 CAN 口，则优先检查 `gs_usb`：

```bash
ip -brief link show type can
```

### 2.3 检查模块

```bash
modinfo gs_usb
sudo modprobe gs_usb
```

典型的缺失表现是 `modinfo` 找不到模块，或 `modprobe` 报当前内核模块目录中不存在 `gs_usb`。

不要把“安装 `can-utils`”与“安装内核驱动”混为一件事：

- `can-utils` 提供 `candump`、`cansend` 等用户态工具；
- `ethtool` 用于查询接口对应的 USB bus-info；
- `gs_usb.ko` 才是把 USB-CAN 设备注册成 SocketCAN 接口的内核驱动。

## 3. 安装用户态依赖

```bash
sudo apt-get update
sudo apt-get install -y \
  git build-essential can-utils ethtool iproute2
```

验证：

```bash
command -v make
command -v candump
command -v ethtool
command -v ip
```

## 4. 检查 Thor 内核构建目录

外部模块必须针对正在运行的同一个内核编译：

```bash
readlink -f /lib/modules/"$(uname -r)"/build
test -d /lib/modules/"$(uname -r)"/build
```

本机的 `/lib/modules/6.8.12-tegra/build` 指向 NVIDIA/Ubuntu 提供的 aarch64 内核头文件树：

```text
/usr/src/linux-headers-6.8.12-tegra-ubuntu24.04_aarch64/
  3rdparty/canonical/linux-noble
```

本机不存在普通路径 `/usr/src/linux-headers-6.8.12-tegra` 并不代表缺少头文件；应以 `/lib/modules/$(uname -r)/build` 是否有效为准。

如果该链接不存在，先安装与设备 BSP/JetPack 和当前内核完全匹配的 NVIDIA 内核头文件。不要用另一个 Ubuntu 内核版本的 headers 勉强编译。

## 5. 编译 `gs_usb.ko`

### 5.1 获取匹配基础版本的源码

```bash
mkdir -p "$HOME/src"
cd "$HOME/src"

git clone --depth 1 --branch v6.8.12 \
  https://git.kernel.org/pub/scm/linux/kernel/git/stable/linux.git \
  linux-6.8.12
```

这里使用 `v6.8.12` 是因为当前 Thor 内核为 `6.8.12-tegra`。如果以后 `uname -r` 变化，必须重新选择匹配的新基础版本并重新编译，不能继续照抄 `v6.8.12`。

### 5.2 建立独立模块目录

```bash
mkdir -p "$HOME/build/gs_usb"
cd "$HOME/build/gs_usb"

cp "$HOME/src/linux-6.8.12/drivers/net/can/usb/gs_usb.c" .
```

创建 `Makefile`：

```makefile
obj-m += gs_usb.o
```

也可用 shell 创建：

```bash
printf '%s\n' 'obj-m += gs_usb.o' > Makefile
```

### 5.3 使用当前 Thor 内核头文件编译

```bash
make -C /lib/modules/"$(uname -r)"/build \
  M="$PWD" modules
```

检查产物：

```bash
ls -lh gs_usb.ko
modinfo ./gs_usb.ko | grep -E 'filename|depends|vermagic'
```

重点检查 `vermagic` 必须包含当前 `uname -r`，本机为：

```text
6.8.12-tegra ... aarch64
```

## 6. 安装并加载模块

安装前先解析精确目标，避免写错内核目录：

```bash
kernel_release="$(uname -r)"
module_target="/lib/modules/${kernel_release}/kernel/drivers/net/can/usb/gs_usb.ko"
printf 'kernel=%s\ntarget=%s\n' "$kernel_release" "$module_target"
```

如果目标文件已经存在，应先判断它是否属于当前系统包，并备份或通过包管理器恢复，不要直接覆盖未知模块。当天的情形是该模块缺失，因此执行：

```bash
sudo install -D -m 644 gs_usb.ko "$module_target"
sudo depmod -a
sudo modprobe gs_usb
```

验证加载状态：

```bash
lsmod | grep '^gs_usb'
modinfo gs_usb | grep -E 'filename|depends|vermagic'
lsusb -t
ip -brief link show type can
```

成功时：

- `lsmod` 中存在 `gs_usb`；
- `modinfo` 指向 `/lib/modules/$(uname -r)/.../gs_usb.ko`；
- `lsusb -t` 对应接口显示 `Driver=gs_usb`；
- `ip` 能看到一个新的 `canX` 接口。

## 7. 激活并固定命名为 `can_piper`

Piper SDK 本地目录：

```text
/home/mtarch/Desktop/zyf/piper_sdk/piper_sdk
```

先列出接口和物理 USB 位置：

```bash
cd /home/mtarch/Desktop/zyf/piper_sdk/piper_sdk
bash find_all_can_port.sh
```

也可以逐个手动查询：

```bash
for iface in $(ip -brief link show type can | awk '{print $1}'); do
  echo "=== $iface ==="
  sudo ethtool -i "$iface" | grep 'bus-info'
done
```

当天确认 Piper USB-CAN 的 bus-info 为 `1-2.3:1.0`，激活命令为：

```bash
cd /home/mtarch/Desktop/zyf/piper_sdk/piper_sdk
bash can_activate.sh can_piper 1000000 "1-2.3:1.0"
```

脚本会完成：

1. 按 bus-info 找到实际 `canX`；
2. 关闭接口；
3. 设置 CAN bitrate 为 `1000000`；
4. 启用接口；
5. 把接口重命名为 `can_piper`。

USB 物理拓扑改变后，`1-2.3:1.0` 可能变化。不要在另一台机器或更换 USB 口后盲目复用，应重新运行 `find_all_can_port.sh` 或 `ethtool -i`。

## 8. 只读验证

### 8.1 检查链路状态和 bitrate

```bash
ip -brief link show can_piper
ip -details link show can_piper
```

预期看到接口为 `UP`，bitrate 为 `1000000`。

### 8.2 观察 CAN 帧

机械臂上电后执行：

```bash
timeout 5 candump can_piper
```

正常情况下应持续看到 CAN 帧。`candump` 是只读检查，不会给机械臂下发运动命令。

如果没有帧，按顺序检查：

1. Piper 是否上电；
2. CAN-H/CAN-L、终端电阻和 USB 接线；
3. 当前名称是否确实对应 `1-2.3:1.0`；
4. bitrate 是否为 1 Mbps；
5. 接口是否处于 `BUS-OFF` 或有大量 error counter；
6. 是否有其他程序占用或重新配置了接口。

### 8.3 Piper SDK 识别测试

在只读 CAN 正常后再运行 SDK 检测：

```bash
python3 /home/mtarch/Desktop/zyf/piper_sdk/piper_sdk/demo/detect_arm.py \
  --can_port can_piper \
  --hz 10 \
  --req_flag 1
```

在正式推理脚本中使用：

```text
--can can_piper
```

## 9. 重启与内核升级注意事项

### 9.1 重启后

`depmod -a` 已建立模块索引，USB modalias 正常时插入设备应自动加载 `gs_usb`。如果没有自动加载：

```bash
sudo modprobe gs_usb
```

然后重新执行：

```bash
cd /home/mtarch/Desktop/zyf/piper_sdk/piper_sdk
bash can_activate.sh can_piper 1000000 "1-2.3:1.0"
```

接口名和 UP/bitrate 配置不会因为 `.ko` 已安装就自动永久保持；当前流程仍依赖激活脚本。

### 9.2 内核升级后必须重编译

当前安装不是 DKMS 包，而是手工编译的外部模块。只安装在：

```text
/lib/modules/6.8.12-tegra/
```

升级 BSP、JetPack 或内核后，如果 `uname -r` 改变，应：

1. 获取新内核匹配的源码/驱动；
2. 使用新的 `/lib/modules/$(uname -r)/build` 重新编译；
3. 安装到新的模块目录；
4. 重新执行 `depmod` 和 `modprobe`。

绝对不要把旧内核的 `gs_usb.ko` 直接复制到新内核目录。`vermagic` 或 `CONFIG_MODVERSIONS` 不匹配时会出现 `Invalid module format`，即使强行加载也不安全。

### 9.3 Secure Boot/模块签名

如果编译成功但加载时报 `Required key not available`，通常是系统启用了模块签名强制验证。此时需要按照该 Thor 系统的 Secure Boot/MOK 流程为模块签名，或使用由平台正式构建系统生成的已签名模块；不要通过不安全参数长期关闭验证。

## 10. 常见故障速查

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| `lsusb` 有设备，但没有 `canX` | `gs_usb` 缺失或未绑定 | 检查 `modinfo`、`modprobe`、`lsusb -t` |
| `modprobe: Module gs_usb not found` | 当前内核模块目录没有驱动 | 按本文针对当前内核编译安装 |
| `Invalid module format` | 内核版本、架构或 modversions 不匹配 | 删除错误思路，针对当前 `uname -r` 重编译 |
| `Required key not available` | 模块签名被拒绝 | 按平台 Secure Boot 流程签名 |
| 有 `canX`，但脚本找不到指定设备 | bus-info 写错或 USB 口变化 | 用 `ethtool -i` 重新确认 |
| `candump` 没数据 | 未上电、接线、bitrate 或 BUS-OFF | 查看 `ip -details link` 并逐项排查 |
| 推理程序提示找不到 `can_piper` | 尚未激活/重命名 | 重新运行 `can_activate.sh` |
| 重启后再次失效 | 模块未自动加载或接口未重新配置 | `modprobe` 后重新运行激活脚本 |

## 11. 当天最终核验清单

```bash
uname -r
modinfo gs_usb | grep -E 'filename|depends|vermagic'
lsusb -t

cd /home/mtarch/Desktop/zyf/piper_sdk/piper_sdk
bash find_all_can_port.sh
bash can_activate.sh can_piper 1000000 "1-2.3:1.0"

ip -brief link show can_piper
ip -details link show can_piper
timeout 5 candump can_piper
```

本机当天最终状态：

```text
kernel:      6.8.12-tegra
architecture: aarch64
driver:      gs_usb
driver path: /lib/modules/6.8.12-tegra/kernel/drivers/net/can/usb/gs_usb.ko
USB bus-info: 1-2.3:1.0
CAN name:    can_piper
bitrate:     1000000
```
