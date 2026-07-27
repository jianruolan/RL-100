import cv2
import gc
import time
import numpy as np
try:
    import viser
except ImportError:
    # 真机采集/推理不依赖 viser；它只用于本文件末尾的可视化 demo。
    viser = None
from scipy.spatial.transform import Rotation as R
import pyrealsense2 as rs
import fpsample


def depth2pc(depth, camera_intrinsics, camera_pose=np.eye(4)):
    height, width = depth.shape
    fx, fy, cx, cy = camera_intrinsics
    z = depth.flatten()

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    u = u.flatten()
    v = v.flatten()

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_camera = np.stack((x, y, z), axis=1)[z > 0]

    points_camera_h = np.concatenate((points_camera, np.ones((points_camera.shape[0], 1))), axis=1)  # Nx4
    points_world = (camera_pose @ points_camera_h.T).T[:, :3]

    return points_world


def point_cloud_downsample(point_cloud, num_points):
    point_cloud = point_cloud[np.all(np.isfinite(point_cloud), axis=1)]
    if len(point_cloud) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)

    # bounding box filter
    x, y, z = point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2]
    bounding_box_mask = (x > 0.2) & (x < 0.8) & (y > -0.23) & (y < 0.25) & (z > 0.3085) & (z < 0.65)
    filtered_point_cloud = point_cloud[bounding_box_mask]
    if len(filtered_point_cloud) == 0:
        filtered_point_cloud = point_cloud
    if len(filtered_point_cloud) == 0:
        return np.zeros((num_points, 3), dtype=np.float32)
    if len(filtered_point_cloud) > 20000:
        sample_idx = np.linspace(0, len(filtered_point_cloud) - 1, 20000, dtype=np.int64)
        filtered_point_cloud = filtered_point_cloud[sample_idx]

    # FPS sampling
    if len(filtered_point_cloud) < num_points:
        filtered_point_cloud = np.concatenate([filtered_point_cloud] * (num_points // len(filtered_point_cloud) + 1), axis=0)
    sample_idx = fpsample.bucket_fps_kdtree_sampling(filtered_point_cloud, num_points)
    point_cloud = filtered_point_cloud[sample_idx]

    return point_cloud


camera_intrinsics = (1348.70988187, 1348.70988187, 967.53239972, 549.01237165)
X_root_camera = np.array([
    [-0.99926635,  0.02147153,  0.03171328,  0.52130602],
    [-0.00456881,  0.7553141 , -0.65534703,  0.44153803],
    [-0.03802479, -0.65501113, -0.75466187,  0.75009051],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])
# rot_mat = X_root_camera[:3, :3]
# rot_euler = R.from_matrix(rot_mat).as_euler('xyz', degrees=True)
# print(rot_euler)
# rot_euler[0] -= 0.2
# rot_euler[1] += 0.5
# rot_mat = R.from_euler('xyz', rot_euler, degrees=True).as_matrix()
# print(rot_mat)
# X_root_camera[:3, :3] = rot_mat


def point_cloud_pose(point_cloud_frame):
    """返回 depth2pc 使用的坐标变换；camera 与训练转换保持一致。"""

    if point_cloud_frame == "camera":
        return np.eye(4, dtype=np.float64)
    if point_cloud_frame == "root":
        return X_root_camera
    raise ValueError(
        "point_cloud_frame must be 'camera' or 'root', "
        f"got {point_cloud_frame!r}"
    )


class RealSense(object):
    def __init__(
        self,
        fps=30,
        depth_width=640,
        depth_height=480,
        color_width=640,
        color_height=480,
        num_points=1024,
        point_cloud_frame="root",
        align_depth_to_color=True,
    ):
        point_cloud_pose(point_cloud_frame)
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.color_width = color_width
        self.color_height = color_height
        self.requested_fps = int(fps)
        self.fps = int(fps)
        self.num_points = num_points
        self.point_cloud_frame = point_cloud_frame
        # 旧 contact 推理默认使用对齐后的深度；ROS bag pick-and-place
        # 转换使用 /depth/image_rect_raw，因此新推理可显式关闭对齐以匹配训练。
        self.align_depth_to_color = bool(align_depth_to_color)

        self.pipeline = None
        self.config = None
        self._started = False
        self.device_serial = None
        # Some D435 units expose only YUYV for color (often on USB2). Start
        # with BGR8 and fall back to YUYV with software conversion if needed.
        self.color_format = rs.format.bgr8
        self.align = rs.align(rs.stream.color)

    @staticmethod
    def _video_profile_fps(device, stream, fmt, width, height):
        """Return FPS values exposed for one exact video stream profile."""

        result = set()
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                try:
                    video = profile.as_video_stream_profile()
                    if (
                        profile.stream_type() == stream
                        and profile.format() == fmt
                        and video.width() == width
                        and video.height() == height
                    ):
                        result.add(int(profile.fps()))
                except RuntimeError:
                    continue
        return result

    def _resolve_common_fps(self, device):
        """Promote an unsupported request to the nearest usable common FPS."""

        depth_fps = self._video_profile_fps(
            device,
            rs.stream.depth,
            rs.format.z16,
            self.depth_width,
            self.depth_height,
        )
        color_fps = self._video_profile_fps(
            device,
            rs.stream.color,
            self.color_format,
            self.color_width,
            self.color_height,
        )
        common = sorted(depth_fps & color_fps)
        if not common:
            raise RuntimeError(
                "RealSense深度/彩色没有共同stream profile: "
                f"depth={self.depth_width}x{self.depth_height}/{depth_fps}, "
                f"color={self.color_width}x{self.color_height}/{color_fps}"
            )
        if self.requested_fps in common:
            selected = self.requested_fps
        else:
            not_slower = [fps for fps in common if fps >= self.requested_fps]
            selected = min(not_slower) if not_slower else max(common)
            print(
                f"[realsense] requested {self.requested_fps}fps is unavailable "
                f"at {self.color_width}x{self.color_height}; using "
                f"{selected}fps (common={common})",
                flush=True,
            )
        self.fps = selected

    def _new_pipeline(self):
        """Create a fresh pipeline/config pair for one start attempt."""

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if self.device_serial is None:
            context = rs.context()
            devices = list(context.query_devices())
            if len(devices) == 1:
                device = devices[0]
                self.device_serial = device.get_info(rs.camera_info.serial_number)
                print(
                    f"[realsense] selected device: "
                    f"{device.get_info(rs.camera_info.name)} "
                    f"serial={self.device_serial}",
                    flush=True,
                )
                self._resolve_common_fps(device)
            del devices, context
        if self.device_serial:
            self.config.enable_device(self.device_serial)
        self.config.enable_stream(
            rs.stream.depth,
            self.depth_width,
            self.depth_height,
            rs.format.z16,
            self.fps,
        )
        self.config.enable_stream(
            rs.stream.color,
            self.color_width,
            self.color_height,
            self.color_format,
            self.fps,
        )

    def _hardware_reset(self):
        """Reset only the selected RealSense and wait for USB re-enumeration."""

        context = rs.context()
        devices = list(context.query_devices())
        selected = None
        for device in devices:
            try:
                serial = device.get_info(rs.camera_info.serial_number)
            except RuntimeError:
                continue
            if self.device_serial is None or serial == self.device_serial:
                selected = device
                break
        if selected is None:
            raise RuntimeError(
                f"cannot find RealSense serial={self.device_serial} for reset"
            )
        selected.hardware_reset()
        print("[realsense] hardware reset sent; waiting for USB re-enumeration", flush=True)
        del selected, devices, context
        gc.collect()
        time.sleep(3.0)

    def start(self, _allow_hardware_reset=True):
        last_error = None
        profile = None
        for attempt in range(5):
            # A failed librealsense profile resolution can retain a UVC handle.
            # Reusing that pipeline makes all following attempts fail with the
            # misleading "UVC device is already opened" message.
            self.pipeline = None
            self.config = None
            gc.collect()
            self._new_pipeline()
            try:
                print(f"[realsense] pipeline.start attempt {attempt + 1}/5", flush=True)
                profile = self.pipeline.start(self.config)
                self._started = True
                break
            except RuntimeError as e:
                last_error = e
                print(f"[realsense] pipeline.start failed: {e}", flush=True)
                error_text = str(e).upper()
                if (
                    self.color_format == rs.format.bgr8
                    and "BGR8" in error_text
                    and "YUYV" in error_text
                ):
                    self.color_format = rs.format.yuyv
                    print(
                        "[realsense] BGR8 profile unavailable; retrying with YUYV "
                        "and software BGR conversion",
                        flush=True,
                    )
                try:
                    self.pipeline.stop()
                except RuntimeError:
                    pass
                self.pipeline = None
                self.config = None
                gc.collect()
                time.sleep(0.5)
        if profile is None:
            if _allow_hardware_reset:
                print(
                    "[realsense] profile resolution failed repeatedly; "
                    "attempting one hardware reset",
                    flush=True,
                )
                self._hardware_reset()
                return self.start(_allow_hardware_reset=False)
            raise RuntimeError(f"failed to start RealSense pipeline after retries: {last_error}")

        # get intrinsics from raw frames first.  On some RealSense setups,
        # calling align.process() during startup can keep waiting even though
        # the raw depth/color streams are already producing frames.
        frames = None
        for i in range(3):
            try:
                print(f"[realsense] waiting for raw frames {i + 1}/3", flush=True)
                frames = self.pipeline.wait_for_frames(3000)
                if frames.get_depth_frame() and frames.get_color_frame():
                    break
            except RuntimeError as e:
                print(f"[realsense] wait_for_frames failed: {e}", flush=True)
                time.sleep(0.1)
        if frames is None or not frames.get_depth_frame() or not frames.get_color_frame():
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self._started = False
            self.pipeline = None
            self.config = None
            if _allow_hardware_reset:
                print(
                    "[realsense] pipeline started but delivered no frames; "
                    "attempting one hardware reset",
                    flush=True,
                )
                self._hardware_reset()
                return self.start(_allow_hardware_reset=False)
            raise RuntimeError("failed to receive RealSense depth/color frames during startup")
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("failed to receive initial RealSense depth/color frames")
        depth_intrinsics = depth_frame.get_profile().as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = (depth_intrinsics.fx, depth_intrinsics.fy, depth_intrinsics.ppx, depth_intrinsics.ppy)
        color_intrinsics = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        self.color_intrinsics = (color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.ppx, color_intrinsics.ppy)

    def stop(self):
        if self.pipeline is not None and self._started:
            self.pipeline.stop()
        self._started = False

    def get_frame(self, require_pc=False):
        frames = None
        for i in range(20):
            try:
                raw_frames = self.pipeline.wait_for_frames(5000)
                frames = (
                    self.align.process(raw_frames)
                    if self.align_depth_to_color
                    else raw_frames
                )
            except RuntimeError as e:
                print(f"[realsense] capture wait_for_frames failed {i + 1}/20: {e}", flush=True)
                time.sleep(0.1)
                continue

            timestamp = frames.get_timestamp() / 1000  # ms -> s
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if depth_frame and color_frame:
                break
        else:
            raise RuntimeError("failed to capture aligned RealSense depth/color frames")

        depth_image = np.array(depth_frame.get_data())
        color_image = np.array(color_frame.get_data())
        if self.color_format == rs.format.yuyv:
            # pyrealsense2 normally exposes YUYV as HxWx2; support the packed
            # Hx(2W) representation from older bindings as well.
            if color_image.ndim == 2 and color_image.shape[1] == 2 * self.color_width:
                color_image = color_image.reshape(self.color_height, self.color_width, 2)
            if color_image.ndim != 3 or color_image.shape[2] != 2:
                raise RuntimeError(f"YUYV 彩色帧形状异常: {color_image.shape}")
            color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGR_YUY2)

        if require_pc:
            # Piper raw->zarr conversion calls depth2pc(depth, intrinsics)
            # without an extrinsic, so deployment must select "camera" to
            # reproduce the exact same coordinate convention.  "root" keeps
            # the legacy fixed-X_root_camera behavior for existing utilities.
            camera_pose = point_cloud_pose(self.point_cloud_frame)
            point_cloud = point_cloud_downsample(
                depth2pc(
                    depth_image * self.depth_scale,
                    self.depth_intrinsics,
                    camera_pose,
                ),
                self.num_points,
            )
        else:
            point_cloud = None

        return {
            'timestamp': timestamp,
            'color': color_image,
            'depth': depth_image,
            'depth_scale': self.depth_scale,
            'point_cloud': point_cloud,
            'point_cloud_frame': self.point_cloud_frame,
            'depth_intrinsics': self.depth_intrinsics,
            'color_intrinsics': self.color_intrinsics,
            'depth_aligned_to_color': self.align_depth_to_color,
            'color_format': 'YUYV' if self.color_format == rs.format.yuyv else 'BGR8',
        }


if __name__ == '__main__':
    if viser is None:
        raise RuntimeError("运行 RealSense 可视化 demo 需要先安装 viser")
    camera = RealSense()
    camera.start()
    
    server = viser.ViserServer(host='127.0.0.1', port=8080)

    while True:
        frame = camera.get_frame(require_pc=True)
        # print('timestamp:', frame['timestamp'])

        server.scene.add_frame(
            f'camera_pose',
            wxyz=R.from_matrix(X_root_camera[:3, :3]).as_quat()[[3, 0, 1, 2]],
            position=X_root_camera[:3, 3],
            axes_length=0.2,
            axes_radius=0.006
        )

        server.scene.add_point_cloud(
            'pc',
            frame['point_cloud'],
            point_size=0.005,
            point_shape="circle",
            colors=(0, 0, 255)
        )
        time.sleep(0.033)
