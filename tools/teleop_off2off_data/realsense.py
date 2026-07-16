import cv2
import time
import numpy as np
import viser
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


class RealSense(object):
    def __init__(
        self,
        fps=30,
        depth_width=640,
        depth_height=480,
        color_width=640,
        color_height=480,
        num_points=1024
    ):
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.color_width = color_width
        self.color_height = color_height
        self.num_points = num_points

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, fps)
        self.align = rs.align(rs.stream.color)

    def start(self):
        last_error = None
        profile = None
        for attempt in range(5):
            try:
                print(f"[realsense] pipeline.start attempt {attempt + 1}/5", flush=True)
                profile = self.pipeline.start(self.config)
                break
            except RuntimeError as e:
                last_error = e
                print(f"[realsense] pipeline.start failed: {e}", flush=True)
                time.sleep(0.5)
        if profile is None:
            raise RuntimeError(f"failed to start RealSense pipeline after retries: {last_error}")

        # get intrinsics from raw frames first.  On some RealSense setups,
        # calling align.process() during startup can keep waiting even though
        # the raw depth/color streams are already producing frames.
        frames = None
        for i in range(20):
            try:
                print(f"[realsense] waiting for raw frames {i + 1}/20", flush=True)
                frames = self.pipeline.wait_for_frames(5000)
                if frames.get_depth_frame() and frames.get_color_frame():
                    break
            except RuntimeError as e:
                print(f"[realsense] wait_for_frames failed: {e}", flush=True)
                time.sleep(0.1)
        if frames is None or not frames.get_depth_frame() or not frames.get_color_frame():
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
        self.pipeline.stop()

    def get_frame(self, require_pc=False):
        frames = None
        for i in range(20):
            try:
                frames = self.align.process(self.pipeline.wait_for_frames(5000))
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

        point_cloud = point_cloud_downsample(depth2pc(
            depth_image * self.depth_scale,
            self.depth_intrinsics,
            X_root_camera
        ), self.num_points) if require_pc else None

        return {
            'timestamp': timestamp,
            'color': color_image,
            'depth': depth_image,
            'depth_scale': self.depth_scale,
            'point_cloud': point_cloud
        }


if __name__ == '__main__':
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
