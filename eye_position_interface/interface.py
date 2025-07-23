import pyrealsense2 as rs
import cv2
import numpy as np
from collections import deque
import mediapipe as mp

class EyePositionInterface:
    """
    接口：捕获左右眼中心在相机坐标系下的三维位置 (X, Y, Z)，单位：厘米。

    使用库：
      - pyrealsense2: Intel RealSense 深度 SDK
      - mediapipe: 面部网格关键点检测
      - numpy: 数组运算
      - opencv-python: 图像帧获取和转换
    """

    def __init__(
        self,
        depth_width=1280, depth_height=720, depth_fps=30,
        color_width=1280, color_height=720, color_fps=30,
        smoothing_size=5,
        min_depth=0.1, max_depth=3.0
    ):
        # 初始化 MediaPipe Face Mesh
        self.mp_face = mp.solutions.face_mesh
        self.face_mesh = self.mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # 启动 RealSense 管道
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, depth_fps)
        cfg.enable_stream(rs.stream.color, color_width, color_height, rs.format.bgr8, color_fps)
        self.profile = self.pipeline.start(cfg)

        # 对齐深度到彩色流
        self.align = rs.align(rs.stream.color)

        # 获取深度相机内参
        intr = self.profile.get_stream(rs.stream.depth) \
                     .as_video_stream_profile() \
                     .get_intrinsics()
        self.intrinsics = intr

        # 平滑滤波历史缓存
        self.left_history = deque(maxlen=smoothing_size)
        self.right_history = deque(maxlen=smoothing_size)

        # 初始化上次有效值
        self.last_left = (0.0, 0.0, 0.0)
        self.last_right = (0.0, 0.0, 0.0)

        # 深度过滤范围（米）
        self.min_depth = min_depth
        self.max_depth = max_depth

        # MediaPipe 眼部关键点索引
        self.LEFT_IDS  = [33, 133, 159, 145]
        self.RIGHT_IDS = [362, 263, 386, 374]

    def _get_region_depth(self, depth_frame, x, y, size=5):
        """
        在 (x,y) 周围 size x size 区域采样非零深度并取平均，返回深度 (米)。
        无有效深度则返回 0.0。
        """
        vals = []
        half = size // 2
        for dx in range(-half, half+1):
            for dy in range(-half, half+1):
                dist = depth_frame.get_distance(x+dx, y+dy)
                if dist > 0:
                    vals.append(dist)
        return float(np.mean(vals)) if vals else 0.0

    def get_eye_positions(self):
        """
        捕获一帧，检测面部并计算左右眼中心三维坐标。
        始终返回上次有效值，保证输出连续。

        返回：
            success (bool)
            left_cm (tuple): 左眼 (X,Y,Z) cm
            right_cm (tuple): 右眼 (X,Y,Z) cm
        """
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        depth_frame = aligned.get_depth_frame()
        color_frame = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            return True, self.last_left, self.last_right

        # 转为 RGB 图供 MediaPipe
        color_img = np.asanyarray(color_frame.get_data())
        h, w = color_img.shape[:2]
        rgb = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return True, self.last_left, self.last_right
        lm = results.multi_face_landmarks[0].landmark

        # 计算像素中心
        def center(ids):
            pts = np.array([[lm[i].x*w, lm[i].y*h] for i in ids])
            return int(pts[:,0].mean()), int(pts[:,1].mean())

        lx, ly = center(self.LEFT_IDS)
        rx, ry = center(self.RIGHT_IDS)

        dz_l = self._get_region_depth(depth_frame, lx, ly)
        dz_r = self._get_region_depth(depth_frame, rx, ry)

        # 深度无效则返回历史
        if not (self.min_depth < dz_l < self.max_depth and
                self.min_depth < dz_r < self.max_depth):
            return True, self.last_left, self.last_right

        pt_l = rs.rs2_deproject_pixel_to_point(self.intrinsics, [lx, ly], dz_l)
        pt_r = rs.rs2_deproject_pixel_to_point(self.intrinsics, [rx, ry], dz_r)

        # 米转厘米
        left = (pt_l[0]*100, pt_l[1]*100, pt_l[2]*100)
        right = (pt_r[0]*100, pt_r[1]*100, pt_r[2]*100)

        # 平滑
        self.left_history.append(left)
        self.right_history.append(right)
        left_s = tuple(np.mean(self.left_history, axis=0).tolist())
        right_s = tuple(np.mean(self.right_history, axis=0).tolist())

        # 更新历史值
        self.last_left, self.last_right = left_s, right_s
        return True, left_s, right_s

    def release(self):
        """
        停止相机、释放资源。
        """
        self.pipeline.stop()
        self.face_mesh.close()


def main():
    iface = EyePositionInterface()
    try:
        while True:
            ok, left, right = iface.get_eye_positions()
            if not ok:
                break
            print(f"Left: {left}, Right: {right}")
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        iface.release()

if __name__ == "__main__":
    main()
