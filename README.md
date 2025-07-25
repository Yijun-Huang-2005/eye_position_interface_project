# eye\_position\_interface

**双眼三维位置捕获接口**\
基于 Intel RealSense D455 深度相机 + MediaPipe Face Mesh，实现左右眼中心点在摄像头坐标系下的三维位置捕获，输出单位为厘米。

---

## 一、安装

1. 克隆仓库

   ```bash
   git clone https://github.com/Yijun-Huang-2005/eye_position_interface_project.git
   cd eye_position_interface
   ```

2. 创建并激活虚拟环境（Windows PowerShell 示例）

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install --upgrade pip
   ```

3. 安装依赖

   ```bash
   pip install -r requirements.txt
   ```

4. （可选）安装成命令行工具

   ```bash
   pip install .
   ```

---

## 二、快速开始

### 作为库调用

```python
from eye_position_interface.interface import EyePositionInterface

# 初始化：设置深度和彩色分辨率、帧率
iface = EyePositionInterface(
    depth_width=1280, depth_height=720, depth_fps=30,
    color_width=1280, color_height=720, color_fps=30
)
try:
    while True:
        success, left, right = iface.get_eye_positions()
        if not success:
            break
        # left 和 right 是 (X, Y, Z) 三元组，单位：毫米
        print(f"Left: {left}, Right: {right}")
finally:
    iface.release()
```

### 作为命令行工具

```bash
eye-position
```

输出示例：

```
Left: (652.0, -43.5, -12.3), Right: (540.0, 21.7, -10.8)
Left: (645.3, -42.8, -12.1), Right: (533.8, 22.1, -11.0)
…
```

按 `Ctrl+C` 或 `Esc` 退出。

---

## 三、API 说明

### `EyePositionInterface`

- **构造函数**

  ```python
  EyePositionInterface(
      depth_width: int, depth_height: int, depth_fps: int,
      color_width: int, color_height: int, color_fps: int,
      smoothing_size: int = 5,
      min_depth: float = 0.1,
      max_depth: float = 3.0
  )
  ```

  - `depth_*`，`color_*`：摄像头分辨率与帧率
  - `smoothing_size`：深度／坐标平滑历史帧数
  - `min_depth`, `max_depth`：深度过滤范围（米）

- **方法 **``

  ```python
  success, left_cm, right_cm = iface.get_eye_positions()
  ```

  - `success` (`bool`)：True 表示获取成功
  - `left_cm`, `right_cm` (`tuple[float, float, float]`)：左右眼中心在相机坐标系下的三维坐标（毫米）

- **方法 **``\
  停止摄像头采集、释放资源。

- **入口函数 **``\
  在命令行模式下从 `console_scripts` 调用，循环打印输出。

---

## 四、依赖与引用

- **Intel RealSense SDK**

  - Python 绑定：`pyrealsense2>=2.50.0`
  - 官方文档：[https://dev.intelrealsense.com](https://dev.intelrealsense.com)

- **MediaPipe**

  - Face Mesh：`mediapipe>=0.8.5`
  - 官方文档：[https://google.github.io/mediapipe/solutions/face\_mesh](https://google.github.io/mediapipe/solutions/face_mesh)

- **OpenCV**：`opencv-python>=4.5.0`（图像读写与处理）

- **NumPy**：`numpy>=1.19.0`（数值运算）

所有依赖已在 `requirements.txt` 中列出。

---

