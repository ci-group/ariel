# minimal_usb_cam.py
import cv2
import threading
import time
from typing import Optional
import numpy as np

class USBMJPGCamera:
    def __init__(self, device='/dev/video0', width=1280, height=720, fps=30):
        self.device = device
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps) if fps is not None else None

        # 强制使用 V4L2 后端
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera: {self.device}")

        # 1. Set FOURCC first
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # 2. Set Resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # 3. Set FPS
        if self.fps is not None:
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 预热：尝试抓几帧
        for _ in range(10):
            ok, _ = self.cap.read()
            if ok:
                break
            time.sleep(0.01)

        self._frame_lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._running = False
        self._t: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._t = threading.Thread(target=self._loop, daemon=False)  # 非 daemon，确保 join
        self._t.start()

    def _loop(self):
        while self._running:
            # Use read() directly for MJPG as it's often more stable with some drivers
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.001)
                continue

            # Check if it's BGR (from MJPG/YUYV auto-decode) or YUYV raw
            if frame.ndim == 3 and frame.shape[2] == 3:
                # OpenCV usually decodes to BGR
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            elif frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 2):
                # YUYV (2 bytes per pixel)
                rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_YUY2)
            else:
                continue

            with self._frame_lock:
                self._latest = rgb
            
            # Very small sleep to prevent CPU hogging
            time.sleep(0.001)

    def get_latest(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def block_until_first_frame(self, timeout=5.0) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.get_latest() is not None:
                return True
            time.sleep(0.01)
        return False

    def stop(self):
        # 安全停止线程、释放资源
        self._running = False
        if self._t is not None:
            self._t.join(timeout=2.0)
            self._t = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
