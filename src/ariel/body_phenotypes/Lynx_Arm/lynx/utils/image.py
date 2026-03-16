# --- image.py ---
import os
import time
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import Wrapper
from PIL import Image

try:
    import imageio.v2 as imageio  # imageio>=2.9
except Exception:
    import imageio

class ImageCaptureWrapper(Wrapper):
    def __init__(self, env, save_dir: str, camera_name: Optional[str] = None,
                 out_size: int = 64, every_n_steps: int = 1, prefix: str = "rgb"):
        super().__init__(env)
        self.save_dir = save_dir
        self.camera_name = camera_name
        self.out_size = out_size
        self.every_n_steps = every_n_steps
        self.prefix = prefix
        os.makedirs(self.save_dir, exist_ok=True)
        self._global_step = 0
        self._episode_idx = 0
        self.env_unwrapped = self.env.unwrapped

    def _grab_frame(self) -> Optional[np.ndarray]:
        # 1) 真机优先：专用接口
        if hasattr(self.env_unwrapped, "get_rgb_frame"):
            # print(f"[ImageCaptureWrapper] Using get_rgb_frame() from env_unwrapped.")
            try:
                f = self.env_unwrapped.get_rgb_frame(self.camera_name)
            except TypeError:
                f = self.env_unwrapped.get_rgb_frame()
            if f is not None:
                return f

        # 2) 兜底：render（如返回 dict，取指定相机或第一个）
        if hasattr(self.env_unwrapped, "render_sim"):
            try:
                out = self.env_unwrapped.render_sim(self.camera_name) if self.camera_name is not None else self.env_unwrapped.render_sim()
            except TypeError:
                out = self.env_unwrapped.render_sim()
            if out is None:
                return None
            if isinstance(out, np.ndarray):
                return out
            if isinstance(out, dict) and out:
                if self.camera_name and self.camera_name in out:
                    return out[self.camera_name]
                return out[next(iter(out.keys()))]
        return None

    def _capture_and_save(self, tag: str) -> Optional[str]:
        frame = self._grab_frame()
        # print(f"[ImageCaptureWrapper] Captured frame at step {self._global_step}, shape: {frame.shape if frame is not None else 'N/A'}")
        if frame is None:
            return None
        img = center_square_resize(frame, self.out_size)
        fn = f"{self.prefix}_{tag}_{self._global_step:08d}.png"
        path = os.path.join(self.save_dir, fn)
        Image.fromarray(img).save(path)
        return path

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._episode_idx += 1
        path = self._capture_and_save(tag=f"ep{self._episode_idx}")
        info = dict(info) if info is not None else {}
        if path is not None:
            info["rgb64x64_path"] = path
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._global_step += 1
        if self._global_step % self.every_n_steps == 0:
            path = self._capture_and_save(tag=f"ep{self._episode_idx}")
            info = dict(info) if info is not None else {}
            if path is not None:
                info["rgb64x64_path"] = path
        return obs, reward, terminated, truncated, info


def center_square_resize(img: np.ndarray, out_size: int = 64) -> np.ndarray:
    if img is None:
        raise ValueError("img is None")
    if img.ndim == 2:
        # 灰度 -> 伪 RGB
        img = np.stack([img]*3, axis=-1)
    elif img.ndim == 3 and img.shape[2] == 4:
        # RGBA -> RGB
        img = img[..., :3]

    h, w = img.shape[:2]
    side = min(h, w)
    top  = (h - side) // 2
    left = (w - side) // 2
    square = img[top:top+side, left:left+side]

    # 转 PIL 统一做高质量缩放（抗锯齿）
    im = Image.fromarray(square)
    im = im.resize((out_size, out_size), Image.BICUBIC)  # 或 Image.LANCZOS
    out = np.asarray(im, dtype=np.uint8)
    return out