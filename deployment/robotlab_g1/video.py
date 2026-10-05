"""Off-screen MuJoCo video recording for deployment demonstrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def ensure_offscreen_framebuffer(model: Any, *, width: int, height: int) -> tuple[int, int]:
    """Grow MuJoCo's runtime off-screen framebuffer for the requested video size."""

    visual_global = model.vis.global_
    visual_global.offwidth = max(int(visual_global.offwidth), int(width))
    visual_global.offheight = max(int(visual_global.offheight), int(height))
    return int(visual_global.offwidth), int(visual_global.offheight)


class MujocoVideoRecorder:
    """Render a tracking-camera MP4 without changing the control loop."""

    def __init__(
        self,
        mujoco_module: Any,
        model: Any,
        data: Any,
        *,
        base_body_id: int,
        output: str | Path,
        fps: float = 30.0,
        source_fps: float = 50.0,
        width: int = 1280,
        height: int = 720,
        distance: float = 4.0,
        azimuth: float = 135.0,
        elevation: float = -18.0,
    ) -> None:
        if not np.isfinite(fps) or fps <= 0.0 or fps > source_fps:
            raise ValueError("video fps must be positive and no greater than policy fps")
        if width <= 0 or height <= 0:
            raise ValueError("video width and height must be positive")
        if width % 2 or height % 2:
            raise ValueError("video width and height must be even for H.264 encoding")
        try:
            import imageio.v2 as imageio
        except ImportError as exc:  # pragma: no cover - deployment dependency
            raise RuntimeError("video recording requires imageio and imageio-ffmpeg") from exc

        self.mj = mujoco_module
        self.model = model
        self.data = data
        self.base_body_id = int(base_body_id)
        self.output = Path(output).expanduser().resolve()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        self.source_fps = float(source_fps)
        self._phase = 0.0
        self.frame_count = 0
        self._closed = False

        # Exported MJCF assets commonly keep MuJoCo's default 640x480
        # off-screen buffer. Renderer rejects larger frames unless the buffer
        # is enlarged first. This changes visualization capacity only.
        ensure_offscreen_framebuffer(model, width=width, height=height)
        self.renderer = self.mj.Renderer(model, height=height, width=width)
        self.camera = self.mj.MjvCamera()
        self.mj.mjv_defaultCamera(self.camera)
        self.camera.type = self.mj.mjtCamera.mjCAMERA_FREE
        self.camera.distance = float(distance)
        self.camera.azimuth = float(azimuth)
        self.camera.elevation = float(elevation)
        self.writer = imageio.get_writer(
            str(self.output),
            fps=self.fps,
            codec="libx264",
            quality=8,
            macro_block_size=None,
        )

    def _append_frame(self) -> None:
        self.camera.lookat[:] = np.asarray(self.data.xpos[self.base_body_id], dtype=float)
        self.renderer.update_scene(self.data, camera=self.camera)
        self.writer.append_data(self.renderer.render())
        self.frame_count += 1

    def capture_initial(self) -> None:
        """Capture the reset pose before the first policy action."""

        self._append_frame()

    def capture_policy_step(self, episode_index: int, policy_step: int) -> None:
        """Sample policy-rate states at the requested output frame rate."""

        del episode_index
        if policy_step < 0:
            self.capture_initial()
            return
        self._phase += self.fps
        if self._phase + 1.0e-9 >= self.source_fps:
            self._phase -= self.source_fps
            self._append_frame()

    def close(self) -> Path:
        if not self._closed:
            try:
                self.writer.close()
            finally:
                self.renderer.close()
                self._closed = True
        return self.output

    def __enter__(self) -> "MujocoVideoRecorder":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
