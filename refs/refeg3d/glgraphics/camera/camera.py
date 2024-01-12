import numpy as np
from functools import lru_cache
from glgraphics.utils.mathutil import projection, rotate_x
from .base import CameraBase

class Camera(CameraBase):
    def __init__(self) -> None:
        self.speed = 1.0
        self.scroll_factor = 0.

        self._view = np.identity(4)
        self._view[:3, 2] = np.array([0, 0, -1])
        self._view[:3, 3] = -np.array([0, 1, 2])
        self._view = self._view.astype("f4")

    @property
    @lru_cache(maxsize=-1)
    def proj(self):
        return projection(fov=19, near=0.001)

    @property
    def view(self):
        center = self.center
        center += self.scroll_factor * self._view[2, :3]

        t = self.trans_from_center(center)
        return np.array([
            [ *self._view[0, :3], t[0]],
            [ *self._view[1, :3], t[1]],
            [ *self._view[2, :3], t[2]],
            [ *self._view[3]]
        ])
    
    @property
    def center(self):
        #according to https://openmvg.readthedocs.io/en/latest/openMVG/cameras/cameras/
        view = self._view
        R = view[:3, :3]
        t = view[:3, 3]
        C = -R.transpose() @ t # camera_center
        return C
    
    def trans_from_center(self, center):
        R = np.identity(4)
        R[:3, :3] = self._view[:3, :3]
        t = np.identity(4)
        t[:3, 3] = center
        return -(R @ t)[:3, 3]

    def key_event(self, key, action, modifiers):
        pass

    def mouse_drag_event(self, x, y, dx, dy):
        pass

    def mouse_scroll_event(self, x_offset, y_offset):
        delta = np.exp(np.abs(self.scroll_factor) * 0.1) - 0.99
        delta = np.clip(delta, 0.001, 10)
        self.scroll_factor += np.sign(y_offset) * delta

    def debug_gui(self, t, frame_t):
        import imgui
        imgui.text(f"camera center: {self.center}")
        imgui.text(f"scroll factor: {self.scroll_factor:.1f}")
