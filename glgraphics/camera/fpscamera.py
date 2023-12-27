import numpy as np
from glgraphics.utils.mathutil import spherical, lookAt, cartesian, projection, rotate_x
from .base import CameraBase


class FPSCamera(CameraBase):
    def __init__(self) -> None:
        super().__init__()
        self.fov = 90.
        self.eye = np.array([0., 1., 1.])
        self.phi = np.deg2rad(220)
        self.theta = np.deg2rad(90)
        self.speed = 1.
        self.frame_t = 1.
        self.dragable = True
        
    @property
    def oriental(self):
        return np.array(spherical(self.theta, self.phi, 1.))
    
    @property
    def proj(self):    
        return projection(fov=self.fov)


    @property
    def view(self):
        return lookAt(eye=self.eye, at=self.eye+self.oriental, up=np.array([0, 1, 0]))
    
    @view.setter
    def view(self, newview:np.ndarray):
        x, y, z = -newview[:3,:3].transpose()[2, :3] # z_axis
        self.theta, self.phi = cartesian(x, y, z)
        self.eye = -newview[:3, 3]


    def key_event(self, key, action, modifiers):
        if key == 119: # W
            self.eye += self.frame_t * self.speed * self.oriental
        elif key==97: # A
            self.eye -= self.frame_t * self.speed * np.cross(self.oriental, np.array([0.,1.,0.]))
        elif key==115: # S
            self.eye -= self.frame_t * self.speed * self.oriental
        elif key==100: # D
            self.eye += self.frame_t * self.speed * np.cross(self.oriental, np.array([0.,1.,0.]))
        elif key==106: # J
            self.theta -= self.frame_t
        elif key==108: # L
            self.theta += self.frame_t
        elif key==105: # I
            self.phi += self.frame_t
        elif key==107: # K
            self.phi -= self.frame_t
        elif key==99: # C
            self.eye[1] -= self.frame_t * self.speed
        elif key==32: # Space
            self.eye[1] += self.frame_t * self.speed
        elif key==46: # >
            self.speed+=self.frame_t
        elif key==44: # <
            self.speed-=self.frame_t
        else:
            print(key)
    
    def mouse_scroll_event(self, x_offset, y_offset):
        # print(x_offset, y_offset)
        self.eye += self.oriental * y_offset * self.speed

    def mouse_drag_event(self, x, y, dx, dy):
        if self.dragable:
            # print(x, y, dx, dy)
            self.theta += 0.5 * self.frame_t * dx
            self.phi += 0.5 * self.frame_t * dy

    def debug_gui(self, t, frame_t):
        import imgui
        from ..widgets import float_widget
        self.frame_t = frame_t
        if not hasattr(self, "wspeed"):
            self.wspeed = float_widget("speed", 0.01, 1, 0.1)
        
        if not hasattr(self, "wfov"):
            self.wfov = float_widget("fov", 0., 180., self.fov)

        self.speed = self.wspeed(self.speed)
        self.fov = self.wfov(self.fov)

        _, self.dragable = imgui.checkbox("camera_drag", self.dragable)
        imgui.text(f"{self.eye.astype('f2')}")
        imgui.text(f"{np.rad2deg(self.theta):.4f}, {np.rad2deg(self.phi):.4f}")
        imgui.text(f"{self.eye - self.oriental}")
