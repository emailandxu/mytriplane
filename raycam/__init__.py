from .cameras import PinholeCamera
from .raybunddle import RayBundle
import numpy as np

def to_pinhole(fov:float=0.8575560548920328, res_h:int=64, res_w:int=64, device:str="cpu") -> PinholeCamera:
    """fov is in rad"""
    camera_angle_x = float(fov)
    focal_length = 0.5 * res_w / np.tan(0.5 * camera_angle_x)
    cx = res_w / 2.0
    cy = res_h / 2.0
    aabb = np.array([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5])
    camera = PinholeCamera(
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            width=res_w,
            height=res_h,
    )
    camera.build(device)
    return camera

