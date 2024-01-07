from moderngl import Context
from raycam.cameras import PinholeCamera, RayBundle
import numpy as np
import math
from glgraphics import Window, makeCoord, makeGround, applyMat, rotate_x
from dataset import Renderings

renderings = Renderings("data/2d/free_iphone_13_pro_2021").to_dataset()

def to_pinhole(fov:float=0.6911112070083618, res_h:int=64, res_w:int=64, device:str="cpu") -> PinholeCamera:
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

cam = to_pinhole()
ray = cam.ray_bundle

class Debug(Window):
    def __init__(self, ctx: "Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.setAxis(makeCoord())
        self.setGround(makeGround())
        self.camera.fov = math.degrees(0.6911112070083618)


        image, c2w = renderings[0]
        dir = ray.directions.cpu().numpy()
        print(c2w.shape)
        print(dir.shape)

        self.setPoints((c2w[0, :3, :3] @ dir.reshape(-1, 3).T).T)
        self.setPoints(c2w[0, :3, 3], np.array([0, 1, 0]))

    def xrender(self, t, frame_t):
        super().xrender(t, frame_t)
        self.render_xobjs()

Debug.run()

