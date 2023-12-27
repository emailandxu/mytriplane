import moderngl as mgl
import moderngl_window as mglw
from glgraphics import Window, makeCoord, makeGround, applyMat, lookAt, rotate_x, XObj
from dataset import Renderings
from options import dataset_kwargs

import numpy as np
import torch
from render.camera_utils import LookAtPoseSampler

resolution=512

class CamVizWindow(Window):
    window_size = (512, 512)
    aspect_ratio = window_size[0] / window_size[1]

    def __init__(self, ctx: "Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)

        self.vao = mglw.geometry.quad_fs()
        self.texture = self.ctx.texture([resolution, resolution], 3)
        self.pcd_prog = self.load_program("default.glsl")
        self.quad_view = lookAt(np.array((0, 0, -1)), np.array((0, 0, 0)), np.array((0, 1, 0)))

        self.xobj = XObj("volume_rendering")
        self.xobj.bind_vao(self.vao)
        self.xobj.bind_texture(self.texture)
        self.xobj.bind_prog(self.pcd_prog)

    def xrender(self, t, frame_t):
        super().xrender(t, frame_t)
        self.xobj.render(self.quad_view, self.camera.proj)
    

CamVizWindow.run()