import moderngl as mgl
from glgraphics import Window, makeCoord, makeGround, applyMat, rotate_x
from dataset import Renderings
from options import dataset_kwargs

import numpy as np
import torch
from render.camera_utils import LookAtPoseSampler

device = "cpu"
cam2world_matrix = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.0], device=device), radius=2.7, device="cpu") # 1, 4, 4
cam2world_matrix = cam2world_matrix.numpy().squeeze(0)
print(cam2world_matrix.shape)

renderings = Renderings(**dataset_kwargs)
print(len(renderings)//2)
get_cam2world = lambda idx: renderings.get(idx)[1].squeeze(0)

class CamVizWindow(Window):
    def __init__(self, ctx: "Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)

        self.setGround(makeGround())
        self.setAxis(makeCoord())

        self.setAxis(applyMat(cam2world_matrix, makeCoord()))

        for i in range(len(renderings)//2):
            mat = get_cam2world(i)
            self.setAxis(applyMat(mat, makeCoord()))


    def xrender(self, t, frame_t):
        super().xrender(t, frame_t)
        self.render_xobjs()
    

CamVizWindow.run()