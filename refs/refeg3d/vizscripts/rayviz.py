import os
import sys
sys.path.insert(0, os.getcwd())

import moderngl as mgl
from glgraphics import Window, makeCoord, makeGround, applyMat, rotate_x
from dataset import Renderings
from options import dataset_kwargs

import numpy as np
import torch
from render.camera_utils import LookAtPoseSampler
from main import make_train, numpify

train_meta = make_train()

device = "cpu"

renderings = Renderings(device="cuda", **dataset_kwargs)
print(len(renderings)//2)
get_cam2world = lambda idx: renderings.get(idx, flag_to_tensor=True)[1]

class CamVizWindow(Window):
    def __init__(self, ctx: "Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)

        self.setGround(makeGround())
        self.setAxis(makeCoord())

        cam2world = get_cam2world(0)
        print(cam2world.shape)
        depths_coarse, sample_coordinates, sample_directions = train_meta["view_rays"](cam2world)
        print(depths_coarse.shape, sample_coordinates.shape, sample_directions.shape)


        pts = numpify(sample_directions)
        colors = np.array([[1,0,0]]).repeat(len(pts), axis=0)
        self.setPoints(pts, colors)

        pts = numpify(sample_coordinates)
        colors = np.array([[1,1,1]]).repeat(len(pts), axis=0)
        self.setPoints(pts, colors)

    def xrender(self, t, frame_t):
        super().xrender(t, frame_t)
        self.render_xobjs()
    

CamVizWindow.run()