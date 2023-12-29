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
train_meta["restore"]()

device = "cuda"

renderings = Renderings(device="cuda", **dataset_kwargs)
print(len(renderings)//2)
get_cam2world = lambda idx: renderings.get(idx, flag_to_tensor=True)[1]

class CamVizWindow(Window):
    def __init__(self, ctx: "Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)

        self.setGround(makeGround() + np.array([[0, -3, 0]]))
        self.setAxis(makeCoord())

        cam2world = get_cam2world(0)
        print(cam2world.shape)
        # torch.Size([1, 786432, 32]) torch.Size([1, 786432, 1])
        depths_coarse, sample_coordinates, sample_directions = train_meta["view_rays"](cam2world)

        feat_coarse, densities_coarse = train_meta["view_volume"](cam2world)
        colors_coarse = feat_coarse[..., :3]
        colors = torch.concat([colors_coarse, densities_coarse], dim=-1)
        # colors = densities_coarse.repeat(1,1,3)
        print(colors, densities_coarse)
        print(feat_coarse.shape, densities_coarse.shape, colors.shape)
        print(depths_coarse.shape, sample_coordinates.shape, sample_directions.shape)

        pts = numpify(sample_coordinates[..., :3])
        # colors = np.array([[1,1,1]]).repeat(len(pts), axis=0)
        colors = numpify(colors).astype("f4")
        self.setPoints(pts, colors)


        self.setAxis(applyMat(numpify(cam2world), makeCoord()))


    def xrender(self, t, frame_t):
        super().xrender(t, frame_t)
        self.render_xobjs()
    

CamVizWindow.run()