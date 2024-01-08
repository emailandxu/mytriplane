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

        prefix = "my"

        def vizcoord():
            pts = np.load(f"data/coord/{prefix}_coord.npy")
            try:
                colors = np.load(f"data/coord/{prefix}_rgb.npy")
            except:
                colors = np.ones_like(pts)

            print(pts.shape, colors.shape)
            if len(pts.shape) == 3:
                pts = pts[0]
                colors = colors[0]
            choice = np.random.choice(pts.shape[0], int(pts.shape[0]*5e-2), replace=False)
            pts = pts[choice]
            colors = colors[choice]
            print(pts.shape, colors.shape, choice.shape)
            self.setPoints(pts, colors)

        def vizdir():
            dirs = np.load(f"data/coord/{prefix}_world_rel_points.npy")
            print(dirs.shape)
            if len(dirs.shape) == 3:
                dirs = dirs[0]
            colors = np.array([[1, 0, 0]]).repeat(len(dirs), axis=0)
            self.setPoints(dirs, colors)

            dirs = np.load(f"data/coord/{prefix}_cam_rel_points.npy")
            print(dirs.shape)
            if len(dirs.shape) == 3:
                dirs = dirs[0]
            dirs = dirs[..., :3]
            colors = np.array([[0, 1, 0]]).repeat(len(dirs), axis=0)
            self.setPoints(dirs, colors)

            dirs = np.load(f"data/coord/{prefix}_direction.npy")
            print(dirs.shape)
            if len(dirs.shape) == 3:
                dirs = dirs[0]
            dirs = dirs[..., :3]
            colors = np.array([[0, 0, 1]]).repeat(len(dirs), axis=0)
            self.setPoints(dirs, colors)


        def vizcam():
            cam2world = np.load(f"data/coord/{prefix}_cam2world.npy")
            if len(cam2world.shape) == 3:
                cam2world = cam2world[0]
            self.setAxis(applyMat(cam2world, makeCoord()))

        vizdir()
        vizcam()
        vizcoord()


    def xrender(self, t, frame_t):
        super().xrender(t, frame_t)
        self.render_xobjs()
    

CamVizWindow.run()