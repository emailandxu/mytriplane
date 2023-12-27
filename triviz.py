import moderngl as mgl
import moderngl_window as mglw
from glgraphics import Window, makeCoord, makeGround, applyMat, lookAt, rotate_x, XObj
from dataset import Renderings
from options import dataset_kwargs

import numpy as np
import torch
from render.camera_utils import LookAtPoseSampler
from main import make_train, imagify, numpify
import torch.nn.functional as F
import imgui

RES = 512

render = make_train()["restore"]()

def pad(image, target_size=RES):
    current_size = image.shape[2] 
    padding_size = max(0, target_size - current_size)

    padding_left = padding_size // 2
    padding_right = padding_size - padding_left
    padded_image = F.pad(image, (padding_left, padding_right, padding_left, padding_right))
    return padded_image

def resize(image, target_size=RES):
    _, _, height, width = image.size()
    resized_image = F.interpolate(image, size=(target_size, target_size), mode='bilinear', align_corners=False)
    return resized_image

class CamVizWindow(Window):
    resolution = RES
    window_size = (resolution, resolution)
    aspect_ratio = window_size[0] / window_size[1]

    def __init__(self, ctx: "Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)

        self.vao = mglw.geometry.quad_fs()
        self.texture = self.ctx.texture([self.resolution, self.resolution], 3)
        self.prog = self.load_program("default.glsl")
        self.quad_view = lookAt(np.array((0, 0, -1)), np.array((0, 0, 0)), np.array((0, 1, 0)))

        self.xobj = XObj("volume_rendering")
        self.xobj.bind_vao(self.vao)
        self.xobj.bind_texture(self.texture)
        self.xobj.bind_prog(self.prog)

        self.setGround(makeGround())

    def render_triplane(self, cam2world):
        cam2world = torch.from_numpy(cam2world).unsqueeze(0).cuda()
        image, depth = render(cam2world)
        image = resize(image)
        image = numpify(imagify(image)).astype(np.uint8)
        image = np.ascontiguousarray(image)
        self.texture.write(image)
        self.xobj.render(self.quad_view, self.camera.proj)

    def xrender(self, t, frame_t):
        super().xrender(t, frame_t)
        # self.render_xobjs()

        cam2world = self.camera.view.copy()
        cam2world[:3, 3] = -cam2world[:3, :3].transpose() @ cam2world[:3, 3] 
        cam2world[:3, :3] = cam2world[:3, :3].transpose() @ rotate_x(-np.pi)[:3, :3]

        self.render_triplane(cam2world)
    

CamVizWindow.run()