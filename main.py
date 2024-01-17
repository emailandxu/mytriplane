from glgraphics import makeCoord, Window, makeGround, bool_widget
import math
import numpy as np

from moderngl import Context
from raycam import to_pinhole, PinholeCamera, RayBundle
import numpy as np
import math
from glgraphics import (
    Window,
    makeCoord,
    makeGround,
    applyMat,
    rotate_x,
    rotate_y,
    lookAt,
)
from dataset import Renderings
from field import TriMipRF
from field import VolumeRender
import torch
import nerfacc
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import trange
import imgui
from functools import partial
from contextlib import contextmanager

DEVICE = "cuda"
PLANE_SIZE = 64
PLANE_FEAT = 16
RES = 128
PREVIEW_RES = RES * 2
AABB = 0.5
DATASET = "data/2d/jupiter"


dataset = Renderings(DATASET, resolution=RES, device=DEVICE).to_dataset(
    flag_to_tensor=True,
    flag_random_background=True,
)

field = TriMipRF(
    plane_size=PLANE_SIZE,
    feature_dim=PLANE_FEAT,
    planes_as_parameters=True,
).cuda()

volume_render = VolumeRender(
    field=field,
    aabb=AABB
) 
# optimizer
lr_base = 1e-3
lr_ramp = 0.00001
lr_lambda = lambda x: lr_ramp ** (float(x) / float(10000))
optimizer = torch.optim.Adam(
    [*volume_render.parameters()],
    lr=lr_base,
)

# loss function
l1 = lambda hypo, ref: (hypo - ref).abs()
l2 = lambda hypo, ref: (hypo - ref) ** 2

def save_pts(pts):
    np.save("pts.npy", pts.detach().cpu().numpy())


def save_field(field):
    torch.save(field.state_dict(), "field.pt")


def load_field(field):
    field.load_state_dict(torch.load("field.pt"))

class Debug(Window):
    def __init__(
        self,
        ctx: "Context" = None,
        wnd: "BaseWindow" = None,
        timer: "BaseTimer" = None,
        **kwargs,
    ):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.oworld_axis = self.setAxis(makeCoord() * 10)
        # self.setGround(makeGround())
        self.camera.eye = np.array([0, 1, 10], dtype="f4")
        self.camera.fov = math.degrees(0.6911112070083618)
        self.camera.dragable = False
        # self.setPoints(np.load("pts.npy"))
        self.plane1 = self.setPlane(
            (np.zeros((RES, RES, 3)) * 255).astype("u1"), center=(1.1, 0, -3)
        )
        self.plane2 = self.setPlane(
            (np.zeros((RES, RES, 3)) * 255).astype("u1"), center=(-1.1, 0, -3)
        )
        self.plane3 = self.setPlane(
            (np.zeros((RES, RES, 3)) * 255).astype("u1"), center=(-1.1, 2.1, -3)
        )
        self.plane4 = self.setPlane(
            (np.zeros((PREVIEW_RES, PREVIEW_RES, 3)) * 255).astype("u1"),
            center=(1.1, 2.1, -3),
        )

        self.step = 0
        self.progress = iter(trange(10000))
        self.lr = lr_base
        self.wviz = bool_widget("viz", False)
        self.wtrain = bool_widget("train", True)

    @torch.no_grad()
    def test_render(self, t, frame_t):
        w2c = lookAt(
            eye=applyMat(rotate_x(t) @ rotate_y(t), np.array([0, 0, 1.5])),
            at=np.array([0.0, 0.0, 0.0]),
            up=np.array([0, 1, 0]),
        )
        c2w = np.linalg.inv(w2c)
        c2w = torch.from_numpy(c2w).unsqueeze(0).cuda()
        volume_render.training = False
        color = volume_render(c2w, PREVIEW_RES)
        volume_render.training = True
        return color

    def xrender(self, t, frame_t):
        super().xrender(t, frame_t)
        self.render_xobjs()
        if imgui.button("load"):
            load_field(field)
        imgui.same_line()
        if imgui.button("save"):
            save_field(field)
        imgui.same_line()
        if imgui.button("reset"):
            field.planes.data = torch.zeros_like(field.planes.data)
            self.progress = iter(trange(10000))

        image, background, c2w = dataset[self.step // 1 % len(dataset)]

        if not self.wtrain():
            field.training = False
            return
        else:
            field.training = True

        try:
            self.step = step = next(self.progress)
        except StopIteration:
            return

        color = volume_render(c2w, res=RES, background=background)

        # Optimize: Both the network and rays will receive gradients
        optimizer.zero_grad()
        loss_map = l2(color, image.squeeze(0).permute(1, 2, 0).reshape(-1, 3)) * 1000
        loss = loss_map.mean()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        hypo = color.detach().reshape(RES, RES, 3).cpu().numpy()
        hypo = (hypo * 255).astype("u1")

        target = (
            image.squeeze(0)
            .permute(1, 2, 0)
            .reshape(-1, 3)
            .detach()
            .reshape(RES, RES, 3)
            .cpu()
            .numpy()
        )
        target = (target * 255).astype("u1")

        loss_map = loss_map.detach().reshape(RES, RES, 3).cpu().numpy() / 1000
        loss_map = (loss_map * 255).clip(0, 255).astype("u1")

        self.plane1.texture.write(hypo)
        self.plane2.texture.write(target)
        self.plane3.texture.write(loss_map)
        self.plane4.texture.write(
            (
                self.test_render(t, frame_t)
                .detach()
                .reshape(PREVIEW_RES, PREVIEW_RES, 3)
                .cpu()
                .numpy()
                * 255
            ).astype("u1")
        )

        changed, self.lr = imgui.slider_float(
            "slide floats",
            self.lr,
            min_value=0.0,
            max_value=lr_base * 0.1,
            format="%.8f",
        )
        if imgui.button("confirm lr"):
            print(f"setting lr {self.lr}")
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.lr
        imgui.text(f"loss: {loss.item():.8f}")


Debug.run()
