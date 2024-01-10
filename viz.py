from glgraphics import makeCoord, Window, makeGround
import math
import numpy as np

from moderngl import Context
from raycam import to_pinhole, PinholeCamera, RayBundle
import numpy as np
import math
from glgraphics import Window, makeCoord, makeGround, applyMat, rotate_x
from dataset import Renderings
from field import TriMipRF
import torch
import nerfacc
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import trange
import imgui
from functools import partial


DEVICE = "cuda"
RES = 64
renderings = Renderings(
    "data/2d/sculpture_bust_of_roza_loewenfeld", resolution=RES, device=DEVICE
).to_dataset(flag_to_tensor=True)
# torch.Size([1, 3, 64, 64]) torch.Size([1, 4, 4])

field = TriMipRF().cuda()
ray: RayBundle = to_pinhole(fov=0.8575560548920328, res_w=RES, res_h=RES).build(DEVICE)


def sigma_fn(t_starts, t_ends, ray_indices, rays_o, rays_d, aabb):
    """Define how to query density for the estimator."""
    t_origins = rays_o[ray_indices]  # (n_samples, 3)
    t_dirs = rays_d[ray_indices]  # (n_samples, 3)

    distance = (t_starts + t_ends)[:, None] / 2.0
    # print(distance)
    positions = t_origins + t_dirs * distance
    # print(positions, positions.max(), positions.min())
    sigmas = field.query_density(x=contraction(positions, aabb))["density"]
    # print(positions, sigmas)
    return sigmas.squeeze(-1)  # (n_samples,) # sigmas must have shape of (N,)


def rgb_sigma_fn(t_starts, t_ends, ray_indices, rays_o, rays_d, aabb):
    t_origins = rays_o[ray_indices]
    t_dirs = rays_d[ray_indices]

    distance = (t_starts + t_ends)[:, None] / 2.0
    positions = t_origins + t_dirs * distance
    res = field.query_density(
        x=contraction(positions, aabb),
        return_feat=True,
    )
    density, feature = res["density"], res["feature"]
    rgb = field.query_rgb(dir=t_dirs, embedding=feature)["rgb"]
    return rgb, density.squeeze(-1)  # sigmas must have shape of (N,)


def contraction(x, aabb):
    aabb_min, aabb_max = aabb[:3], aabb[3:]
    x = (x - aabb_min) / (aabb_max - aabb_min)
    return x


def save_pts(pts):
    np.save("pts.npy", pts.detach().cpu().numpy())


aabb = torch.tensor([-3, -3, -3, 3, 3, 3], device=DEVICE)
estimator = nerfacc.OccGridEstimator(roi_aabb=[0, 0, 0, 1, 1, 1]).cuda()


lr_base = 1e-3
lr_ramp = 0.00001
lr_lambda = lambda x: lr_ramp ** (float(x) / float(10000))
optimizer = torch.optim.Adam(
    [
        *field.parameters(),
    ],
    lr=lr_base,
)
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

l1 = lambda hypo, ref: (hypo - ref).abs()
l2 = lambda hypo, ref: (hypo - ref) ** 2

progress = iter(trange(1000))

class Debug(Window):
    def __init__(
        self,
        ctx: "Context" = None,
        wnd: "BaseWindow" = None,
        timer: "BaseTimer" = None,
        **kwargs,
    ):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.setAxis(makeCoord())
        # self.setGround(makeGround())
        self.camera.eye = np.array([0, 1, 10], dtype="f4")
        self.camera.fov = math.degrees(0.6911112070083618)
        self.camera.dragable = False
        # self.setPoints(np.load("pts.npy"))
        self.plane1 = self.setPlane(
            (np.random.rand(*(RES, RES, 3)) * 255).astype("u1"), center=(1.1, 0, -3)
        )
        self.plane2 = self.setPlane(
            (np.random.rand(*(RES, RES, 3)) * 255).astype("u1"), center=(-1.1, 0, -3)
        )
        self.plane3 = self.setPlane(
            (np.random.rand(*(RES, RES, 3)) * 255).astype("u1"), center=(-1.1, 2.1, -3)
        )

        self.points_rayo = self.setPoints(np.zeros((1, 3)), np.array([[0,1,0]]))
        self.points_rayd = self.setPoints(np.zeros((64 * 64, 3)), )
        self.line_cam = self.setLines(np.array([[0,0,0], [1, 2, 3]]), np.array([[0,1,0],[1,0,0]]))
        self.lr = lr_base


    def xrender(self, t, frame_t):
        super().xrender(t, frame_t)
        self.render_xobjs()

        try:
            step = next(progress)
        except StopIteration:
            return
        image, c2w = renderings[step//50 % len(renderings)]
        rays_o = ray.origins.reshape(-1, 3) + c2w[0, :3, 3]
        rays_d = (c2w[0, :3, :3] @ ray.directions.reshape(-1, 3).T).T
        

        points_rayd = rays_d.detach().cpu().contiguous().numpy().astype("f4")
        points_rayo = rays_o[[0]].detach().cpu().contiguous().numpy().astype("f4")
        # self.points_rayd.vbo.write(points_rayd)
        self.points_rayo.vbo.write(points_rayo)
        self.line_cam.vbo.write(np.stack([points_rayo[0], points_rayo[0] + 3 * points_rayd.reshape(RES, RES, 3)[RES//2, RES//2]]))

        from tqdm import trange

        with torch.no_grad():
            estimator.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: field.query_density(x)["density"],
                occ_thre=0,
            )
            # print(torch.nonzero(estimator.binaries).shape)
            ray_indices, t_starts, t_ends = estimator.sampling(
                rays_o,
                rays_d,
                sigma_fn=partial(sigma_fn, rays_o=rays_o, rays_d=rays_d, aabb=aabb),
                near_plane=0.1,
                far_plane=5.0,
                early_stop_eps=1e-4, alpha_thre=1e-4,
                render_step_size = 1e-3 * 10
                # early_stop_eps=1e-4, alpha_thre=1e-2,

            )
        assert ray_indices.shape[0] > 0
        # print("rays o, rays d:", rays_o.shape, rays_d.shape)
        # print("sampled rays:", ray_indices.shape)

        # Differentiable Volumetric Rendering.
        # colors: (n_rays, 3). opaicity: (n_rays, 1). depth: (n_rays, 1).
        color, opacity, depth, extras = nerfacc.rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=rays_o.shape[0],
            rgb_sigma_fn=partial(rgb_sigma_fn, rays_o=rays_o, rays_d=rays_d, aabb=aabb),
        )

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
        loss_map = (loss_map * 255 + 25).clip(0, 255).astype("u1")


        self.plane1.texture.write(hypo)
        self.plane2.texture.write(target)
        self.plane3.texture.write(loss_map)
        changed, self.lr = imgui.slider_float(
            "slide floats", self.lr,
            min_value=0.0, max_value=lr_base*0.1,
            format="%.8f"
        )
        if imgui.button("confirm lr"):
            print(f"setting lr {self.lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.lr
        imgui.text(f"loss: {loss.item():.8f},sampled rays: {ray_indices.shape[0]}")


Debug.run()
