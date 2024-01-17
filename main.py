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
import torch
import nerfacc
import torch.nn.functional as F
from matplotlib import pyplot as plt
from tqdm import trange
import imgui
from functools import partial
from contextlib import contextmanager

DEVICE = "cuda"
PLANE_SIZE = 92
PLANE_FEAT = 16
RES = 128
PREVIEW_RES = 128
NEARPLANE = 1.0
FARPLANE = 2.7
RENDERSTEP = (FARPLANE - NEARPLANE) / 92
AABB = 0.5
DATASET = "data/2d/example"


dataset = Renderings(DATASET, resolution=RES, device=DEVICE).to_dataset(
    flag_to_tensor=True,
    flag_random_background=True,
)

field = TriMipRF(
    plane_size=PLANE_SIZE,
    feature_dim=PLANE_FEAT,
    planes_as_parameters=True,
).cuda()

# the renderings aabb
aabb = torch.tensor([-AABB, -AABB, -AABB, AABB, AABB, AABB], device=DEVICE)
estimator = nerfacc.OccGridEstimator(
    roi_aabb=[0, 0, 0, 1, 1, 1]
).cuda()  # due to nvdiffrast texture uv sample, it must be in 0-1

# optimizer
lr_base = 1e-3
lr_ramp = 0.00001
lr_lambda = lambda x: lr_ramp ** (float(x) / float(10000))
optimizer = torch.optim.Adam(
    [*field.parameters()],
    lr=lr_base,
)

# loss function
l1 = lambda hypo, ref: (hypo - ref).abs()
l2 = lambda hypo, ref: (hypo - ref) ** 2


def sigma_fn(t_starts, t_ends, ray_indices, rays_o, rays_d, field):
    """Define how to query density for the estimator."""
    positions = (
        rays_o[ray_indices] + rays_d[ray_indices] * (t_starts + t_ends)[:, None] / 2.0
    )
    # print(positions, positions.max(), positions.min())
    sigmas = field.query_density(x=positions)["density"]
    # print(positions, sigmas)
    return sigmas.squeeze(-1)  # (n_samples,) # sigmas must have shape of (N,)


def rgb_sigma_fn(t_starts, t_ends, ray_indices, rays_o, rays_d, field):
    positions = (
        rays_o[ray_indices] + rays_d[ray_indices] * (t_starts + t_ends)[:, None] / 2.0
    )
    res = field.query_density(
        x=positions,
        return_feat=True,
    )
    density, feature = res["density"], res["feature"]
    rgb = field.query_rgb(dir=rays_d[ray_indices], embedding=feature)["rgb"]
    return rgb, density.squeeze(-1)  # sigmas must have shape of (N,)


def contraction(x, aabb):
    aabb_min, aabb_max = aabb[:3], aabb[3:]
    x = (x - aabb_min) / (aabb_max - aabb_min)
    return x

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

    @contextmanager
    def debugviz(
        self,
    ):
        do_debug = self.wviz()

        if not hasattr(self, "osample_points"):
            self.ocamline = self.setLines(
                np.array([[0, 0, 0], [1, 2, 3]]), np.array([[0, 1, 0], [1, 0, 0]])
            )
            self.oaabb_axis = self.setAxis(
                contraction(makeCoord() / (AABB * 2), aabb.cpu().numpy())
            )
            self.osample_points = self.setPoints(np.zeros((64 * 64 * 64, 3)))
            self.wgrid_sample = bool_widget("grid_sample", True)

        self.ocamline.visible = do_debug
        self.oaabb_axis.visible = do_debug
        self.osample_points.visible = do_debug
        self.oworld_axis.visible = do_debug

        yield (
            do_debug,
            self.ocamline,
            self.oaabb_axis,
            self.osample_points,
            self.wgrid_sample,
        )

        self.ocamline.visible = do_debug
        self.oaabb_axis.visible = do_debug
        self.osample_points.visible = do_debug

    @torch.no_grad()
    def test_render(self, t, frame_t):
        w2c = lookAt(
            eye=applyMat(rotate_x(t) @ rotate_y(t), np.array([1, 1, 1]))
            + np.array([0.5, 0.5, 0.5]),
            at=np.array([0.5, 0.5, 0.5]),
            up=np.array([0, 1, 0]),
        )
        c2w = np.linalg.inv(w2c)
        c2w = torch.from_numpy(c2w).cuda()

        ray = to_pinhole(
            fov=0.8575560548920328, res_w=PREVIEW_RES, res_h=PREVIEW_RES
        ).build(DEVICE)
        rays_o = ray.origins.reshape(-1, 3) + c2w[:3, 3]
        rays_d = (c2w[:3, :3] @ ray.directions.reshape(-1, 3).T).T
        # print(torch.nonzero(estimator.binaries).shape)
        ray_indices, t_starts, t_ends = estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=partial(sigma_fn, rays_o=rays_o, rays_d=rays_d, field=field),
            near_plane=NEARPLANE,
            far_plane=FARPLANE,
            early_stop_eps=1e-2,
            alpha_thre=1e-2,
            render_step_size=RENDERSTEP / 2,
            stratified=False
            # early_stop_eps=1e-4, alpha_thre=1e-2,
        )

        imgui.text(f"ray indices:{ray_indices.shape[0]}")
        with torch.no_grad():
            color, opacity, depth, extras = nerfacc.rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=rays_o.shape[0],
                rgb_sigma_fn=partial(rgb_sigma_fn, rays_o=rays_o, rays_d=rays_d, field=field),
            )

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

        ray: RayBundle = to_pinhole(fov=0.8575560548920328, res_w=RES, res_h=RES).build(
            DEVICE
        )
        rays_o = ray.origins.reshape(-1, 3) + contraction(c2w[0, :3, 3], aabb)
        rays_d = (c2w[0, :3, :3] @ ray.directions.reshape(-1, 3).T).T
        estimator.update_every_n_steps(
            step=self.step,
            occ_eval_fn=lambda x: field.query_density(x)["density"],
            occ_thre=1e-2,
        )

        # print(torch.nonzero(estimator.binaries).shape)
        ray_indices, t_starts, t_ends = estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=partial(sigma_fn, rays_o=rays_o, rays_d=rays_d, field=field),
            near_plane=NEARPLANE,
            far_plane=FARPLANE,
            early_stop_eps=1e-4,
            alpha_thre=1e-4,
            render_step_size=RENDERSTEP,
            stratified=True
            # early_stop_eps=1e-4, alpha_thre=1e-2,
        )

        assert ray_indices.shape[0] > 0, "estimator doesn't allow any sample points"

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

        with self.debugviz() as (
            do_debug,
            ocamline,
            oaabb_axis,
            osample_points,
            wgrid_sample,
        ):
            if do_debug:
                center_index = RES // 2
                center_point_rayd = (
                    rays_d.detach()
                    .reshape(RES, RES, 3)[center_index, center_index]  # center point
                    .cpu()
                    .contiguous()
                    .numpy()
                    .astype("f4")
                )
                point_rayo = (
                    rays_o[[0]].detach().cpu().contiguous().numpy().astype("f4")
                )

                ocamline.vbo.write(
                    np.stack(
                        [
                            point_rayo + NEARPLANE,
                            point_rayo + FARPLANE * center_point_rayd,
                        ]
                    )
                )

                # if use cube grid:
                if wgrid_sample():
                    pts = (
                        torch.stack(
                            torch.meshgrid(
                                torch.linspace(0, 1, steps=32),
                                torch.linspace(0, 1, steps=32),
                                torch.linspace(0, 1, steps=32),
                            ),
                            dim=-1,
                        )
                        .reshape(-1, 3)
                        .cuda()
                    )
                else:  # else use camera ray
                    pts = (
                        rays_o[ray_indices]
                        + rays_d[ray_indices] * (t_starts + t_ends)[:, None] / 2.0
                    )

                pts = torch.nn.functional.pad(
                    pts, (0, 0, 0, 64 * 64 * 64 - len(pts)), mode="constant", value=-200
                )  # pad with -200 so that the pad point will not be see
                colors = (
                    torch.ones(pts.shape[0], 4, device=DEVICE)
                    * field.query_density(pts)["density"]
                )

                # densities = field.query_density(pts)["density"]
                osample_points.vbo.write(pts.detach().cpu().numpy().astype("f4"))
                osample_points.cbo.write(colors.detach().cpu().numpy().astype("f4"))

        if not self.wtrain():
            field.training = False
            return
        else:
            field.training = True

        try:
            self.step = step = next(self.progress)
        except StopIteration:
            return

        # Differentiable Volumetric Rendering.
        # colors: (n_rays, 3). opaicity: (n_rays, 1). depth: (n_rays, 1).
        color, opacity, depth, extras = nerfacc.rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=rays_o.shape[0],
            rgb_sigma_fn=partial(rgb_sigma_fn, rays_o=rays_o, rays_d=rays_d, field=field),
            render_bkgd=background,
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
        loss_map = (loss_map * 255).clip(0, 255).astype("u1")

        self.plane1.texture.write(hypo)
        self.plane2.texture.write(target)
        self.plane3.texture.write(loss_map)

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
        imgui.text(f"loss: {loss.item():.8f},sampled rays: {ray_indices.shape[0]}")


Debug.run()
