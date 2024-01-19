from typing import Any
import torch
import nerfacc
from functools import partial
from .trimipRF import TriMipRF
from raycam import to_pinhole, RayBundle

RES = 256
AABB = 0.5
FOV = 0.8575560548920328
NEARPLANE = 1.0
FARPLANE = 2.7
RENDERSTEP = (FARPLANE - NEARPLANE) / 92


class VolumeRender(torch.nn.Module):
    def __init__(
        self,
        field: TriMipRF,
        aabb: float = AABB,
        device: str = "cuda",
    ):
        super(VolumeRender, self).__init__()
        self.field = field
        self.device = device
        self.aabb = torch.tensor(
            [-aabb, -aabb, -aabb, aabb, aabb, aabb], device=self.device
        )
        self.estimator = nerfacc.OccGridEstimator(roi_aabb=[0, 0, 0, 1, 1, 1]).to(
            self.device
        )  # due to nvdiffrast texture uv sample, it must be in 0-1

        self.step = -1

    @staticmethod
    def _sigma_fn(
        t_starts, t_ends, ray_indices, rays_o, rays_d, planes, field: TriMipRF
    ):
        """Define how to query density for the estimator."""
        positions = (
            rays_o[ray_indices]
            + rays_d[ray_indices] * (t_starts + t_ends)[:, None] / 2.0
        )
        # print(positions, positions.max(), positions.min())
        sigmas = field.query_density(x=positions, planes=planes)["density"]
        # print(positions, sigmas)
        return sigmas.squeeze(-1)  # (n_samples,) # sigmas must have shape of (N,)

    @staticmethod
    def _rgb_sigma_fn(
        t_starts, t_ends, ray_indices, rays_o, rays_d, planes, field: TriMipRF
    ):
        positions = (
            rays_o[ray_indices]
            + rays_d[ray_indices] * (t_starts + t_ends)[:, None] / 2.0
        )
        result = field.query_density(
            x=positions,
            planes=planes,
            return_feat=True,
        )
        density, feature = result["density"], result["feature"]
        rgb = field.query_rgb(dir=rays_d[ray_indices], embedding=feature)["rgb"]
        return rgb, density.squeeze(-1)  # sigmas must have shape of (N,)

    def contraction(self, x):
        aabb_min, aabb_max = self.aabb[:3], self.aabb[3:]
        x = (x - aabb_min) / (aabb_max - aabb_min)
        return x

    def to_rays(self, c2w, res, fov):
        ray: RayBundle = to_pinhole(fov=fov, res_w=res, res_h=res).build(self.device)
        N = c2w.shape[0]  # batch size
        rays_o = ray.origins.unsqueeze(0).repeat(N, 1, 1, 1).reshape(
            N, -1, 3
        ) + self.contraction(c2w[..., None, :3, 3])

        rays_d = (
            c2w[..., :3, :3]
            @ ray.directions.unsqueeze(0)
            .repeat(N, 1, 1, 1)
            .reshape(N, -1, 3)
            .permute(0, 2, 1)
        ).permute(0, 2, 1)
        return rays_o, rays_d

    def forward(
        self,
        c2w,
        res=64,
        fov=FOV,
        planes=None,
        background=None,
        near=NEARPLANE,
        far=FARPLANE,
        render_step=RENDERSTEP,
        update_estimator=True,
    ):
        # breakpoint()
        batched_rays_o, batched_rays_d = self.to_rays(c2w, res, fov)
        colors = []

        for idx, (rays_o, rays_d) in enumerate(zip(batched_rays_o, batched_rays_d)):
            if update_estimator and self.training:
                self.step += 1

                def occ_eval_fn(x):
                    # breakpoint()
                    return self.field.query_density(x, planes=planes[idx])["density"]

                self.estimator.update_every_n_steps(
                    step=self.step,  # set to 0 update immediately
                    occ_eval_fn=lambda x: occ_eval_fn(x),
                    occ_thre=1e-2,
                    n=1,
                )

            ray_indices, t_starts, t_ends = self.estimator.sampling(
                rays_o,
                rays_d,
                sigma_fn=partial(
                    self._sigma_fn,
                    rays_o=rays_o,
                    rays_d=rays_d,
                    planes=planes[idx],
                    field=self.field,
                ),
                near_plane=near,
                far_plane=far,
                early_stop_eps=1e-4,
                alpha_thre=0,  # this is related to render step
                render_step_size=render_step,
                # stratified=self.training
                stratified=True,
            )
            # breakpoint()
            assert ray_indices.shape[0] > 0, "estimator doesn't allow any sample points"

            color, opacity, depth, extras = nerfacc.rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=rays_o.shape[0],
                rgb_sigma_fn=partial(
                    self._rgb_sigma_fn,
                    rays_o=rays_o,
                    rays_d=rays_d,
                    planes=planes[idx],
                    field=self.field,
                ),
                render_bkgd=background[idx],
            )
            colors.append(color)
        return torch.stack(colors)


if __name__ == "__main__":
    import numpy as np
    from glgraphics import applyMat, lookAt, rotate_x, rotate_y

    field = TriMipRF(
        plane_size=64,
        feature_dim=16,
        planes_as_parameters=True,
    ).cuda()
    t = 0

    w2c = lookAt(
        eye=applyMat(rotate_x(t) @ rotate_y(t), np.array([1, 1, 1]))
        + np.array([0.5, 0.5, 0.5]),
        at=np.array([0.5, 0.5, 0.5]),
        up=np.array([0, 1, 0]),
    )
    c2w = np.linalg.inv(w2c)
    c2w = torch.from_numpy(c2w).unsqueeze(0).cuda()

    trirender = VolumeRender(field, device="cuda")
    color = trirender(c2w)

    print(color.shape)
    print(list(trirender.state_dict().keys()))
