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

DEVICE = "cuda"
RES = 48
renderings = Renderings("data/2d/free_iphone_13_pro_2021", resolution=RES, device=DEVICE).to_dataset(
    flag_to_tensor=True
)
image, c2w = renderings[0]
# torch.Size([1, 3, 64, 64]) torch.Size([1, 4, 4])
print(image.shape, c2w.shape)

field = TriMipRF().cuda()
ray: RayBundle = to_pinhole(fov=, res_w=image.shape[-2], res_h=image.shape[-1]).build(DEVICE)
rays_o = ray.origins.reshape(-1, 3) + c2w[0, :3, 3]
rays_d = (c2w[0, :3, :3] @ ray.directions.reshape(-1, 3).T).T

def sigma_fn(t_starts, t_ends, ray_indices):
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


def rgb_sigma_fn(t_starts, t_ends, ray_indices):
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
    return rgb, density.squeeze(-1) # sigmas must have shape of (N,)

def contraction(x, aabb):
    aabb_min, aabb_max = aabb[:3], aabb[3:]
    x = (x - aabb_min) / (aabb_max - aabb_min)
    return x

def save_pts(pts):
    np.save("pts.npy", pts.detach().cpu().numpy())


aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=DEVICE)
estimator = nerfacc.OccGridEstimator(
    roi_aabb=[0, 0, 0, 1, 1, 1]
).cuda()


try:
    from tqdm import trange
    progress = trange(2000)
    optimizer = torch.optim.Adam(
        [
            *field.parameters(),
        ],
        lr=1e-4,
    )
    l1 = lambda hypo, ref: (hypo - ref).abs()
    l2 = lambda hypo, ref: (hypo - ref) ** 2
    for step in progress:

        with torch.no_grad():
            estimator.update_every_n_steps(
                step=step,
                occ_eval_fn=lambda x: field.query_density(x)['density'],
                occ_thre=1e-4,
            )
            # print(torch.nonzero(estimator.binaries).shape)
            ray_indices, t_starts, t_ends = estimator.sampling(
            rays_o, rays_d, sigma_fn=sigma_fn, near_plane=0.1, far_plane=3.0,
                early_stop_eps=1e-4, alpha_thre=1e-4, stratified=True
            )

        # print("rays o, rays d:", rays_o.shape, rays_d.shape)
        # print("sampled rays:", ray_indices.shape)

        # Differentiable Volumetric Rendering.
        # colors: (n_rays, 3). opaicity: (n_rays, 1). depth: (n_rays, 1).
        color, opacity, depth, extras = nerfacc.rendering(
        t_starts, t_ends, ray_indices, n_rays=rays_o.shape[0], rgb_sigma_fn=rgb_sigma_fn
        )

        # Optimize: Both the network and rays will receive gradients
        optimizer.zero_grad()
        loss_map = l2(color, image.squeeze(0).permute(1,2,0).reshape(-1, 3))
        loss = loss_map.mean()
        loss.backward()
        optimizer.step()
        progress.desc = " ".join(("sampled rays:", str((ray_indices.shape[0])**0.5))) + ", " + \
        " ".join(("loss", str(loss.item())))
except KeyboardInterrupt:
    pass
from matplotlib import pyplot as plt

plt.imshow(image.squeeze(0).permute(1,2,0).reshape(-1, 3).detach().reshape(RES,RES, 3).cpu().numpy())
plt.show()
plt.imshow(color.detach().reshape(RES,RES, 3).cpu().numpy())
plt.show()
