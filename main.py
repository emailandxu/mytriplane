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

device = "cuda"
renderings = Renderings("data/2d/free_iphone_13_pro_2021", device=device).to_dataset(
    flag_to_tensor=True
)
image, c2w = renderings[0]
# torch.Size([1, 3, 64, 64]) torch.Size([1, 4, 4])
print(image.shape, c2w.shape)

field = TriMipRF().cuda()

ray: RayBundle = to_pinhole(res_w=image.shape[-2], res_h=image.shape[-1]).build(device)
rays_o = ray.origins.reshape(-1, 3) + c2w[0, :3, 3]
rays_d = (c2w[0, :3, :3] @ ray.directions.reshape(-1, 3).T).T
print("rays o, rays d:", rays_o.shape, rays_d.shape)

optimizer = torch.optim.Adam(
    [
        *field.parameters(),
    ],
    lr=1e-1,
)


def sigma_fn(t_starts, t_ends, ray_indices):
    """Define how to query density for the estimator."""
    t_origins = rays_o[ray_indices]  # (n_samples, 3)
    t_dirs = rays_d[ray_indices]  # (n_samples, 3)
    positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
    sigmas = field.query_density(x=contraction(positions, aabb))["density"]
    # print(positions, sigmas)
    return sigmas  # (n_samples,)


def rgb_sigma_fn(t_starts, t_ends, ray_indices):
    t_origins = rays_o[ray_indices]
    t_dirs = rays_d[ray_indices]

    positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0

    save_pts(positions)
    
    # positions = self.contraction(positions)
    res = field.query_density(
        x=contraction(positions, aabb),
        return_feat=True,
    )
    density, feature = res["density"], res["feature"]
    rgb = field.query_rgb(dir=t_dirs, embedding=feature)["rgb"]
    return rgb, density

def contraction(x, aabb):
    aabb_min, aabb_max = aabb[:3], aabb[3:]
    x = (x - aabb_min) / (aabb_max - aabb_min)
    return x

def save_pts(pts):
    np.save("pts.npy", pts.detach().cpu().numpy())


aabb = torch.tensor([-1.5, -1.5, -1.5, 1.5, 1.5, 1.5], device=device)
samples_per_ray = 48
render_step_size = (
    (aabb[3:] - aabb[:3]).max()
    * math.sqrt(3)
    / samples_per_ray
).item()
print(render_step_size)
# viz()
occ_grid = nerfacc.OccupancyGrid(
    roi_aabb=aabb
).cuda()



from tqdm import trange
progress = trange(1000)
for i in progress:
    with torch.no_grad():
        occ_grid.every_n_step(
            step=0,
            occ_eval_fn=lambda x: field.query_density(
                x=contraction(x, aabb),
            )['density']
            * render_step_size,
            occ_thre=5e-3,
        )

        ray_indices, t_starts, t_ends = nerfacc.ray_marching(
            rays_o,
            rays_d,
            scene_aabb=aabb,
            grid=occ_grid,
            sigma_fn=sigma_fn,
            render_step_size=render_step_size,
            stratified=True,
            early_stop_eps=-1,
        )

    # Differentiable Volumetric Rendering.
    # colors: (n_rays, 3). opaicity: (n_rays, 1). depth: (n_rays, 1).
    color, opacity, depth = nerfacc.rendering(
        t_starts, t_ends, ray_indices, n_rays=rays_o.shape[0], rgb_sigma_fn=rgb_sigma_fn
    )
    # Optimize: Both the network and rays will receive gradients
    optimizer.zero_grad()
    loss = F.l1_loss(color, image.squeeze(0).permute(1,2,0).reshape(-1, 3))
    loss.backward()
    optimizer.step()
    
    progress.desc = " ".join(("sampled rays:", (ray_indices.shape[0] / samples_per_ray)**0.5)) + \
    " ".join(("loss", loss.item()))


print(color.shape, color.detach().sum().item())
from matplotlib import pyplot as plt

plt.imshow(image.squeeze(0).permute(1,2,0).reshape(-1, 3).detach().reshape(64,64, 3).cpu().numpy())
plt.show()
plt.imshow(color.detach().reshape(64,64, 3).cpu().numpy())
plt.show()
