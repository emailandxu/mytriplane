# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# %%
import torch
import numpy as np
from tqdm import trange
from field import TriMipRF, VolumeRender
from glgraphics import lookAt
from generative import (
    LRMGenerator,
    kwargs_camera_embedder,
    kwargs_dino_model,
    kwargs_transformer_model,
)
from imageio import imread
from glgraphics import (
    applyMat,
    rotate_x,
    rotate_y,
    lookAt,
)

from dataset import Renderings
from matplotlib import pyplot as plt
from pathlib import Path

import wandb


# %%
FOV = 0.8575560548920328
RES = 256
LR = 1e-5
STEPS = 1000

wandb.init(project="mytriplane-generative-init", config=dict(res=RES, lr=LR, steps=STEPS))


# %%
def to_camera(c2w):
    focal_length = 0.5 / np.tan(0.5 * FOV)
    RT = torch.linalg.inv(c2w)[..., :3, :].reshape(-1, 12)  # N, 12
    camera = torch.concat(
        [
            RT,
            torch.tensor(
                [[focal_length, focal_length, 0.5, 0.5]],
                dtype=torch.float32,
                device="cuda",
            ).repeat(RT.shape[0], 1),
        ],
        dim=1,
    )  # N, 16
    return camera


generator = LRMGenerator(
    kwargs_camera_embedder, kwargs_dino_model, kwargs_transformer_model
).cuda()

volume_render = VolumeRender(
    field=TriMipRF(
        plane_size=64,
        feature_dim=40,
        planes_as_parameters=False,
    ),
    aabb=0.5,
)


datasets = [
    Renderings(dir, resolution=RES, device="cuda").to_dataset(
        flag_random_background=True
    )
    for dir in map(lambda p: p.as_posix(), Path("data/2d").glob("*"))
]


optimizer = torch.optim.Adam(
    [*generator.parameters(), *volume_render.parameters()],
    lr=LR,
)

# loss function
l1 = lambda hypo, ref: (hypo - ref).abs()
l2 = lambda hypo, ref: (hypo - ref) ** 2
progress = trange(STEPS)
loss_hist = []

# %%
try:
    for i in progress:
        dataset = datasets[i // 10 % len(datasets)]
        image_0, background, c2w_0 = dataset[0]
        image, background, c2w = dataset[i % len(dataset)]
        # torch.Size([1, 3, 40, 64, 64])
        planes = (
            generator.forward_planes(image_0, to_camera(c2w))
            .squeeze(0)
            .permute(0, 2, 3, 1)
            .contiguous()
        )
        color = volume_render.forward(
            c2w, res=RES, fov=FOV, planes=planes, background=background, far=3
        )

        optimizer.zero_grad()
        loss_map = l2(color, image.squeeze(0).permute(1, 2, 0).reshape(-1, 3)) * 1000
        loss = loss_map.mean()
        loss.backward()
        optimizer.step()

        loss_hist.append(loss.item())
        wandb.log(dict(loss=loss.item()))
        progress.desc = f"{loss.item():.4f}"

        if i % 10 == 0:
            with torch.no_grad():
                wandb.log(
                    dict(
                        input_image=wandb.Image(
                            image_0.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        ),
                        target_image=wandb.Image(
                            image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                        ),
                        hypo_image=wandb.Image(
                            color.reshape(RES, RES, 3).detach().cpu().numpy()
                        ),
                    )
                )


except KeyboardInterrupt:
    pass
# %%
with torch.no_grad():
    dataset = Renderings(
        "data/val/2d/football", resolution=RES, device="cuda"
    ).to_dataset(flag_random_background=True)
    # dataset = datasets[0]
    image_0, background, c2w_0 = dataset[0]
    image, background, c2w = dataset[3]
    planes = (
        generator.forward_planes(image_0, to_camera(c2w))
        .squeeze(0)
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    color = volume_render.forward(
        c2w, res=RES, fov=FOV, planes=planes, background=background, far=3
    )


    wandb.log(
    dict(
        input_image=wandb.Image(
            image_0.squeeze(0).permute(1, 2, 0).cpu().numpy()
        ),
        target_image=wandb.Image(
            image.squeeze(0).permute(1, 2, 0).cpu().numpy()
        ),
        hypo_image=wandb.Image(
            color.reshape(RES, RES, 3).detach().cpu().numpy()
        ),
    )
)
# %%
