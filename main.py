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
RES = 128
LR = 1e-5
STEPS = 7000
BATCH_SIZE = 5

wandb.init(
    project="mytriplane-generative-batched-val",
    config=dict(res=RES, lr=LR, steps=STEPS),
)


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

val_dataset = Renderings(
    "data/val/2d/football", resolution=RES, device="cuda"
).to_dataset(flag_random_background=True)
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
for i in progress:
    # ============== batchfy =================-
    temps = []
    for _ in range(BATCH_SIZE):
        dataset = datasets[np.random.randint(0, len(datasets))]
        _image_0, _, _c2w_0 = dataset[0]
        _image, _background, _c2w = dataset[np.random.randint(1, len(dataset))]
        temps.append([_image_0, _c2w_0, _image, _background, _c2w])

    image_0 = torch.concat([temp[0] for temp in temps], dim=0)
    c2w_0 = torch.concat([temp[1] for temp in temps], dim=0)
    image = torch.concat([temp[2] for temp in temps], dim=0)
    background = torch.concat([temp[3] for temp in temps], dim=0)
    c2w = torch.concat([temp[4] for temp in temps], dim=0)

    # =========================================

    N = c2w.shape[0]  # batch size
    # torch.Size([1, 3, 40, 64, 64])
    # channel first, N, 3, C, W, H
    planes = generator.forward_planes(image_0, to_camera(c2w))
    # to channel last, N, 3, W, H, C
    planes = planes.permute(0, 1, 3, 4, 2).contiguous()

    color = volume_render.forward(
        c2w, res=RES, fov=FOV, planes=planes, background=background, far=3
    )

    optimizer.zero_grad()
    loss_map = l2(color, image.permute(0, 2, 3, 1).reshape(N, -1, 3)) * 1000
    loss = loss_map.mean()
    loss.backward()
    optimizer.step()

    loss_hist.append(loss.item())
    progress.desc = f"{loss.item():.4f}"

    wandb.log(dict(loss=loss.item()))

    # ============= log validata ==============
    if i % 10 == 0:
        with torch.no_grad():
            rnd_index = np.random.randint(0, BATCH_SIZE)

            def do_val():
                # dataset = datasets[0]
                image_0, background, c2w_0 = val_dataset[0]
                image, background, c2w = val_dataset[np.random.randint(1, len(val_dataset))]
                N = c2w.shape[0]
                # channel first, N, 3, C, W, H
                planes = generator.forward_planes(image_0, to_camera(c2w))
                # to channel last, N, 3, W, H, C
                planes = planes.permute(0, 1, 3, 4, 2).contiguous()
                color = volume_render.forward(
                    c2w, res=RES, fov=FOV, planes=planes, background=background, far=3
                )
                loss_map = l2(color, image.permute(0, 2, 3, 1).reshape(N, -1, 3)) * 1000
                loss = loss_map.mean()
                return image, color, loss
            
            val_image, val_color, val_loss = do_val()

            wandb.log(
                dict(
                    input_image=wandb.Image(
                        image_0[rnd_index].permute(1, 2, 0).cpu().numpy()
                    ),
                    target_image=wandb.Image(
                        image[rnd_index].permute(1, 2, 0).cpu().numpy()
                    ),
                    hypo_image=wandb.Image(
                        color[rnd_index].reshape(RES, RES, 3).detach().cpu().numpy()
                    ),
                    val_target_image=wandb.Image(
                        val_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
                    ),
                    val_hypo_image=wandb.Image(
                        val_color.reshape(RES, RES, 3).detach().cpu().numpy()
                    ),
                    val_loss=val_loss.item(),
                )
            )

# %%
