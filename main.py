#%%
import torch

from render.ray_marcher import MipRayMarcher2
from render.renderer import ImportanceRenderer
from render.ray_sampler import RaySampler
from render.camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from networks.OSGDecoder import OSGDecoder


import numpy as np
from tqdm import tqdm, trange
from pathlib import Path
from matplotlib import pyplot as plt
from options import rendering_kwargs, dataset_kwargs
from dataset import Renderings
from torch.optim import Adam
from functools import partial
from util import save_video
#%%
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#%%
normalize = lambda t: (t-t.mean())/t.std()
imagify = lambda t: (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
numpify = lambda t: t.squeeze(0).detach().cpu()
show = lambda image:plt.imshow(numpify(imagify(image))); plt.show()
#%%

class TriPlaneDecoder(torch.nn.Module):
    def __init__(self, rendering_kwargs) -> None:
        super().__init__()
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32}).cuda()
    
    def __call__(self, planes, cam2world_matrix, intrinsics, resolution, ray_sampler, renderer):
        ray_origins, ray_directions = ray_sampler(cam2world_matrix, intrinsics, resolution)
        feature_samples, depth_samples, weights_samples = renderer(planes, self.decoder, ray_origins, ray_directions, rendering_kwargs) # channels last

        N, M, _ = ray_origins.shape
        # Reshape into 'raw' neural-rendered image
        H = W = resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        rgb_image = feature_image[:, :3] # raw rgb images, rest chaneels are for super resolution

        return rgb_image, depth_image

    
#%%
device = "cuda"
resolution = 128

cam2world_matrix = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device) # 1, 4, 4
renderings = Renderings(device=device, resolution=resolution, **dataset_kwargs)
dataset = renderings.to_dataset(flag_to_tensor=True)
image, extrinsic = dataset[0]
show(image)

#%%
# 1, 3, 32, 256, 256
planes = torch.randn(1, 3, 32, 256, 256, device=device, requires_grad=True)
decoder = TriPlaneDecoder(rendering_kwargs)
ray_sampler = RaySampler()
importance_renderer = ImportanceRenderer()
intrinsics = FOV_to_intrinsics(49.13434264120263, device=device).reshape(-1, 3, 3) # 1, 3, 3
render = partial(decoder, planes, intrinsics=intrinsics, resolution=resolution, ray_sampler=ray_sampler, renderer=importance_renderer)

#%%
loaded_decoder = TriPlaneDecoder(rendering_kwargs)
loaded_decoder.load_state_dict(torch.load("data/model.pt"))
loaded_planes = torch.from_numpy(np.load("data/planes.npy")).cuda()
loaded_render = partial(loaded_decoder, loaded_planes, intrinsics=intrinsics, resolution=resolution, ray_sampler=ray_sampler, renderer=importance_renderer)


#%%
def train():
    lr_base = 0.1
    lr_ramp = 0.01
    epochs = 150
    update_times = epochs * len(dataset)
    optimizer = Adam([planes], lr=lr_base)
    lr_lambda=lambda x: lr_ramp**(float(x)/float(update_times))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    l2 = lambda hypo, ref: (hypo - ref)**2
    progress = trange(epochs)
    for iter in progress: 
        for image, cam2world_matrix in dataset:
        # rgb_image, depth_image = render(planes, cam2world_matrix, intrinsics, resolutio)
            rgb_image, depth_image = render(cam2world_matrix=cam2world_matrix)
            loss_map = l2(rgb_image, image)
            loss = loss_map.mean()
            loss.backward()
            progress.desc = str(loss.item())
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
train()

#%%

cam2world_sequence = [_cam2world_matrix for _, _cam2world_matrix in dataset]
# images_target = np.stack(
#     [renderings.get(i, flag_to_tensor=False) for i in range(len(dataset))],
#     axis=0
# )
# save_video(images_target, "target.mp4", resolutio, fps=5)

images = np.stack(
    [numpify(imagify(loaded_render(_cam2world_matrix)[0])) for _cam2world_matrix in cam2world_sequence],
    axis=0,
)
save_video(images[:, ::-1], "temp1.mp4", resolution, fps=5)

# %%
torch.save(decoder.state_dict(), "data/model.pt")
np.save("data/planes.npy", planes.cpu().detach().numpy())
# %%
