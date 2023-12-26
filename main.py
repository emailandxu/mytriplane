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
from imageio import imread_v2 as imgread
from matplotlib import pyplot as plt
from options import rendering_kwargs, dataset_kwargs
import cv2
#%%
normalize = lambda t: (t-t.mean())/t.std()
imagify = lambda t: (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
numpify = lambda t: t.squeeze(0).detach().cpu()
show = lambda image:plt.imshow(numpify(imagify(image))); plt.show()
#%%
class RenderingsDataSet():
    def __init__(self, rootdir, resolution=64, device="cpu", **kwargs) -> None:
        """
        image are in shape 1 x 512 x 512 x 4,
        the extrinsics are 1 x 4 x 4 matrix, project camera to world. 
        """
        pngs = Path(rootdir).glob("*.png")
        npys = Path(rootdir).glob("*.npy")

        self.device = device
        self.resize = lambda img : cv2.resize(img, (resolution, resolution))
        self.paired_paths = list(zip(sorted(pngs), sorted(npys)))
    
    def get(self, idx, flag_resize=True, flag_alphablend=False, flag_matrix_3x4=False, flag_matrix_world2cam=False):
        pngpath, npypath = self.paired_paths[idx]        
        # print(pngpath, npypath)

        image, extrinsic = imgread(pngpath), np.load(npypath)
        
        if flag_resize:
            image = self.resize(image)

        if flag_alphablend:
            image = image[..., :3] * image[..., [3]] # multiply alpha
        else:
            image = image[..., :3]

        if flag_matrix_3x4:
            extrinsic = np.concatenate([extrinsic, np.array([[0, 0, 0, 1]])], axis=0)

        if flag_matrix_world2cam:
            # maybe wrong, not tested
            extrinsic[:3, :3] = extrinsic[:3, :3].transpose() # from world to camera into camera to world
            eye = -extrinsic[:3, :3] @ extrinsic[:3, 3]
            extrinsic[:3, 3] = eye


        # unsqueeze and to channel first tensor
        image = image / 255
        image = image[np.newaxis, :].astype("f4") # 1 x 512 x 512 x 4
        image = torch.from_numpy(image).to(self.device).permute(0, 3, 1, 2) 
        
        extrinsic = extrinsic[np.newaxis, :].astype("f4") # 1 x 4 x 4
        extrinsic = torch.from_numpy(extrinsic).to(self.device)

        return image, extrinsic
    
    def __len__(self):
        return len(self.paired_paths)

    def __getitem__(self, idx):
        return self.get(idx)

class TriPlaneRender():
    ray_sampler = RaySampler()
    renderer = ImportanceRenderer()
    def __init__(self, rendering_kwargs) -> None:
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32}).cuda()
    
    def __call__(self, planes, cam2world_matrix, intrinsics, resolution):
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, resolution)
        feature_samples, depth_samples, weights_samples = self.renderer(planes, self.decoder, ray_origins, ray_directions, rendering_kwargs) # channels last

        N, M, _ = ray_origins.shape
        # Reshape into 'raw' neural-rendered image
        H = W = resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        rgb_image = feature_image[:, :3] # raw rgb images, rest chaneels are for super resolution

        return rgb_image, depth_image

    
#%%
device = "cuda"
fov_deg = 49.13434264120263
neural_rendering_resolution = 128
intrinsics = FOV_to_intrinsics(fov_deg, device=device).reshape(-1, 3, 3) # 1, 3, 3
cam2world_matrix = LookAtPoseSampler.sample(3.14/2, 3.14/2, torch.tensor([0, 0, 0.2], device=device), radius=2.7, device=device) # 1, 4, 4
# 1, 3, 32, 256, 256
planes = torch.randn(1, 3, 32, 256, 256, device=device, requires_grad=True)
render = TriPlaneRender(rendering_kwargs)
dataset = RenderingsDataSet(device=device, resolution=neural_rendering_resolution, **dataset_kwargs)
# image, extrinsic = dataset.get(0, flag_resize=True, flag_alphablend=False)
image, extrinsic = dataset[0]
show(image)


#%%
from torch.optim import Adam
from functools import partial

optimizer = Adam([planes], lr=1e-2)
l2 = lambda hypo, ref: (hypo - ref)**2
progress = trange(200)
image, _cam2world_matrix = dataset[0]

partial_render = partial(render, planes, intrinsics=intrinsics, resolution=neural_rendering_resolution)

for iter in progress: 
    # for image, cam2world_matrix in dataset:
    # rgb_image, depth_image = render(planes, cam2world_matrix, intrinsics, neural_rendering_resolution)
    rgb_image, depth_image = partial_render(cam2world_matrix=cam2world_matrix)
    loss_map = l2(rgb_image, image)
    loss = loss_map.mean()
    loss.backward()
    progress.desc = str(loss.item())
    optimizer.step()
    optimizer.zero_grad()

#%%
from util import save_video
images = np.stack(
    [numpify(imagify(partial_render(_cam2world_matrix)[0])) for _, _cam2world_matrix in dataset],
    axis=0,
)
save_video(images, "temp.mp4", neural_rendering_resolution, fps=5)
    
# %%
