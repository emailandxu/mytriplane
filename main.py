# %%
import torch

from render.ray_marcher import MipRayMarcher2
from render.renderer import ImportanceRenderer
from render.ray_sampler import RaySampler
from render.camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from networks.OSGDecoder import OSGDecoder
from networks.superresolution import SuperresolutionHybrid8XDC


import numpy as np
from tqdm import tqdm, trange
from pathlib import Path
from matplotlib import pyplot as plt
from dataset import Renderings
from torch.optim import Adam
from functools import partial
from util import save_video

# %%
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
normalize = lambda t: (t - t.mean()) / t.std()
imagify = lambda t: (t.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
numpify = lambda t: t.squeeze(0).detach().cpu().numpy()
show = lambda image: plt.imshow(numpify(imagify(image)))
plt.show()


class TriPlaneDecoder(torch.nn.Module):
    def __init__(self, rendering_kwargs) -> None:
        super().__init__()
        self.decoder = OSGDecoder(
            32,
            {
                "decoder_lr_mul": rendering_kwargs.get("decoder_lr_mul", 1),
                "decoder_output_dim": 32,
            },
        ).cuda()

        self.supres = SuperresolutionHybrid8XDC(
            channels=32,
            img_resolution=rendering_kwargs.get("image_resolution", 512),
            sr_num_fp16_res=0,
            sr_antialias=rendering_kwargs["sr_antialias"]
        ).cuda()

        self.ws = torch.zeros((1, 14, 512), device="cuda", requires_grad=False)

    def __call__(
        self,
        planes,
        cam2world_matrix,
        intrinsics,
        triplane_output_res,
        ray_sampler,
        renderer,
        rendering_kwargs,
    ):
        ray_origins, ray_directions = ray_sampler(
            cam2world_matrix, intrinsics, triplane_output_res
        )
        feature_samples, depth_samples, weights_samples = renderer(
            planes, self.decoder, ray_origins, ray_directions, rendering_kwargs
        )  # channels last

        N, M, _ = ray_origins.shape
        # Reshape into 'raw' neural-rendered image
        H = W = triplane_output_res
        feature_image = (
            feature_samples.permute(0, 2, 1)
            .reshape(N, feature_samples.shape[-1], H, W)
            .contiguous()
        )
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)
        rgb_image = feature_image[
            :, :3
        ]  # raw rgb images, rest chaneels are for super resolution

        rgb_image = feature_image[:, :3]

        sr_image = self.supres(rgb_image, feature_image, self.ws, noise_mode=rendering_kwargs['superresolution_noise_mode'])


        # return rgb_image, depth_image
        return sr_image, depth_image


def make_train():
    from options import rendering_kwargs, dataset_kwargs

    # make dataset
    device = "cuda"
    triplane_output_res = 128
    renderings = Renderings(device=device, **dataset_kwargs)
    dataset = renderings.to_dataset(flag_to_tensor=True)

    # make render
    # 1, 3, 32, 256, 256
    planes = torch.randn(1, 3, 32, 256, 256, device=device, requires_grad=True)
    decoder = TriPlaneDecoder(rendering_kwargs)
    ray_sampler = RaySampler()
    importance_renderer = ImportanceRenderer()
    intrinsics = FOV_to_intrinsics(49.13434264120263, device=device).reshape(
        -1, 3, 3
    )  # 1, 3, 3

    common_args = dict(
        intrinsics=intrinsics,
        triplane_output_res=triplane_output_res,
        ray_sampler=ray_sampler,
        renderer=importance_renderer,
        rendering_kwargs=rendering_kwargs,
    )
    render = partial(decoder, planes, **common_args)

    # make optimizer
    lr_base = 1e-1
    lr_ramp = 0.001
    maxiter = 150 * len(dataset)
    optimizer = Adam([planes, *decoder.parameters()], lr=lr_base)
    lr_lambda = lambda x: lr_ramp ** (float(x) / float(maxiter))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    l1 = lambda hypo, ref: (hypo - ref).abs()
    l2 = lambda hypo, ref: (hypo - ref) ** 2

    def train(early_stop=0):
        progress = trange(maxiter)
        for iter in progress:
            j = np.random.randint(0, len(dataset))
            # rgb_image, depth_image = render(planes, cam2world_matrix, intrinsics, resolutio)
            image, cam2world_matrix = dataset[j]
            rgb_image, depth_image = render(cam2world_matrix=cam2world_matrix)
            loss_map = l1(rgb_image, image)
            loss = loss_map.mean()
            loss.backward()
            progress.desc = f"loss: {loss.item():.4f}, lr:{lr_lambda(iter):.4f}"

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # if early_stop > 0 and iter*len(dataset) + j > early_stop:
            if early_stop > 0 and iter > early_stop:
                return render
        return render

    def render_video(therender=None, cam2worlds=None):
        """if no parameter was given, it will use
        the dataset cam2worlds, and the training render"""
        if therender is None:
            therender = render
        if cam2worlds is None:
            cam2worlds = [cam2world for _, cam2world in dataset]

        images = np.stack(
            [numpify(imagify(therender(cam2world)[0])) for cam2world in cam2worlds],
            axis=0,
        )
        return images

    def post_train():
        dataset_name = os.path.basename(dataset_kwargs["rootdir"])
        images = render_video()[:, ::-1]
        save_video(
            images,
            f"data/videos/{dataset_name}.mp4",
            images.shape[1],
            fps=5,
        )
        torch.save(decoder.state_dict(), f"data/models/decoder_{dataset_name}.pt")
        np.save(f"data/models/planes_{dataset_name}.npy", planes.cpu().detach().numpy())

    def restore():
        dataset_name = os.path.basename(dataset_kwargs["rootdir"])
        decoder.load_state_dict(torch.load(f"data/models/decoder_{dataset_name}.pt"))
        # for inplace assignment
        with torch.no_grad():
            planes[:] = torch.from_numpy(
                np.load(f"data/models/planes_{dataset_name}.npy")
            ).cuda()
        return render

    def offline_render(save_path):
        cam2worlds = [
            LookAtPoseSampler.sample(
                np.pi * factor * 8,
                np.pi * (1 - factor),
                torch.zeros(3).cuda(),
                radius=1.5,
                device="cuda",
            )
            for factor in tqdm(
                np.linspace(0, 1, 300), desc="video rendering:", disable=True
            )
        ]

        images = render_video(render, cam2worlds)
        # print("encoding video..")
        save_video(images[:, ::-1], f"{save_path}", images.shape[1], fps=30)

    def view_rays(cam2world):
        ray_origins, ray_directions = ray_sampler(cam2world, intrinsics, triplane_output_res)
        depths_coarse = importance_renderer.make_depth_coarse(
            ray_origins, ray_directions, rendering_kwargs
        )
        # Coarse Pass
        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape
        sample_coordinates = (
            ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)
        ).reshape(batch_size, -1, 3)
        sample_directions = (
            ray_directions.unsqueeze(-2)
            .expand(-1, -1, samples_per_ray, -1)
            .reshape(batch_size, -1, 3)
        )
        return depths_coarse, sample_coordinates, sample_directions

    return {
        "train": train,
        "post_train": post_train,
        "restore": restore,
        "render_video": render_video,
        "dataset": lambda: dataset,
        "common_args": lambda: common_args,
        "view_rays": view_rays,
        "offline_render": offline_render,
    }


# %%

if __name__ == "__main__":
    train_meta = make_train()
    try:
        train_meta["train"]()
    except KeyboardInterrupt as e:
        pass

    # train_meta["post_train"]()
    train_meta["offline_render"]("temp.mp4")

    # train_meta["restore"]()

# %%
