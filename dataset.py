from pathlib import Path
from imageio import imread_v2 as imgread
import numpy as np
import cv2
import torch
from typing import Callable
from functools import partial


class Renderings:
    def __init__(self, rootdir, resolution=64, device="cpu", **kwargs) -> None:
        """
        image are in shape 1 x 512 x 512 x 4,
        the extrinsics are 1 x 4 x 4 matrix, project camera to world.
        """
        pngs = Path(rootdir).glob("*.png")
        npys = Path(rootdir).glob("*.npy")

        self.device = device
        self.resize = lambda img: cv2.resize(img, (resolution, resolution))
        self.paired_paths = list(zip(sorted(pngs), sorted(npys)))

    def get(
        self,
        idx,
        flag_resize=True,
        flag_random_background=False,
    ):
        pngpath, npypath = self.paired_paths[idx]
        # print(pngpath, npypath)

        image, w2c = imgread(pngpath), np.load(npypath)

        image = image / 255

        if flag_resize:
            image = self.resize(image)

        if flag_random_background:
            opacity = image[..., [3]]
            background = np.random.rand(1, 3)
            image = image[..., :3] * opacity + (background * (1 - opacity))
        else:
            opacity = image[..., [3]]
            background = np.zeros((1, 3))
            image = image[..., :3] * opacity + (background * (1 - opacity))

        # unsqueeze and to channel first tensor
        image = (
            image[np.newaxis, :].astype("f4").transpose(0, 3, 1, 2)
        )  # 1 x 3 x 512 x 512
        # background = background[np.newaxis, :].astype("f4").transpose(0, 3, 1, 2) # 1 x 1 x 512 x 512
        w2c = w2c[np.newaxis, :].astype("f4")  # 1 x 4 x 4

        image = torch.from_numpy(image).to(self.device)
        background = torch.from_numpy(background).to(self.device)
        w2c = torch.from_numpy(w2c).to(self.device)

        return image, background, w2c

    def __len__(self):
        return len(self.paired_paths)

    def to_dataset(self, *args, **kwargs):
        get = partial(Renderings.get, self, *args, **kwargs)
        return GenericDataset(get, len(self))


from functools import lru_cache


class GenericDataset:
    def __init__(self, get: Callable, length: int = 0):
        self.get = get
        self.length = length

    def __len__(self):
        return self.length

    @lru_cache(maxsize=128)
    def __getitem__(self, idx):
        return self.get(idx)  # it seems not pass self into the get function


if __name__ == "__main__":
    from options import dataset_kwargs
    from functools import partial

    renderings = Renderings("data/2d/sculpture")
    # get =renderings.get

    img, extr = renderings.get(0)
    print(img.shape, extr.shape)

    dataset = renderings.to_dataset()
    img, extr = dataset[0]
    print(img.shape, extr.shape)
