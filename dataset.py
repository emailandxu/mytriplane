from pathlib import Path
from imageio import imread_v2 as imgread
import numpy as np
import cv2
import torch
from typing import Callable
from functools import partial

def rotate_x(a):
    s, c = np.sin(a), np.cos(a)
    return np.array([[1,  0, 0, 0], 
                     [0,  c, s, 0], 
                     [0, -s, c, 0], 
                     [0,  0, 0, 1]]).astype(np.float32)


class Renderings():
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
    
    def get(self, idx, flag_resize=True, flag_alphablend=False, flag_matrix_3x4=False, flag_to_tensor=False):
        pngpath, npypath = self.paired_paths[idx]        
        # print(pngpath, npypath)

        image, extrinsic = imgread(pngpath), np.load(npypath)
        extrinsic[:3, :3] = extrinsic[:3, :3] @ rotate_x(-np.pi)[:3, :3]
        
        if flag_resize:
            image = self.resize(image)

        if flag_alphablend:
            image = image[..., :3] * image[..., [3]] # multiply alpha
        else:
            image = image[..., :3]

        if flag_matrix_3x4:
            extrinsic = np.concatenate([extrinsic, np.array([[0, 0, 0, 1]])], axis=0)

        # unsqueeze and to channel first tensor
        image = image / 255
        image = image[np.newaxis, :].astype("f4").transpose(0, 3, 1, 2) # 1 x 3 x 512 x 512 
        extrinsic = extrinsic[np.newaxis, :].astype("f4") # 1 x 4 x 4

        if flag_to_tensor:
            image = torch.from_numpy(image).to(self.device)
            extrinsic = torch.from_numpy(extrinsic).to(self.device)

        return image, extrinsic
    
    def __len__(self):
        return len(self.paired_paths)
    
    def to_dataset(self, *args, **kwargs):
        get = partial(Renderings.get, self, *args, **kwargs)
        return GenericDataset(get, len(self))

    
class GenericDataset:
    def __init__(self, get:Callable, length:int=0):
        self.get = get
        self.length = length
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.get(idx) # it seems not pass self into the get function
    
if __name__ == "__main__":
    from options import dataset_kwargs
    from functools import partial
    
    renderings = Renderings(**dataset_kwargs)
    # get =renderings.get

    img, extr = renderings.get(0)
    print(img.shape, extr.shape)
    
    dataset = renderings.to_dataset()
    img, extr = dataset[0]
    print(img.shape, extr.shape)