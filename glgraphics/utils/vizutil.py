import numpy as np
import trimesh
from matplotlib import pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
from PIL import Image
import cv2

import torch

from .mathutil import *

def show_uv(mesh):
    """Assume the mesh has uv"""
    uv = mesh.visual.uv
    fig, ax = plt.subplots()
    ax.triplot(uv[..., 0], uv[..., 1], mesh.faces, linewidth=0.1)
    ax.set_aspect('equal')
    ax.set_title('UV Coordinates')
    
    # Render the Axes content onto a Figure
    fig.canvas.draw()
    
    # Convert the Figure canvas to a PIL image
    width, height = fig.get_size_inches() * fig.dpi
    image = Image.frombytes('RGB', (int(width), int(height)), fig.canvas.tostring_rgb())
    
    plt.close(fig)

    return image

def grid(_batch_of_imgs, title=""):
    """
    assume the channel of input is at the last dimension
    """
    import math
    from torchvision.utils import make_grid
    if _batch_of_imgs.requires_grad:
        batch_of_imgs = _batch_of_imgs.detach()
    else:
        batch_of_imgs = _batch_of_imgs
    
    batch = batch_of_imgs.shape[0]
    nrow = math.ceil(batch**0.5)
    grid_img = make_grid(batch_of_imgs.permute(0, 3, 1, 2), nrow=nrow).permute(1, 2, 0)
    grid_img = (grid_img.cpu().numpy() * 255).astype("u1")
    return Image.fromarray(grid_img)

def jshow(images):
    """images is a list of PIL.Image"""
    
     # Function to display image based on index
    @interact(it=(0, len(images) - 1 ))
    def display_image(it):
        return images[it]

def jpreview_with_refs(forward, texs_hist, the_refs, res):   
    from skimage.metrics import peak_signal_noise_ratio
    proj = projection()
    model = translate(*the_refs.modpos) @ scale(the_refs.modscale)

    eyes = the_refs.eyes
    
    @interact(hist=(0, len(texs_hist)-1, 1), it=(0, len(the_refs)-1), dist=(0, 2., 0.1), blend=(0, 1, 1))
    def render(hist, it, dist, blend):
        eye = eyes[it] * dist

        #print(dist, it, eye.astype("f2"))

        view = lookAt(eye=eye, at=np.array([0,0,0]), up=np.array([0, 1, 0]))

        mvp = proj @ view @ model
        
        hypo = forward(mvp, texs_hist[hist], res)
        if isinstance(hypo, torch.Tensor):
            hypo = (hypo.squeeze().cpu().numpy() * 255)

        if blend == 0:
            result = hypo.astype("u1")
        else:
            ref = the_refs[it]
            ref = cv2.flip(cv2.resize(ref, (res, res)), 0)
    
            result = ((1 - blend) * hypo + blend * ref).astype("u1")
    
            # Calculate PSNR using skimage.metrics.peak_signal_noise_ratio
            psnr = peak_signal_noise_ratio(ref, hypo, data_range=255)
            # print(f"PSNR: {psnr} dB")
            
        return Image.fromarray(result[::-1])
    
if __name__ == "__main__":
    cici = trimesh.load("resources/cici_cloth/cici_cloth.obj", force='mesh')
    tex = (np.random.rand(512, 512, 3)*255).astype("u1")
    