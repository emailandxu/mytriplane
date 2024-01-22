#%%
import torch
import numpy as np
from matplotlib import pyplot as plt
from generative import (
    LRMGenerator,
    kwargs_camera_embedder,
    kwargs_dino_model,
    kwargs_transformer_model,
)


path = "/home/xushuli/git-repo/OpenLRM/.cache/lrm-base-obj-v1.pth"
checkpoint = torch.load(path)["weights"]
keys = list(checkpoint.keys())
for key in keys:
    if "synthesizer" in key:
        checkpoint.pop(key)
#%%
generator = LRMGenerator(
    kwargs_camera_embedder, kwargs_dino_model, kwargs_transformer_model
).cuda()


#%%