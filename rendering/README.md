# 🪐 Objaverse-XL Rendering Script

![266879371-69064f78-a752-40d6-bd36-ea7c15ffa1ec](https://github.com/allenai/objaverse-xl/assets/28768645/41edb8e3-d2f6-4299-a7f8-418f1ef28029)

Scripts for rendering Objaverse-XL with [Blender](https://www.blender.org/). Rendering is the process of taking pictures of the 3D objects. These images can then be used for training AI models.

## 🖥️ Setup

1. Download Blender:

```bash
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz && \
  tar -xf blender-3.2.2-linux-x64.tar.xz && \
  rm blender-3.2.2-linux-x64.tar.xz
```

2. Render Your GLB:
given glb file name like below as ```YOUR_GLB```, it will render ../data/3d/YOUR_GLB.glb into ../data/2d/YOUR_GLB directory.
```
sh render.sh example
```

## Claim
Borrowed from objaverse-xl official rendering [page](https://github.com/allenai/objaverse-xl/tree/68df6a08e7c97379e75485f134c3d1469faae7c0/scripts/rendering)