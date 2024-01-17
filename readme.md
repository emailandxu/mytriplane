
# H3D Triplane

This is an nerf implentation using Triplane with a debug view that can visualize the sample points along the rays.

![Vase Training](data/attachment/vase_training.png)

## 1. Render Dataset From GLB

place **YOUR** glb model into ```data/3d/example.glb```

```
cd rendering
sh render.sh example
```
it will render the the 3d glb model into ```data/2d/example```

## 2. Train
```
python main.py
```

## Reference
[Tri-MipRF](https://wbhu.github.io/projects/Tri-MipRF/)