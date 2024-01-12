
# H3D Triplane

This is an nerf implentation using Triplane with a debug view that can visualize the sample points along the rays.


## Enviorment
to do
## How to Run?
### First

place a glb model into data/3d/example.glb.

### Second
```
# to rendering directory
cd rendering

# download blender for rendering
wget https://download.blender.org/release/Blender3.2/blender-3.2.2-linux-x64.tar.xz && \
tar -xf blender-3.2.2-linux-x64.tar.xz && \
rm blender-3.2.2-linux-x64.tar.xz

# render
sh render.sh
```
it will render the the 3d glb model into data/2d/example

### Third
```
python main.py
```

## Reference
[Tri-MipRF](https://wbhu.github.io/projects/Tri-MipRF/)