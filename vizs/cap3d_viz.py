import sys
sys.path.append("/data1/xushuli/git-repo/mytriplane/")
import os
import math
import imgui
import pickle
import numpy as np
from moderngl import Context
from glgraphics import Window, makeCoord, int_widget
from imageio import imread
from pathlib import Path

RES = 512
INFO = pickle.load(open("data/cap3d/cap3d_human_objaverse_my.pkl", "rb"))
KEYS = list(INFO.keys())
N = len(KEYS)
DIRS = [
    Path(
        "/data1/xushuli/git-repo/mytriplane/data/cap3d/cap3d_human_objaverse_my"
    ).joinpath(f"Cap3D_imgs_view{i}").as_posix()
    for i in range(8)
]


class Cap3D(Window):
    def __init__(
        self,
        ctx: "Context" = None,
        wnd: "BaseWindow" = None,
        timer: "BaseTimer" = None,
        **kwargs
    ):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.camera.eye = np.array([-1, -1, 8], dtype="f4")
        self.camera.fov = math.degrees(0.6911112070083618)
        self.camera.dragable = False

        xv, yv = np.meshgrid(np.arange(-4, 4, 2), np.arange(-2, 2, 2))
        grid = np.concatenate([xv[..., np.newaxis], yv[..., np.newaxis]], axis=-1)
        grid = grid.reshape(-1, 2)

        self.planes = []
        for pos in grid:
            print(pos)
            self.planes.append(
                self.setPlane(
                    (np.random.rand(*(RES, RES, 3)) * 255).astype("u1"), center=(*pos, -3)
                )
            )
        print(len(self.planes))
        self.index = int(0)

    def xrender(self, t, frame_t):
        super().xrender(t, frame_t)
        self.render_xobjs()

        changed, self.index = imgui.slider_int(f"slider", self.index, 0, N-1, format="%d")
        imgui.same_line()
        if imgui.button("+"):
            self.index += 1
            changed = True
        imgui.same_line()
        if imgui.button("-"):
            self.index -= 1
            changed = True

        label = INFO[KEYS[self.index]]
        imgui.text(repr(label))

        if changed:
            for i in range(8):
                path = DIRS[i] + f"/{KEYS[self.index]}"
                assert os.path.exists(path)
                image = imread(path)
                self.planes[i].texture.write(image.astype("u1"))


if __name__ == "__main__":
    Cap3D.run()


