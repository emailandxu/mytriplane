
import numpy as np
from .utils.mathutil import posemat


class XObjBase():
    def __init__(self, name="undefined") -> None:
        self.scale = np.ones(3)
        self.center = np.zeros(3) # x, y, z
        self.quat = np.array([0., 0., 0., 1.]) # x, y, z, w
        self.scale_offset = np.ones(3)

        self.name = name
        self.visible = True

    @property
    def posemat(self):
        return posemat(self.center, self.quat, self.scale)

class XObj(XObjBase):
    def __init__(self, name="undefined") -> None:
        super().__init__(name)

    def bind_vao(self, vao):
        self.vao = vao

    def bind_prog(self, prog):
        self.prog = prog

    def bind_texture(self, texture):
        self.texture = texture

    def render(self, view, proj, vao=None, prog=None):
        if not self.visible:
            return
        
        assert vao is not None or self.vao is not None
        assert prog is not None or self.prog is not None

        if vao is not None:
            self.bind_vao(vao)

        if prog is not None:
            self.bind_prog(prog)

        if hasattr(self, "texture"):
            self.texture.use(0)
            self.prog['texture0'].value = 0

        m, v, p = self.posemat, view, proj

        # note that transpose is essential, from row major to column major
        mvp = (p @ v @ m).transpose().astype("f4")
        mvp = np.ascontiguousarray(mvp) # make it contiguous
        self.prog["mvp"].write(mvp)
        # self.vao.render(self.prog)
        self.vao.render(self.prog)
