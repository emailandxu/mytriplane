
from typing import List
import imgui
import numpy as np


import moderngl as mgl
from moderngl_window import geometry, resources, run_window_config
from moderngl_window.opengl.vao import VAO

from .xobj import XObj
from .base import WindowBase
from .camera import FPSCamera, Camera

from .utils.mathutil import *
from .utils.meshutil import makeCoord, makeGround, applyMat
from .widgets import *

class Window(WindowBase):
    def __init__(self, ctx: "mgl.Context" = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.xobjs = []
        self.xtasks = {}
        self.camera = FPSCamera()
        self.default_prog = self.load_program("default.glsl")
        self.pcd_prog = self.load_program("pcd.glsl")

    def setAxis(self, vertices:List[np.ndarray], name=None):
        """assume vertices and colors in shape (n, 3, 6),
         axis is defined by three six point from 3 point3 to to 3 point3 """

        vertices = np.concatenate(vertices, axis=0).reshape(-1, 3, 6)

        assert len(vertices.shape) == 3 and vertices.shape[1:] == (3, 6)

        n = vertices.shape[0]
        colors = np.array([[[1, 0, 0, 1] * 2, [0, 1, 0, 1] * 2, [0, 0, 1, 1] * 2]], dtype="f4").repeat(n, axis=0) # n x 3 x 6
       
        xobj = XObj("axis" + (f"_{name}" if name else "") )
        vao = VAO(mode=mgl.LINES)

        vao.buffer(np.array(vertices, dtype="f4"), '3f', 'in_position')
        vao.buffer(np.array(colors,  dtype="f4"), '4f', 'in_rgba')

        xobj.bind_vao(vao)
        xobj.bind_prog(self.pcd_prog)       
        self.xobjs.append(xobj)
        return xobj
    
    def setPlane(self, tex, center=None, quat=None, scale=None, name=None):
        xobj = XObj("plane" + (f"_{name}" if name else ""))

        vao = geometry.quad_fs()
        texture = self.ctx.texture(tex.shape[:2], tex.shape[2], data=np.ascontiguousarray(tex[::-1]))

        xobj.bind_vao(vao)
        xobj.bind_prog(self.default_prog)
        xobj.bind_texture(texture)

        xobj.scale = scale if scale is not None else np.array([1, 1, 1])
        xobj.center = center if center is not None else np.array([0, 0, 0])
        xobj.quat = quat if quat is not None else np.array([0, 0, 0, 1])

        if xobj.center is not None:
            mat = np.identity(4)
            mat[:3, 3] = xobj.center
            xobj.posemat = mat

        self.xobjs.append(xobj)
        return xobj
    
    def setPoints(self, points, rgbs=None, center=None, quat=None, scale=None, name=None):
        if rgbs is None:
            rgbs = np.ones_like(points)

        if rgbs.shape[-1] == 3:
            rgbs = np.concatenate([rgbs, np.ones_like(rgbs[..., [0]])], axis=-1)

        xobj = XObj("points" + (f"_{name}" if name else ""))
        vao = VAO(mode=mgl.POINTS)
        vbo = vao.buffer(np.array(points, dtype="f4"), "3f", "in_position")
        cbo = vao.buffer(np.array(rgbs, dtype="f4"), "4f", "in_rgba")

        xobj.bind_vao(vao)
        xobj.bind_vbo(vbo)
        xobj.bind_cbo(cbo)

        xobj.bind_prog(self.pcd_prog)

        xobj.scale = scale if scale is not None else np.array([1, 1, 1])
        xobj.center = center if center is not None else np.array([0, 0, 0])
        xobj.quat = quat if quat is not None else np.array([0, 0, 0, 1])

        self.xobjs.append(xobj)
        return xobj

    def setLines(self, vertices, rgbs=None):
        """start point pair with end point"""
        if rgbs is None:
            rgbs = np.array([[1, 1, 1, 1]]).astype("f4").repeat(len(vertices),1)

        if rgbs.shape[-1] == 3:
            rgbs = np.concatenate([rgbs, np.ones_like(rgbs[..., [0]])], axis=-1)

        xobj = XObj("ground")
        vao = VAO(mode=mgl.LINES)
        
        vbo = vao.buffer(np.array(vertices, dtype="f4"), '3f', 'in_position')
        cbo = vao.buffer(np.array(rgbs,  dtype="f4"), '4f', 'in_rgba')

        xobj.bind_vbo(vbo)
        xobj.bind_cbo(cbo)
        xobj.bind_vao(vao)
        xobj.bind_prog(self.pcd_prog)       
        self.xobjs.append(xobj)
        return xobj

    def setGround(self, vertices=None, colors=None):
        if vertices is None:
            vertices = makeGround()

        return self.setLines(vertices, colors)
        

    

    def key_event(self, key, action, modifiers):
        super().key_event(key, action, modifiers)
        if action=="ACTION_PRESS" or action == 768:
            self.xtasks[key] = lambda : self.camera.key_event(key, action, modifiers)
        elif key in self.xtasks:
            self.xtasks.pop(key)
        else:
            print(key, action)
            
    def mouse_scroll_event(self, x_offset, y_offset):
        super().mouse_scroll_event(x_offset, y_offset)
        self.camera.mouse_scroll_event(x_offset, y_offset)

    def mouse_drag_event(self, x, y, dx, dy):
        super().mouse_drag_event(x, y, dx, dy)
        self.camera.mouse_drag_event(x, y, dx, dy)

    def render_xobjs(self):
        for xobj in self.xobjs:
            # xobj.quat = mat2quat(rotate_x(t) @ rotate_y(t))
            xobj.render(self.camera.view, self.camera.proj)
    
    def xrender(self, t, frame_t):
        imgui.text(f"{1/frame_t:.4f}")
        self.camera.debug_gui(t, frame_t)

        for idx, xobj in enumerate(self.xobjs):
            # imgui.text(xobj.name)
            _, xobj.visible = imgui.checkbox(str(idx) + ":" + xobj.name, xobj.visible)
            imgui.same_line()
        # for above same line
        imgui.text("")

        # assert len(self.xobjs) > 0

        for xtask in self.xtasks.values():
            xtask()


    @classmethod
    def run(cls):
        run_window_config(cls, args=["-wnd", "pygame2"])