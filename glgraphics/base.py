import os
import numpy as np

import imgui
import moderngl
import moderngl_window as mglw
from moderngl_window.integrations.imgui import ModernglWindowRenderer

class WindowBase(mglw.WindowConfig):
    """
    implement xrender method to draw things
    """

    gl_version = (3, 3)
    title = "XDebugging"
    window_size = (768, 512)
    aspect_ratio = window_size[0] / window_size[1]
    resizable = True

    resource_dir = os.path.join(os.path.dirname(__file__), "resources")

    def __init__(self, ctx: moderngl.Context = None, wnd: "BaseWindow" = None, timer: "BaseTimer" = None, **kwargs):
        super().__init__(ctx, wnd, timer, **kwargs)
        self.ctx = ctx
        imgui.create_context()
        self.imgui = ModernglWindowRenderer(self.wnd)

    def render(self, t: float, frame_t: float):
        """integrate imgui"""
        self.ctx.clear(0, 0, 0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        imgui.new_frame()
        # super().render(time, frame_time) # may casue not implented error
        self.xrender(t, frame_t)
        imgui.render()
        self.imgui.render(imgui.get_draw_data())

    def xrender(self, t, frame_t):
        pass

    def resource(self, name):
        thepath = os.path.join(self.resource_dir, name)
        assert os.path.exists(thepath)
        return thepath

    def key_event(self, key, action, modifiers):
        self.imgui.key_event(key, action, modifiers)

    def resize(self, width: int, height: int):
        self.imgui.resize(width, height)

    def mouse_position_event(self, x, y, dx, dy):
        self.imgui.mouse_position_event(x, y, dx, dy)

    def mouse_drag_event(self, x, y, dx, dy):
        self.imgui.mouse_drag_event(x, y, dx, dy)

    def mouse_scroll_event(self, x_offset, y_offset):
        self.imgui.mouse_scroll_event(x_offset, y_offset)

    def mouse_press_event(self, x, y, button):
        self.imgui.mouse_press_event(x, y, button)

    def mouse_release_event(self, x: int, y: int, button: int):
        self.imgui.mouse_release_event(x, y, button)

    def unicode_char_entered(self, char):
        self.imgui.unicode_char_entered(char)



