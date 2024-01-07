class CameraBase():

    def proj(self):
        raise NotImplementedError()

    def view(self):
        raise NotImplementedError()

    def key_event(self, key, action, modifiers):
        pass

    def mouse_drag_event(self, x, y, dx, dy):
        pass

    def mouse_scroll_event(self, x_offset, y_offset):
        pass

    def debug_gui(self, t, frame_t):
        pass