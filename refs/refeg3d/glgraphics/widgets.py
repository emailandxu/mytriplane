import imgui
import numpy as np
from .utils.mathutil import projection, lookAt


def float4_widget(name, min, max, default_values=None):
    if default_values is None:
        default_values = (0, 0, 0, 0)

    assert len(default_values) == 4
    value = list(default_values)

    def widget(new_value=None):
        nonlocal value
        
        if new_value is not None:
            value = new_value
        
        _, value[:] = imgui.slider_float4(f"{name}", *value, min, max, format="%.4f")

        imgui.same_line()
        if imgui.button(f"reset {name}"):
            value = list(default_values)
        return value

    return widget

def float3_widget(name, min, max, default_values=None):
    if default_values is None:
        default_values = (0, 0, 0)

    assert len(default_values) == 3
    value = list(default_values)

    def widget(new_value=None):
        nonlocal value
        
        if new_value is not None:
            value = new_value
        
        _, value[:] = imgui.slider_float3(f"{name}", *value, min, max, format="%.4f")

        imgui.same_line()
        if imgui.button(f"reset {name}"):
            value = list(default_values)
        return value

    return widget


def float_widget(name, min, max, default_values=None):
    if default_values is None:
        default_values = 0.

    assert type(default_values) is float
    value = default_values

    def widget(new_value=None):
        nonlocal value
        
        if new_value is not None:
            value = new_value

        _, value = imgui.slider_float(f"{name}", value, min, max, format="%.4f")

        imgui.same_line()
        if imgui.button(f"reset {name}"):
            value = default_values
        return value

    return widget


def int_widget(name, min, max, default_value=None):
    if default_value is None:
        default_value = 0
    
    assert isinstance(default_value, int)
    value = default_value

    def widget():
        nonlocal value
        _, value = imgui.slider_int(f"{name}", value, min, max, format="%d")
        return value
    return widget

def bool_widget(name, default=False):
    bool=default
    def widget():
        nonlocal bool
        _, bool = imgui.checkbox(f"if_{name}", bool)
        return bool
    return widget

def camera_widget(aspect, fov=90.0, near=0.1, far=100.0, campos=(0, 0, 3)):       
    pos_widget = float3_widget("camera_pos", -5, 5, campos)
    lookat_widget = float3_widget("camera_lookat", -5, 5, (0, 0, 0))
    
    def move_camera():
        with imgui.begin("camera"):
            x, y, z = np.array(pos_widget())
            lookat = np.array(lookat_widget())
        proj = projection(fov=fov, aspect_ratio=aspect, near=near, far=far)
        view = lookAt(
            eye=(x, y, z),
            at=lookat,
            up=(0., 1., 0.)
        )
        light_pos = np.array([-x, y, -z])
        return proj.astype("f4"), view.astype("f4"), light_pos.astype("f4")

    return move_camera
