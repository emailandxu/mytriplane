#version 330

#if defined VERTEX_SHADER
uniform mat4 mvp;

in vec3 in_position;
in vec4 in_rgba;
out vec4 rgba;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    rgba = in_rgba;
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;
uniform sampler2D texture0;
in vec4 rgba;

void main() {
    fragColor = rgba;
}
#endif