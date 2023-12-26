#version 330

#if defined VERTEX_SHADER
uniform mat4 mvp;

in vec3 in_position;
in vec3 in_rgb;
out vec3 rgb;

void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    rgb = in_rgb;
}

#elif defined FRAGMENT_SHADER

out vec4 fragColor;
uniform sampler2D texture0;
in vec3 rgb;

void main() {
    fragColor = vec4(rgb, 1.0);
}
#endif