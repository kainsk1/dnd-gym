#version 330 core
#include "common.glsl"
in vec2 in_pos;
uniform vec4 u_box;     // x0,y0,x1,y1 world
uniform vec2 u_off;
out vec2 v_w;
void main() {
    v_w = mix(u_box.xy, u_box.zw, in_pos) + u_off;
    gl_Position = vec4(w2n(v_w), 0.0, 1.0);
}
