#version 330 core
#include "common.glsl"
in vec2 in_pos;     // unit quad
in vec2 in_c;       // centre (world)
in vec2 in_size;    // world size (cells)
in vec4 in_r;       // rot, pivot_x, pivot_y (0 = rotate about centre), unused
in vec4 in_uv;      // u0 v0 u1 v1
in vec4 in_col;     // tint rgb, alpha
out vec2 v_uv;
out vec4 v_col;
out float v_rot;
void main() {
    vec2 loc = (in_pos - 0.5) * in_size;
    float a = in_r.x;
    vec2 w = in_c + rot2(a) * loc;
    v_uv = mix(in_uv.xy, in_uv.zw, in_pos);
    v_col = in_col;
    v_rot = a;
    gl_Position = vec4(w2n(w), 0.0, 1.0);
}
