#version 330 core
#include "common.glsl"
in vec2 in_pos;
in vec2 in_c;     // world centre
in vec4 in_p;     // half-size x, half-size y, rotation, shape
in vec4 in_col;   // rgb, alpha
in vec4 in_a;     // aux, additive (0..1), seed, 0
uniform sampler2D u_fog;
uniform vec2 u_mapsize;
uniform float u_usefog;
out vec2 v_l;
flat out vec4 v_col; flat out vec4 v_a; flat out float v_shape;
void main() {
    vec2 l = in_pos * 2.0 - 1.0;
    vec2 loc = rot2(in_p.z) * (l * in_p.xy);
    vec2 w = in_c + loc;
    float vis = 1.0;
    if (u_usefog > 0.5) vis = smoothstep(0.2, 0.7, texture(u_fog, in_c / u_mapsize).r);
    v_l = l; v_col = vec4(in_col.rgb, in_col.a * vis); v_a = in_a; v_shape = in_p.w;
    gl_Position = vec4(w2n(w), 0.0, 1.0);
}
