#version 330 core
#include "common.glsl"
in vec2 in_pos;
in vec4 in_c;     // cx, cy, radius, rotation
in vec4 in_col;   // tint rgb (lighting), alpha
in vec4 in_p;     // desat, flash, shimmer, squash
in vec4 in_q;     // condition tint rgb, mix
in vec4 in_m;     // mode(0 body,1 shadow,2 overlay), layer, overlay type, aux
out vec2 v_l;
flat out vec4 v_col; flat out vec4 v_p; flat out vec4 v_q; flat out vec4 v_m; flat out float v_rad;
void main() {
    int mode = int(in_m.x + 0.5);
    float ext = mode == 0 ? 1.1 : (mode == 1 ? 1.5 : 2.4);
    vec2 l = (in_pos * 2.0 - 1.0) * ext;
    vec2 loc = l;
    if (mode == 0) { loc.y *= in_p.w; loc = rot2(in_c.w) * loc; }
    vec2 w = in_c.xy + loc * in_c.z;
    v_l = l; v_col = in_col; v_p = in_p; v_q = in_q; v_m = in_m; v_rad = in_c.z;
    gl_Position = vec4(w2n(w), 0.0, 1.0);
}
