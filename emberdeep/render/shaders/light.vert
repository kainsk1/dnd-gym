#version 330 core
#include "common.glsl"
in vec2 in_pos;
in vec2 in_lpos;
in vec4 in_lrad;     // bright, total, height, kind
in vec4 in_lcol;     // rgb (intensity premultiplied), shadow softness
out vec2 v_lpos;
flat out vec4 v_lrad;
flat out vec4 v_lcol;
void main() {
    vec2 w = in_lpos + (in_pos * 2.0 - 1.0) * in_lrad.y * 1.04;
    v_lpos = in_lpos; v_lrad = in_lrad; v_lcol = in_lcol;
    gl_Position = vec4(w2n(w), 0.0, 1.0);
}
