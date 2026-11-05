#version 330 core
#include "common.glsl"
in vec2 in_pos;       // unit quad
in vec2 in_cell;      // cell origin
in vec4 in_m;         // material, variant, rotation(0..3), shape(0 floor,1 wall,2 pillar,3 low wall)
out vec2 v_world;
flat out vec4 v_m;
void main() {
    v_world = in_cell + in_pos;
    v_m = in_m;
    gl_Position = vec4(w2n(v_world), 0.0, 1.0);
}
