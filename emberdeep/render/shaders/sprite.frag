#version 330 core
#include "common.glsl"
in vec2 v_uv;
in vec4 v_col;
in float v_rot;
uniform sampler2D u_pa;   // albedo rgb + alpha (sRGB)
uniform sampler2D u_pn;   // normal xy, roughness, emissive
layout(location = 0) out vec4 o_alb;
layout(location = 1) out vec4 o_nrm;
layout(location = 2) out vec4 o_emi;
void main() {
    vec4 a = texture(u_pa, v_uv);
    float alpha = a.a * v_col.a;
    if (alpha < 0.004) discard;
    vec4 n = texture(u_pn, v_uv);
    vec2 nxy = n.xy * 2.0 - 1.0;
    nxy = rot2(v_rot) * nxy;
    vec3 N = normalize(vec3(nxy, 1.0));
    vec3 col = srgb2lin(a.rgb) * v_col.rgb;
    o_alb = vec4(col, alpha);
    o_nrm = vec4(N.xy * 0.5 + 0.5, 0.0, alpha);
    o_emi = vec4(col * n.a * 3.0, alpha);
}
