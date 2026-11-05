#version 330 core
#include "common.glsl"
uniform sampler2D u_galb;
uniform sampler2D u_gemi;
uniform sampler2D u_fog;       // R visible, G explored (map space, linear)
uniform vec2 u_mapsize;
uniform vec3 u_mist;
out vec4 o;
void main() {
    ivec2 ip = ivec2(gl_FragCoord.xy);
    vec2 p = frag_world();
    vec2 warp = (vec2(vnoise(p * 1.7 + u_time * 0.05), vnoise(p * 1.7 + 31.0 - u_time * 0.04)) - 0.5) * 0.30;
    vec2 f = texture(u_fog, (p + warp) / u_mapsize).rg;
    float vis = smoothstep(0.25, 0.8, f.r);
    float ex = smoothstep(0.15, 0.6, max(f.g, f.r));
    vec3 a = texelFetch(u_galb, ip, 0).rgb;
    vec3 e = texelFetch(u_gemi, ip, 0).rgb;
    float g = luma(a);
    vec3 mem = mix(vec3(g), a, 0.35) * vec3(0.20, 0.25, 0.34) + e * 0.12;
    float n = fbm3(p * 0.55 + vec2(u_time * 0.035, -u_time * 0.022));
    vec3 mist = u_mist * (0.35 + 1.5 * n * n);
    vec3 col = mix(mist, mem, ex);
    o = vec4(col, 1.0 - vis);
}
