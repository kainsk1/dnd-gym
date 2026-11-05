#version 330 core
#include "common.glsl"
uniform sampler2D u_galb;
uniform sampler2D u_gemi;
uniform sampler2D u_gnrm;
uniform sampler2D u_dark;
uniform vec2 u_mapsize;
uniform vec3 u_ambient;
out vec4 o;
void main() {
    ivec2 ip = ivec2(gl_FragCoord.xy);
    vec4 a = texelFetch(u_galb, ip, 0);
    vec4 e = texelFetch(u_gemi, ip, 0);
    vec2 p = frag_world();
    float dk = texture(u_dark, p / u_mapsize).r;
    float var = (0.85 + 0.3 * vnoise(p * 0.35)) * (1.0 + 0.9 * texelFetch(u_gnrm, ip, 0).z);
    o = vec4((a.rgb * u_ambient * var + e.rgb) * (1.0 - 0.9 * dk), 1.0);
}
