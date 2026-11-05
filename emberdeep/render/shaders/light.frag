#version 330 core
#include "common.glsl"
in vec2 v_lpos;
flat in vec4 v_lrad;
flat in vec4 v_lcol;
uniform sampler2D u_galb;
uniform sampler2D u_gnrm;
uniform sampler2D u_sdf;
uniform sampler2D u_dark;
uniform vec2 u_mapsize;
out vec4 o;

float soft_shadow(vec2 p, vec2 l, float wallpix, float k) {
    vec2 d = l - p;
    float dist = length(d);
    if (dist < 1e-3) return 1.0;
    vec2 dir = d / dist;
    float t = 0.05;
    if (wallpix > 0.5) {
        float s0 = texture(u_sdf, p / u_mapsize).r;
        t = max(0.05, -s0 + 0.1);
    }
    float res = 1.0;
    for (int i = 0; i < STEPS; i++) {
        float s = texture(u_sdf, (p + dir * t) / u_mapsize).r;
        res = min(res, k * s / (t + 0.12));
        t += clamp(s, 0.07, 0.7);
        if (t >= dist - 0.08 || res < 0.0) break;
    }
    return smoothstep(0.0, 1.0, clamp(res, 0.0, 1.0));
}

void main() {
    ivec2 ip = ivec2(gl_FragCoord.xy);
    vec2 p = frag_world();
    vec2 d = v_lpos - p;
    float dist = length(d);
    float bright = v_lrad.x, total = v_lrad.y;
    if (dist >= total) discard;
    vec4 ga = texelFetch(u_galb, ip, 0);
    vec4 gn = texelFetch(u_gnrm, ip, 0);
    vec3 N = vec3(gn.xy * 2.0 - 1.0, 0.0);
    N.z = sqrt(max(1.0 - dot(N.xy, N.xy), 0.05));
    float t1 = smoothstep(bright * 0.35, bright, dist);
    float t2 = smoothstep(bright * 0.8, total, dist);
    float att = (1.0 - 0.6 * t1) * (1.0 - t2);
    att *= att * (3.0 - 2.0 * att) * 0.35 + att * 0.65;
    float h = v_lrad.z + dist * 0.22;
    vec3 Ld = normalize(vec3(d, h));
    float diff = clamp(dot(N, Ld) / Ld.z, 0.0, 2.2);
    diff = mix(diff, 1.0, 0.25);
    float sh = soft_shadow(p, v_lpos, gn.z, v_lcol.w);
    float dk = texture(u_dark, p / u_mapsize).r;
    float kk = att * sh * (1.0 - dk);
    vec3 H = normalize(Ld + vec3(0.0, 0.0, 1.0));
    float spec = pow(max(dot(N, H), 0.0), 18.0 + 90.0 * (1.0 - ga.a)) * (1.0 - ga.a) * 0.55;
    vec3 col = v_lcol.rgb * kk * (ga.rgb * diff + vec3(spec));
    o = vec4(col, 1.0);
}
