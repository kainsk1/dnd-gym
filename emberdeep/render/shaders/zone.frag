#version 330 core
#include "common.glsl"
in vec2 v_w;
uniform sampler2D u_mask;   // soft zone mask over the box
uniform vec4 u_box;
uniform vec2 u_off;
uniform int u_kind;
uniform float u_alpha;
uniform vec2 u_zc;          // zone centre (world)
uniform float u_zr;
out vec4 o;

float stars(vec2 p, float sc) {
    vec2 g = floor(p * sc), f = fract(p * sc) - 0.5;
    vec2 h = hash22(g) - 0.5;
    return smoothstep(0.12, 0.0, length(f - h * 0.6));
}

void main() {
    vec2 uv = (v_w - u_off - u_box.xy) / (u_box.zw - u_box.xy);
    float m = texture(u_mask, uv).r;
    if (m < 0.003) discard;
    vec2 p = v_w;
    float t = u_time;
    vec3 col = vec3(1.0); float a = 0.0; float add = 0.0;
    float d = length(p - u_zc);
    if (u_kind == 0) {            // web
        float n = fbm3(p * 1.3);
        vec2 q = p * 2.2 + n * 0.8;
        float l1 = smoothstep(0.06, 0.0, abs(fract(q.x + q.y * 0.5) - 0.5) - 0.44) ;
        float l2 = smoothstep(0.06, 0.0, abs(fract(q.y - q.x * 0.6) - 0.5) - 0.44);
        float l3 = smoothstep(0.05, 0.0, abs(fract(length(p - u_zc) * 1.4 + n * 0.3) - 0.5) - 0.46);
        a = clamp(l1 + l2 + l3 * 0.8, 0.0, 1.0) * (0.55 + 0.45 * n) + 0.07;
        col = vec3(0.93, 0.95, 1.0); a = clamp(a, 0.0, 0.85) * m;
    } else if (u_kind == 1) {     // fog cloud / stinking cloud
        float n = fbm4(p * 0.9 + vec2(t * 0.04, -t * 0.03)) ;
        float n2 = fbm3(p * 1.7 - vec2(t * 0.06, t * 0.02));
        a = smoothstep(0.2, 0.95, n * 0.7 + n2 * 0.5 + 0.1) * m * 0.92;
        col = vec3(0.78, 0.82, 0.88) * (0.75 + 0.35 * n2);
    } else if (u_kind == 2) {     // stinking cloud (green)
        float n = fbm4(p * 0.9 + vec2(t * 0.05, -t * 0.04));
        float n2 = fbm3(p * 1.9 - vec2(t * 0.1, t * 0.03));
        a = smoothstep(0.2, 0.95, n * 0.7 + n2 * 0.5 + 0.1) * m * 0.88;
        col = mix(vec3(0.35, 0.55, 0.12), vec3(0.65, 0.78, 0.25), n2);
    } else if (u_kind == 3) {     // darkness
        float n = fbm3(p * 1.1 + t * 0.05);
        a = smoothstep(0.25, 0.75, m) * (0.93 + 0.07 * n);
        col = vec3(0.0, 0.0, 0.02) + vec3(0.04, 0.0, 0.07) * n * 0.6;
    } else if (u_kind == 4) {     // spirit guardians: radiant ground glyph
        float rr = d / max(u_zr, 0.5);
        float ringv = smoothstep(0.08, 0.0, abs(rr - 0.93)) * 0.9 + smoothstep(0.06, 0.0, abs(rr - 0.6)) * 0.35;
        float spokes = smoothstep(0.03, 0.0, abs(fract(atan(p.y - u_zc.y, p.x - u_zc.x) / 6.2832 * 12.0 + t * 0.15) - 0.5) - 0.47) * smoothstep(0.95, 0.55, rr) * 0.25;
        a = (ringv * 0.16 + spokes * 0.25 + (1.0 - rr) * 0.05) * m;
        col = vec3(1.0, 0.88, 0.5); add = 1.0;
    } else if (u_kind == 5) {     // moonbeam
        float rr = d / max(u_zr, 0.5);
        float shaft = smoothstep(1.0, 0.0, rr);
        float n = fbm3(p * 1.4 + vec2(0.0, -t * 0.4));
        a = (shaft * (0.28 + 0.2 * n) + smoothstep(0.2, 0.0, abs(rr - 0.95)) * 0.12) * m;
        col = vec3(0.65, 0.78, 1.0); add = 1.0;
    } else if (u_kind == 6) {     // spike growth
        vec2 g = floor(p * 3.0); vec2 f = fract(p * 3.0) - 0.5;
        vec2 h = (hash22(g) - 0.5) * 0.5;
        float dd = length(f - h);
        float sp = smoothstep(0.22, 0.0, dd) * (0.55 + 0.45 * smoothstep(0.2, 0.0, dd));
        a = (sp * 0.85 + 0.12) * m;
        col = mix(vec3(0.32, 0.22, 0.12), vec3(0.65, 0.55, 0.38), smoothstep(0.1, 0.0, dd));
    } else if (u_kind == 7) {     // entangle
        float n = fbm3(p * 2.0);
        float l = smoothstep(0.12, 0.0, abs(fract(p.x * 1.2 + n * 2.0 + sin(p.y * 1.7) * 0.4) - 0.5) - 0.36);
        float l2 = smoothstep(0.12, 0.0, abs(fract(p.y * 1.4 - n * 2.0 + sin(p.x * 1.3) * 0.4) - 0.5) - 0.36);
        a = clamp(l + l2, 0.0, 1.0) * m * 0.8 + 0.1 * m;
        col = mix(vec3(0.1, 0.3, 0.06), vec3(0.25, 0.5, 0.12), n);
    } else if (u_kind == 8) {     // grease
        float n = fbm3(p * 1.5 + t * 0.03);
        a = (0.35 + 0.25 * n) * m;
        col = vec3(0.04, 0.035, 0.03) + vec3(0.2, 0.18, 0.3) * pow(n, 3.0);
    } else if (u_kind == 9) {     // silence
        float rr = d / max(u_zr, 0.5);
        a = (smoothstep(0.06, 0.0, abs(fract(rr * 3.0 - t * 0.2) - 0.5) - 0.46) * 0.25 + 0.08) * m;
        col = vec3(0.55, 0.62, 0.78);
    } else if (u_kind == 10) {    // sleet storm
        vec2 q = p * vec2(5.0, 1.5) + vec2(0.0, 1.0) * t * 4.0 + vec2(p.y * 1.2, 0.0);
        float s = smoothstep(0.88, 1.0, vnoise(q)) ;
        float s2 = smoothstep(0.9, 1.0, vnoise(q * 1.7 + 5.0));
        a = (0.12 + s * 0.55 + s2 * 0.4) * m; col = vec3(0.82, 0.9, 1.0);
    } else if (u_kind == 11) {    // wall of fire / fire
        float n = fbm4(vec2(p.x * 2.0, p.y * 2.0 + t * 1.8));
        float f = smoothstep(0.35, 0.95, n * 0.9 + m * 0.15) * m;
        col = mix(vec3(0.8, 0.18, 0.02), vec3(1.0, 0.6, 0.2), pow(f, 2.0)); a = f * 0.5; add = 0.85;
    } else if (u_kind == 12) {    // flaming sphere / daylight glow
        float rr = d / max(u_zr, 0.4);
        a = smoothstep(1.0, 0.0, rr) * m * 0.35;
        col = vec3(1.0, 0.65, 0.25) * (0.8 + 0.4 * fbm3(p * 3.0 + t)); add = 1.0;
    } else if (u_kind == 13) {    // daylight
        float rr = d / max(u_zr, 0.4);
        a = smoothstep(1.0, 0.0, rr) * m * 0.35;
        col = vec3(1.0, 0.95, 0.75); add = 1.0;
    } else if (u_kind == 14) {    // faerie fire / generic magic motes
        float s = stars(p + vec2(0.0, -t * 0.1), 5.0) * (0.6 + 0.4 * sin(t * 3.0 + p.x * 4.0));
        a = (s + 0.12) * m; col = vec3(0.7, 0.4, 1.0); add = 1.0;
    } else {                      // generic hazard tint
        float n = fbm3(p * 1.5 + t * 0.05);
        a = (0.25 + 0.2 * n) * m; col = vec3(0.6, 0.3, 0.8); add = 0.5;
    }
    a *= u_alpha;
    if (a < 0.002) discard;
    o = vec4(col * a, a * (1.0 - add));
}
