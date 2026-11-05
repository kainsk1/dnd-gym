#version 330 core
#include "common.glsl"
in vec2 v_l;
flat in vec4 v_col; flat in vec4 v_p; flat in vec4 v_q; flat in vec4 v_m; flat in float v_rad;
uniform sampler2DArray u_tok;
out vec4 o;

float ring(float r, float c, float w) { float x = (r - c) / w; return exp(-x * x); }

void main() {
    int mode = int(v_m.x + 0.5);
    float r = length(v_l);
    if (mode == 0) {
        vec2 uv = v_l / 1.1 * 0.5 + 0.5;
        vec4 t = texture(u_tok, vec3(uv, v_m.y));
        vec3 c = pow(t.rgb, vec3(2.2));
        float g = luma(c);
        c = mix(c, vec3(g) * vec3(0.92, 0.95, 1.0), v_p.x);
        c = mix(c, v_q.rgb * (0.4 + g), v_q.a);
        float a = t.a * v_col.a;
        float sh = v_p.z;
        if (sh > 0.0) {
            float s = 0.5 + 0.5 * sin(u_time * 5.0 + v_l.x * 4.0 + v_l.y * 6.0 + vnoise(v_l * 5.0 + u_time) * 5.0);
            a *= mix(1.0, 0.38 + 0.2 * s, sh);
            c += vec3(0.25, 0.4, 0.5) * s * sh * 0.25;
        }
        c *= v_col.rgb;
        c += vec3(1.0, 0.85, 0.7) * v_p.y * t.a;
        o = vec4(c * a, a);
    } else if (mode == 1) {
        float a = v_col.a * smoothstep(1.0, 0.45, r);
        o = vec4(0.0, 0.0, 0.0, a);
    } else {
        int ty = int(v_m.z + 0.5);
        float ang = atan(v_l.y, v_l.x);
        float a = 0.0; vec3 c = v_col.rgb; float add = 1.0;
        float t = u_time;
        if (ty == 0) {                                  // active turn ring
            float dash = 0.65 + 0.35 * step(0.5, fract(ang / 6.2832 * 14.0 + t * 0.35));
            a = ring(r, 1.14, 0.045) * dash * 1.4 + ring(r, 1.14, 0.22) * 0.45 + ring(r, 1.32, 0.18) * 0.12;
            a *= 0.78 + 0.22 * sin(t * 4.0);
        } else if (ty == 1) {                           // hover
            a = ring(r, 1.1, 0.04) * 0.9 + ring(r, 1.1, 0.14) * 0.25;
        } else if (ty == 2) {                           // selected
            float d2 = step(0.5, fract(ang / 6.2832 * 10.0 - t * 0.3));
            a = ring(r, 1.16, 0.035) * 1.3 + ring(r, 1.26, 0.03) * d2 + ring(r, 1.2, 0.18) * 0.22;
        } else if (ty == 3) {                           // poisoned
            float n = fbm3(v_l * 3.2 + vec2(0.0, -t * 0.7));
            a = smoothstep(0.5, 0.75, n) * smoothstep(0.6, 1.05, r) * smoothstep(1.75, 1.15, r);
        } else if (ty == 4) {                           // frightened
            float n = fbm3(vec2(ang * 1.5 + t * 0.8, r * 2.5 - t * 0.6));
            a = smoothstep(0.45, 0.8, n) * ring(r, 1.25, 0.28) * 0.9;
        } else if (ty == 5) {                           // paralysed / stunned crackle
            float n = vnoise(vec2(ang * 3.0 + floor(t * 9.0) * 7.0, r * 5.0));
            float arc = ring(r + (n - 0.5) * 0.45, 1.15, 0.035);
            a = arc * step(0.35, vnoise(vec2(ang * 2.0, floor(t * 9.0)))) * 1.6 + ring(r, 1.1, 0.25) * 0.12;
        } else if (ty == 6) {                           // petrified dust
            float n = vnoise(v_l * 14.0 + floor(t * 2.0));
            a = step(0.82, n) * ring(r, 1.05, 0.4) * 0.55; add = 0.0;
        } else if (ty == 7) {                           // restrained: web
            float rad = abs(fract(ang / 6.2832 * 8.0 + 0.5) - 0.5) * 2.0;
            float sp = smoothstep(0.05, 0.0, rad * r * 0.45);
            float conc = smoothstep(0.1, 0.0, abs(fract(r * 3.5) - 0.5) - 0.42) * step(r, 1.15);
            a = (sp * step(r, 1.18) + conc * 0.8) * 0.65; add = 0.4;
        } else if (ty == 8) {                           // rage
            float n = fbm3(vec2(ang * 2.2 + sin(t) * 0.3, r * 3.0 - t * 3.2));
            a = smoothstep(0.35, 0.8, n) * smoothstep(0.85, 1.05, r) * smoothstep(1.7, 1.1, r) * 1.1;
        } else if (ty == 9) {                           // blessed motes
            float k = 0.0;
            for (int i = 0; i < 6; i++) {
                float aa = float(i) * 1.047 + t * 0.7;
                vec2 pp = vec2(cos(aa), sin(aa)) * (1.2 + 0.1 * sin(t * 2.0 + float(i)));
                k += exp(-dot(v_l - pp, v_l - pp) * 70.0);
            }
            a = k + ring(r, 1.2, 0.2) * 0.12;
        } else if (ty == 10) {                          // orbiting stars
            float k = 0.0;
            for (int i = 0; i < 3; i++) {
                float aa = float(i) * 2.094 + t * 2.4;
                vec2 pp = vec2(cos(aa), sin(aa)) * 1.12;
                vec2 q = v_l - pp;
                k += exp(-dot(q, q) * 90.0) + exp(-abs(q.x) * 25.0 - abs(q.y) * 140.0) * 0.6 + exp(-abs(q.y) * 25.0 - abs(q.x) * 140.0) * 0.6;
            }
            a = k;
        } else if (ty == 11) {                          // shield bubble
            float fr = smoothstep(0.7, 1.15, r) * smoothstep(1.3, 1.12, r);
            float hex = 0.6 + 0.4 * vnoise(v_l * 6.0 + t * 0.5);
            a = (fr * 0.5 * hex + ring(r, 1.15, 0.03) * 0.8);
        } else if (ty == 12) {                          // charmed
            a = ring(r, 1.1, 0.3) * (0.5 + 0.5 * sin(t * 3.0)) * 0.55;
        } else if (ty == 13) {                          // generic soft glow
            a = smoothstep(1.7, 0.2, r) * (0.7 + 0.3 * sin(t * 3.5)) * 0.8;
        } else if (ty == 14) {                          // concentration
            float d2 = step(0.5, fract(ang / 6.2832 * 18.0 + t * 0.5));
            a = ring(r, 1.0, 0.025) * d2 * 0.9;
        } else if (ty == 16) {                          // facing wedge
            vec2 q = rot2(-v_m.w) * v_l;
            float w = smoothstep(0.34, 0.0, abs(q.y) + (1.0 - q.x) * 0.9) * step(0.78, q.x) * step(q.x, 1.2);
            a = w * 0.55;
        } else if (ty == 17) {                          // dying pulse
            a = ring(r, 1.05, 0.12) * (0.4 + 0.6 * pow(0.5 + 0.5 * sin(t * 2.5), 2.0));
        } else if (ty == 18) {                          // object glint ring
            a = ring(r, 0.95, 0.1) * 0.5 * (0.5 + 0.5 * sin(t * 2.0));
        }
        o = vec4(c * a * v_col.a, a * v_col.a * (1.0 - add));
    }
}
