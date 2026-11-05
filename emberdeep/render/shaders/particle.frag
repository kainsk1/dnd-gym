#version 330 core
#include "common.glsl"
in vec2 v_l;
flat in vec4 v_col; flat in vec4 v_a; flat in float v_shape;
out vec4 o;
void main() {
    int sh = int(v_shape + 0.5);
    float r = length(v_l);
    float a = 0.0;
    float aux = v_a.x;
    if (sh == 0) a = pow(smoothstep(1.0, 0.0, r), 2.0);
    else if (sh == 1) a = exp(-r * r * 3.5) * smoothstep(1.0, 0.85, r);
    else if (sh == 2) a = exp(-v_l.y * v_l.y * 9.0) * smoothstep(1.0, 0.0, abs(v_l.x)) * (0.4 + 0.6 * smoothstep(-1.0, 1.0, v_l.x));
    else if (sh == 3) { float n = fbm3(v_l * 2.2 + v_a.z * 13.0 + u_time * 0.15); a = smoothstep(1.0, 0.25, r) * (0.35 + 1.1 * n); }
    else if (sh == 4) { float w = max(aux, 0.02); float x = (r - (1.0 - w * 1.5)) / w; a = exp(-x * x) * smoothstep(1.0, 0.9, r); }
    else if (sh == 5) a = smoothstep(1.0, 0.7, abs(v_l.x) * 1.0 + abs(v_l.y) * 2.4);
    else if (sh == 6) a = exp(-abs(v_l.x) * 7.0 - abs(v_l.y) * 40.0) + exp(-abs(v_l.y) * 7.0 - abs(v_l.x) * 40.0) + exp(-r * r * 14.0);
    else if (sh == 7) a = smoothstep(1.0, 0.7, r);
    else if (sh == 8) { float x = smoothstep(0.15, 0.0, abs(r - 0.8)); a = x * (0.5 + 0.5 * step(0.5, fract(atan(v_l.y, v_l.x) * 2.5 + aux))); }
    else if (sh == 9) a = smoothstep(1.0, 0.7, abs(v_l.x)) * smoothstep(1.0, 0.0, abs(v_l.y) * 1.0) * (0.55 + 0.45 * smoothstep(0.6, 0.0, abs(v_l.y)));
    else if (sh == 10) { float n = vnoise(v_l * 4.0 + v_a.z * 9.0); a = smoothstep(1.0, 0.6, r) * (0.5 + 0.5 * n); }
    else if (sh == 11) { float x = abs(v_l.y) - (0.6 - abs(v_l.x) * 0.5); a = smoothstep(0.1, -0.1, x) * smoothstep(1.0, 0.8, abs(v_l.x)); }
    a *= v_col.a;
    if (a < 0.002) discard;
    o = vec4(v_col.rgb * a, a * (1.0 - v_a.y));
}
