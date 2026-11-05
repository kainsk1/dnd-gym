uniform vec2 u_center;     // world (cell) coordinates at the screen centre
uniform float u_scale;     // pixels per cell (internal resolution)
uniform vec2 u_screen;     // internal framebuffer size in pixels
uniform float u_time;

vec2 w2n(vec2 w) {
    vec2 p = (w - u_center) * u_scale * 2.0 / u_screen;
    return vec2(p.x, -p.y);
}
#ifdef FRAGMENT_SHADER
vec2 frag_world() {
    vec2 n = gl_FragCoord.xy / u_screen * 2.0 - 1.0;
    return u_center + vec2(n.x, -n.y) * u_screen * 0.5 / u_scale;
}
#endif
float hash11(float p) { p = fract(p * 0.1031); p *= p + 33.33; p *= p + p; return fract(p); }
float hash21(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
vec2 hash22(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * vec3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xx + p3.yz) * p3.zy);
}
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    float a = hash21(i), b = hash21(i + vec2(1, 0)), c = hash21(i + vec2(0, 1)), d = hash21(i + vec2(1, 1));
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}
float fbm3(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 3; i++) { s += a * vnoise(p); p = p * 2.03 + 11.7; a *= 0.5; }
    return s / 0.875;
}
float fbm4(vec2 p) {
    float s = 0.0, a = 0.5;
    for (int i = 0; i < 4; i++) { s += a * vnoise(p); p = p * 2.03 + 11.7; a *= 0.5; }
    return s / 0.9375;
}
vec3 srgb2lin(vec3 c) { return pow(max(c, vec3(0.0)), vec3(2.2)); }
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
mat2 rot2(float a) { float c = cos(a), s = sin(a); return mat2(c, s, -s, c); }
