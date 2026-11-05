#version 330 core
in vec2 v_uv;
uniform sampler2D u_src;
uniform vec2 u_texel;      // 1/src size
uniform float u_first;
uniform float u_threshold;
out vec4 o;
vec3 pre(vec3 c) {
    if (u_first < 0.5) return c;
    float l = max(c.r, max(c.g, c.b));
    float k = max(l - u_threshold, 0.0) / max(l, 1e-4);
    return c * k;
}
void main() {
    vec2 t = u_texel;
    vec3 c = pre(texture(u_src, v_uv).rgb) * 4.0;
    c += pre(texture(u_src, v_uv + vec2(-t.x, -t.y)).rgb);
    c += pre(texture(u_src, v_uv + vec2(t.x, -t.y)).rgb);
    c += pre(texture(u_src, v_uv + vec2(-t.x, t.y)).rgb);
    c += pre(texture(u_src, v_uv + vec2(t.x, t.y)).rgb);
    o = vec4(min(c / 8.0, vec3(30.0)), 1.0);
}
