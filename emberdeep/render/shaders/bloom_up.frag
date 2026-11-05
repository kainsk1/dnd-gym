#version 330 core
in vec2 v_uv;
uniform sampler2D u_src;
uniform vec2 u_texel;      // 1/src size (the smaller one)
out vec4 o;
void main() {
    vec2 t = u_texel * 1.0;
    vec3 c = texture(u_src, v_uv + vec2(-t.x * 2.0, 0.0)).rgb;
    c += texture(u_src, v_uv + vec2(-t.x, t.y)).rgb * 2.0;
    c += texture(u_src, v_uv + vec2(0.0, t.y * 2.0)).rgb;
    c += texture(u_src, v_uv + vec2(t.x, t.y)).rgb * 2.0;
    c += texture(u_src, v_uv + vec2(t.x * 2.0, 0.0)).rgb;
    c += texture(u_src, v_uv + vec2(t.x, -t.y)).rgb * 2.0;
    c += texture(u_src, v_uv + vec2(0.0, -t.y * 2.0)).rgb;
    c += texture(u_src, v_uv + vec2(-t.x, -t.y)).rgb * 2.0;
    o = vec4(c / 12.0, 1.0);
}
