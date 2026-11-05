#version 330 core
in vec2 v_uv;
uniform sampler2D u_src;
out vec4 o;
void main() { o = vec4(texture(u_src, v_uv).rgb, 1.0); }
