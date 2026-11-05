#version 330 core
in vec2 v_uv;
uniform sampler2D u_ui;
out vec4 o;
void main() { o = texture(u_ui, vec2(v_uv.x, 1.0 - v_uv.y)); }
