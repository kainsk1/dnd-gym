#version 330 core
in vec2 v_uv;
uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloom_amt;
uniform float u_exposure;
uniform vec3 u_lift;
uniform vec3 u_gain;
uniform float u_sat;
uniform float u_contrast;
uniform float u_vignette;
uniform float u_grain;
uniform float u_chroma;
uniform float u_flash;
uniform vec3 u_flash_col;
uniform float u_time;
out vec4 o;
vec3 aces(vec3 x) { return clamp((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0); }
float hash(vec2 p) { vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }
void main() {
    vec2 d = v_uv - 0.5;
    float r2 = dot(d, d);
    vec2 off = d * (u_chroma * (0.4 + 2.5 * r2));
    vec3 c;
    if (u_chroma > 0.0004) {
        c = vec3(texture(u_scene, v_uv + off).r, texture(u_scene, v_uv).g, texture(u_scene, v_uv - off).b);
    } else c = texture(u_scene, v_uv).rgb;
    c += texture(u_bloom, v_uv).rgb * u_bloom_amt;
    c *= u_exposure;
    c += u_flash_col * u_flash;
    float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
    c = mix(vec3(l), c, u_sat);
    c = c * u_gain + u_lift * (1.0 - clamp(l * 2.0, 0.0, 1.0)) * 0.012;
    c = aces(c);
    c = pow(c, vec3(1.0 / 2.2));
    c = (c - 0.5) * u_contrast + 0.5;
    float v = 1.0 - u_vignette * smoothstep(0.18, 0.62, r2 * 1.9);
    c *= v;
    float g = hash(gl_FragCoord.xy + fract(u_time) * 91.7) - 0.5;
    c += g * u_grain * (1.0 - 0.5 * l);
    o = vec4(clamp(c, 0.0, 1.0), 1.0);
}
