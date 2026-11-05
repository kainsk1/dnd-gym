#version 330 core
#include "common.glsl"
in vec2 v_world;
flat in vec4 v_m;
uniform sampler2DArray u_alb;
uniform sampler2DArray u_nrm;
uniform sampler2D u_cellmap;   // R material, G shape, B underlay material, A variant (all /255)
uniform sampler2D u_sdf;       // solid SDF in cells (negative inside)
uniform sampler2D u_info;      // R shore dist, G chasm depth, B lava proximity
uniform vec2 u_mapsize;
uniform vec4 u_mat[32];        // x struct(cell-local uv) y ragged z kind w tile size (cells)
uniform vec3 u_grade_tint;
layout(location = 0) out vec4 o_alb;
layout(location = 1) out vec4 o_nrm;
layout(location = 2) out vec4 o_emi;

const int NV = 4;
struct Surf { vec3 col; vec2 n; float rough; float emi; float h; };

Surf sample_mat(int mat, vec2 wp, int variant, int rot, bool structured, float tile) {
    Surf s;
    vec2 uv;
    float lw = 1.0;
    if (structured) {
        vec2 f = fract(wp) - 0.5;
        for (int i = 0; i < 4; i++) if (i < rot) f = vec2(-f.y, f.x);
        uv = f + 0.5;
        vec4 a = texture(u_alb, vec3(uv, float(mat * NV + variant)));
        vec4 n = texture(u_nrm, vec3(uv, float(mat * NV + variant)));
        vec2 nn = n.xy * 2.0 - 1.0;
        for (int i = 0; i < 4; i++) if (i < rot) nn = vec2(nn.y, -nn.x);
        s.col = a.rgb; s.h = a.a; s.n = nn; s.rough = n.b; s.emi = n.a;
    } else {
        uv = wp / tile;
        vec2 blk = floor(wp / (tile * 1.5));
        int va = 0;
        int vb = 1;
        float k = smoothstep(0.3, 0.7, vnoise(wp * 0.55 + 3.1));
        vec4 a1 = texture(u_alb, vec3(uv, float(mat * NV + va)));
        vec4 n1 = texture(u_nrm, vec3(uv, float(mat * NV + va)));
        vec2 uv2 = uv * 0.93 + vec2(0.37, 0.21);
        vec4 a2 = texture(u_alb, vec3(uv2, float(mat * NV + vb)));
        vec4 n2 = texture(u_nrm, vec3(uv2, float(mat * NV + vb)));
        s.col = mix(a1.rgb, a2.rgb, k); s.h = mix(a1.a, a2.a, k);
        s.n = mix(n1.xy, n2.xy, k) * 2.0 - 1.0; s.rough = mix(n1.b, n2.b, k); s.emi = mix(n1.a, n2.a, k);
    }
    return s;
}

void main() {
    vec2 wp = v_world;
    int mat = int(v_m.x + 0.5);
    int variant = int(v_m.y + 0.5);
    int rot = int(v_m.z + 0.5);
    int shape = int(v_m.w + 0.5);
    vec4 mi = u_mat[mat];
    ivec2 cell = ivec2(floor(wp));
    vec2 uvm = wp / u_mapsize;
    float sdf = texture(u_sdf, uvm).r;
    vec4 info = texture(u_info, uvm);
    float kind = mi.z;

    // ---- floor-like base under pillars / low walls --------------------------------------------
    int floor_mat = mat;
    if (shape == 2 || shape == 3) floor_mat = int(texelFetch(u_cellmap, cell, 0).b * 255.0 + 0.5);
    vec4 fmi = u_mat[floor_mat];

    // ---- ragged natural boundaries between organic floor materials -----------------------------
    int base_mat = (shape >= 2) ? floor_mat : mat;
    vec4 bmi = (shape >= 2) ? fmi : mi;
    if ((shape == 0 || shape >= 2) && bmi.y > 0.5) {
        vec2 d = (vec2(vnoise(wp * 3.3), vnoise(wp * 3.3 + 17.3)) - 0.5) * 0.75;
        ivec2 nc = ivec2(floor(wp + d));
        if (nc.x >= 0 && nc.y >= 0 && nc.x < int(u_mapsize.x) && nc.y < int(u_mapsize.y)) {
            vec4 c = texelFetch(u_cellmap, nc, 0);
            int om = int(c.r * 255.0 + 0.5);
            if (int(c.g * 255.0 + 0.5) == 0 && u_mat[om].y > 0.5 && u_mat[om].z < 1.5) base_mat = om;
        }
        bmi = u_mat[base_mat];
    }
    kind = bmi.z;
    bool structured = bmi.x > 0.5;
    int cvar = variant;
    Surf s = sample_mat(base_mat, wp, cvar, rot, structured, bmi.w);

    vec3 albedo = srgb2lin(s.col);
    vec2 nxy = s.n;
    float ao = 1.0;
    float wallflag = 0.0;
    vec3 emi = vec3(0.0);
    float rough = s.rough;

    // low-frequency dirt / tint variation so the tiling never reads
    float lf = vnoise(wp * 0.21) * 0.6 + vnoise(wp * 0.63 + 9.0) * 0.4;
    albedo *= 0.82 + 0.36 * lf;

    // ---- liquids -------------------------------------------------------------------------------
    if (kind > 1.5 && kind < 2.5 || kind > 4.5 && kind < 5.5) {          // shallow / deep water
        float deep = kind > 4.5 ? 1.0 : 0.0;
        vec2 u1 = wp * 0.42 + vec2(u_time * 0.035, u_time * 0.02);
        vec2 u2 = wp * 0.77 - vec2(u_time * 0.025, -u_time * 0.045);
        vec4 w1 = texture(u_nrm, vec3(u1, float(base_mat * NV + 1)));
        vec4 w2 = texture(u_nrm, vec3(u2, float(base_mat * NV + 2)));
        float h1 = texture(u_alb, vec3(u1, float(base_mat * NV + 1))).a;
        float h2 = texture(u_alb, vec3(u2, float(base_mat * NV + 2))).a;
        vec2 rip = (w1.xy + w2.xy - 1.0) * (deep > 0.5 ? 1.3 : 1.0);
        nxy = rip + s.n * 0.3;
        float caus = pow(1.0 - abs(h1 - h2), 6.0);
        float shore = clamp(info.r, 0.0, 1.0);
        float foam = smoothstep(0.30, 0.0, info.r) * (0.55 + 0.45 * sin(u_time * 1.3 + wp.x * 5.0 + wp.y * 3.0 + vnoise(wp * 4.0) * 6.0));
        albedo = mix(albedo, albedo * 0.35, deep * 0.5);
        albedo *= mix(1.15, 0.55, shore);
        emi += vec3(0.012, 0.03, 0.035) * caus * (1.0 - deep) * (0.4 + 0.6 * shore);
        albedo = mix(albedo, vec3(0.55, 0.7, 0.72), foam * 0.35 * (1.0 - deep * 0.5));
        rough = 0.06;
    } else if (kind > 2.5 && kind < 3.5) {                                // lava
        vec2 f1 = wp * 0.5 + vec2(u_time * 0.012, u_time * 0.02);
        vec2 f2 = wp * 0.5 * 1.3 - vec2(u_time * 0.018, u_time * 0.007) + 0.3;
        vec2 wob = vec2(sin(wp.y * 6.0 + u_time * 1.5), cos(wp.x * 6.0 + u_time * 1.2)) * 0.004;
        vec4 la = texture(u_alb, vec3(f1 + wob, float(base_mat * NV + variant)));
        vec4 lb = texture(u_alb, vec3(f2 + wob, float(base_mat * NV + (variant + 1) % NV)));
        vec4 na = texture(u_nrm, vec3(f1 + wob, float(base_mat * NV + variant)));
        vec4 nb = texture(u_nrm, vec3(f2 + wob, float(base_mat * NV + (variant + 1) % NV)));
        float em = max(na.a, nb.a * 0.9);
        float mixk = smoothstep(0.35, 0.65, vnoise(wp * 0.8 + u_time * 0.05));
        vec3 c = mix(la.rgb, lb.rgb, mixk);
        em = mix(na.a, nb.a, mixk);
        albedo = srgb2lin(c) * 0.9;
        nxy = mix(na.xy, nb.xy, mixk) * 2.0 - 1.0;
        float pulse = 0.85 + 0.15 * sin(u_time * 1.7 + wp.x * 1.3 + wp.y * 0.9) + 0.12 * sin(u_time * 3.1 + wp.x * 4.1);
        emi += srgb2lin(c) * em * 1.35 * pulse + vec3(1.0, 0.25, 0.03) * em * em * 0.6;
        rough = 0.5;
    } else if (kind > 3.5 && kind < 4.5) {                                // chasm
        float depth = smoothstep(0.0, 1.7, info.g * 2.5);
        albedo *= mix(1.0, 0.015, depth);
        float lip = smoothstep(0.35, 0.0, info.g * 2.5);
        albedo += vec3(0.02) * lip;
        ao = mix(1.0, 0.1, depth);
        nxy *= 0.5;
    }

    // ---- structural shapes ---------------------------------------------------------------------
    vec2 g;
    if (shape == 1) {                                                      // wall block
        float e = 0.06;
        g = vec2(texture(u_sdf, (wp + vec2(e, 0)) / u_mapsize).r - texture(u_sdf, (wp - vec2(e, 0)) / u_mapsize).r,
                 texture(u_sdf, (wp + vec2(0, e)) / u_mapsize).r - texture(u_sdf, (wp - vec2(0, e)) / u_mapsize).r);
        g = normalize(g + vec2(1e-5));
        float depth = -sdf;
        float bev = smoothstep(0.26, 0.0, depth);
        nxy += g * bev * bev * 1.1;
        albedo *= 1.0 + 1.1 * smoothstep(0.16, 0.03, depth) - 0.85 * smoothstep(0.035, 0.0, depth);
        albedo *= mix(1.0, 0.35, smoothstep(0.7, 2.2, depth));   // deep rock fades into darkness
        wallflag = 1.0;
        albedo *= vec3(1.0, 1.04, 1.16) * 1.25;
    } else if (shape == 2) {                                               // round pillar
        vec2 lp = fract(wp) - 0.5;
        float r = length(lp);
        float inside = smoothstep(0.43, 0.405, r);
        // pillar material from the cell itself
        Surf ps = sample_mat(mat, wp, variant, rot, true, 1.0);
        vec3 pc = srgb2lin(ps.col) * (1.25 - 0.45 * smoothstep(0.1, 0.42, r));
        vec2 pn = normalize(lp + 1e-5) * smoothstep(0.05, 0.42, r) * 0.9 + ps.n * 0.3;
        float shadow = smoothstep(0.62, 0.40, r);                          // contact shadow ring
        albedo *= mix(1.0, 0.35, shadow * (1.0 - inside));
        albedo = mix(albedo, pc, inside);
        nxy = mix(nxy, pn, inside);
        float rim = smoothstep(0.42, 0.38, r) * smoothstep(0.30, 0.41, r);
        albedo *= 1.0 - 0.5 * rim * 0.5;
        wallflag = inside;
        rough = mix(rough, ps.rough, inside);
    } else if (shape == 3) {                                               // low wall (crate stack / parapet)
        vec2 lp = abs(fract(wp) - 0.5);
        float m = max(lp.x, lp.y);
        float inside = smoothstep(0.47, 0.44, m);
        Surf ps = sample_mat(mat, wp, variant, rot, true, 1.0);
        vec3 pc = srgb2lin(ps.col) * 1.15;
        vec2 dirv = (fract(wp) - 0.5);
        vec2 pn = ((lp.x > lp.y) ? vec2(sign(dirv.x), 0.0) : vec2(0.0, sign(dirv.y))) * smoothstep(0.3, 0.46, m) * 0.8 + ps.n * 0.4;
        albedo *= mix(1.0, 0.4, smoothstep(0.62, 0.45, m) * (1.0 - inside));
        albedo = mix(albedo, pc, inside);
        nxy = mix(nxy, pn, inside);
    }

    // contact shadows / ambient occlusion next to solid blocks (floors only)
    if (shape == 0 || shape >= 2) {
        float a = smoothstep(0.0, 0.8, sdf);
        ao *= mix(0.22, 1.0, smoothstep(0.0, 1.0, a));
    }
    // thin dark ink where floor meets wall
    if (shape == 0) albedo *= 1.0 - 0.45 * smoothstep(0.07, 0.0, sdf) * step(0.0, sdf);

    albedo *= u_grade_tint;
    vec3 nrm3 = normalize(vec3(nxy, 1.0));
    o_alb = vec4(albedo * (0.55 + 0.45 * ao), rough);
    o_nrm = vec4(nrm3.xy * 0.5 + 0.5, wallflag, ao);
    o_emi = vec4(emi, 1.0);
}
