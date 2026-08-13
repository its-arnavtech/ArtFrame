#version 330 core

in vec2 v_uv;
out float frag_pressure;

uniform sampler2D u_pressure;
uniform sampler2D u_divergence;
uniform vec2 u_texel_size;

vec2 safe_uv(vec2 uv) {
    return clamp(uv, 0.5 * u_texel_size, vec2(1.0) - 0.5 * u_texel_size);
}

void main() {
    float left = texture(u_pressure, safe_uv(v_uv - vec2(u_texel_size.x, 0.0))).r;
    float right = texture(u_pressure, safe_uv(v_uv + vec2(u_texel_size.x, 0.0))).r;
    float bottom = texture(u_pressure, safe_uv(v_uv - vec2(0.0, u_texel_size.y))).r;
    float top = texture(u_pressure, safe_uv(v_uv + vec2(0.0, u_texel_size.y))).r;
    float divergence = texture(u_divergence, v_uv).r;
    float dx2 = u_texel_size.x * u_texel_size.x;
    float dy2 = u_texel_size.y * u_texel_size.y;
    float numerator = (left + right) * dy2 + (bottom + top) * dx2;
    numerator -= divergence * dx2 * dy2;
    frag_pressure = numerator / max(2.0 * (dx2 + dy2), 0.00000001);
}
