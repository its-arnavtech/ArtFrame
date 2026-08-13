#version 330 core

in vec2 v_uv;
out float frag_divergence;

uniform sampler2D u_velocity;
uniform vec2 u_texel_size;

vec2 safe_uv(vec2 uv) {
    return clamp(uv, 0.5 * u_texel_size, vec2(1.0) - 0.5 * u_texel_size);
}

void main() {
    float left = texture(u_velocity, safe_uv(v_uv - vec2(u_texel_size.x, 0.0))).x;
    float right = texture(u_velocity, safe_uv(v_uv + vec2(u_texel_size.x, 0.0))).x;
    float bottom = texture(u_velocity, safe_uv(v_uv - vec2(0.0, u_texel_size.y))).y;
    float top = texture(u_velocity, safe_uv(v_uv + vec2(0.0, u_texel_size.y))).y;
    frag_divergence = 0.5 * (
        (right - left) / u_texel_size.x
        + (top - bottom) / u_texel_size.y
    );
}
