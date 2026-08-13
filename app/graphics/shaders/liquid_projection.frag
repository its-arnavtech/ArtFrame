#version 330 core

in vec2 v_uv;
out vec2 frag_velocity;

uniform sampler2D u_velocity;
uniform sampler2D u_pressure;
uniform vec2 u_texel_size;

vec2 safe_uv(vec2 uv) {
    return clamp(uv, 0.5 * u_texel_size, vec2(1.0) - 0.5 * u_texel_size);
}

void main() {
    float left = texture(u_pressure, safe_uv(v_uv - vec2(u_texel_size.x, 0.0))).r;
    float right = texture(u_pressure, safe_uv(v_uv + vec2(u_texel_size.x, 0.0))).r;
    float bottom = texture(u_pressure, safe_uv(v_uv - vec2(0.0, u_texel_size.y))).r;
    float top = texture(u_pressure, safe_uv(v_uv + vec2(0.0, u_texel_size.y))).r;
    vec2 gradient = 0.5 * vec2(
        (right - left) / u_texel_size.x,
        (top - bottom) / u_texel_size.y
    );
    frag_velocity = texture(u_velocity, v_uv).xy - gradient;
}
