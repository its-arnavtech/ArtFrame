#version 330 core

in vec2 v_uv;
out float frag_curl;

uniform sampler2D u_velocity;
uniform vec2 u_texel_size;

vec2 safe_uv(vec2 uv) {
    return clamp(uv, 0.5 * u_texel_size, vec2(1.0) - 0.5 * u_texel_size);
}

void main() {
    float left_y = texture(u_velocity, safe_uv(v_uv - vec2(u_texel_size.x, 0.0))).y;
    float right_y = texture(u_velocity, safe_uv(v_uv + vec2(u_texel_size.x, 0.0))).y;
    float bottom_x = texture(u_velocity, safe_uv(v_uv - vec2(0.0, u_texel_size.y))).x;
    float top_x = texture(u_velocity, safe_uv(v_uv + vec2(0.0, u_texel_size.y))).x;
    frag_curl = 0.5 * ((right_y - left_y) - (top_x - bottom_x));
}
