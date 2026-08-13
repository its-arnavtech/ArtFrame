#version 330 core

in vec2 v_uv;
out vec2 frag_velocity;

uniform sampler2D u_velocity;
uniform sampler2D u_curl;
uniform vec2 u_texel_size;
uniform float u_timestep;
uniform float u_strength;

vec2 safe_uv(vec2 uv) {
    return clamp(uv, 0.5 * u_texel_size, vec2(1.0) - 0.5 * u_texel_size);
}

void main() {
    float left = abs(texture(u_curl, safe_uv(v_uv - vec2(u_texel_size.x, 0.0))).r);
    float right = abs(texture(u_curl, safe_uv(v_uv + vec2(u_texel_size.x, 0.0))).r);
    float bottom = abs(texture(u_curl, safe_uv(v_uv - vec2(0.0, u_texel_size.y))).r);
    float top = abs(texture(u_curl, safe_uv(v_uv + vec2(0.0, u_texel_size.y))).r);
    float center_curl = texture(u_curl, v_uv).r;
    vec2 gradient = 0.5 * vec2(right - left, top - bottom);
    vec2 normal = gradient / max(length(gradient), 0.00001);
    vec2 force = vec2(normal.y, -normal.x) * center_curl * u_strength;
    frag_velocity = texture(u_velocity, v_uv).xy + force * u_timestep * 60.0;
}
