#version 330 core

in vec2 v_uv;
out vec2 frag_velocity;

uniform sampler2D u_velocity;
uniform float u_timestep;
uniform float u_injection_strength;
uniform float u_velocity_scale;
uniform float u_source_radius;

uniform int u_left_active;
uniform vec2 u_left_position;
uniform vec2 u_left_velocity;
uniform float u_left_pinch;
uniform float u_left_openness;
uniform float u_left_influence;
uniform int u_right_active;
uniform vec2 u_right_position;
uniform vec2 u_right_velocity;
uniform float u_right_pinch;
uniform float u_right_openness;
uniform float u_right_influence;

vec2 injected_force(
    int source_active,
    vec2 source_position,
    vec2 source_velocity,
    float pinch,
    float openness,
    float influence
) {
    if (source_active == 0 || influence <= 0.0001) {
        return vec2(0.0);
    }
    float speed = length(source_velocity);
    if (speed < 0.0001) {
        return vec2(0.0);
    }
    float radius = u_source_radius * mix(0.75, 1.55, openness);
    vec2 delta = v_uv - source_position;
    float falloff = exp(-dot(delta, delta) / max(2.0 * radius * radius, 0.000001));
    float gesture_strength = mix(0.35, 1.0, pinch);
    float magnitude = min(speed * u_velocity_scale, 3.0);
    return normalize(source_velocity) * magnitude * gesture_strength * falloff * influence;
}

void main() {
    vec2 velocity = texture(u_velocity, v_uv).xy;
    vec2 force = injected_force(
        u_left_active, u_left_position, u_left_velocity, u_left_pinch, u_left_openness,
        u_left_influence
    );
    force += injected_force(
        u_right_active, u_right_position, u_right_velocity, u_right_pinch, u_right_openness,
        u_right_influence
    );
    frag_velocity = velocity + force * u_injection_strength * u_timestep;
}
