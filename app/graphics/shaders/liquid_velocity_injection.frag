#version 330 core

in vec2 v_uv;
out vec2 frag_velocity;

uniform sampler2D u_velocity;
uniform float u_timestep;
uniform float u_injection_strength;
uniform float u_velocity_scale;
uniform float u_source_radius;

uniform int u_left_active;
uniform int u_left_tip_count;
uniform vec2 u_left_tip0_position;
uniform vec2 u_left_tip0_velocity;
uniform vec2 u_left_tip1_position;
uniform vec2 u_left_tip1_velocity;
uniform vec2 u_left_tip2_position;
uniform vec2 u_left_tip2_velocity;
uniform vec2 u_left_tip3_position;
uniform vec2 u_left_tip3_velocity;
uniform vec2 u_left_tip4_position;
uniform vec2 u_left_tip4_velocity;
uniform float u_left_pinch;
uniform float u_left_openness;
uniform float u_left_influence;

uniform int u_right_active;
uniform int u_right_tip_count;
uniform vec2 u_right_tip0_position;
uniform vec2 u_right_tip0_velocity;
uniform vec2 u_right_tip1_position;
uniform vec2 u_right_tip1_velocity;
uniform vec2 u_right_tip2_position;
uniform vec2 u_right_tip2_velocity;
uniform vec2 u_right_tip3_position;
uniform vec2 u_right_tip3_velocity;
uniform vec2 u_right_tip4_position;
uniform vec2 u_right_tip4_velocity;
uniform float u_right_pinch;
uniform float u_right_openness;
uniform float u_right_influence;

vec2 fingertip_force(
    vec2 source_position,
    vec2 source_velocity,
    float pinch,
    float openness,
    float influence
) {
    float speed = length(source_velocity);
    if (speed < 0.0001 || influence <= 0.0001) {
        return vec2(0.0);
    }
    float radius = u_source_radius * mix(0.42, 0.72, openness);
    vec2 delta = v_uv - source_position;
    float falloff = exp(-dot(delta, delta) / max(2.0 * radius * radius, 0.000001));
    float gesture_strength = mix(0.55, 1.0, pinch);
    float magnitude = min(speed * u_velocity_scale, 3.0);
    return normalize(source_velocity) * magnitude * gesture_strength * falloff * influence * 0.52;
}

vec2 left_force() {
    if (u_left_active == 0) return vec2(0.0);
    vec2 force = vec2(0.0);
    if (u_left_tip_count > 0) force += fingertip_force(u_left_tip0_position, u_left_tip0_velocity, u_left_pinch, u_left_openness, u_left_influence);
    if (u_left_tip_count > 1) force += fingertip_force(u_left_tip1_position, u_left_tip1_velocity, u_left_pinch, u_left_openness, u_left_influence);
    if (u_left_tip_count > 2) force += fingertip_force(u_left_tip2_position, u_left_tip2_velocity, u_left_pinch, u_left_openness, u_left_influence);
    if (u_left_tip_count > 3) force += fingertip_force(u_left_tip3_position, u_left_tip3_velocity, u_left_pinch, u_left_openness, u_left_influence);
    if (u_left_tip_count > 4) force += fingertip_force(u_left_tip4_position, u_left_tip4_velocity, u_left_pinch, u_left_openness, u_left_influence);
    return force;
}

vec2 right_force() {
    if (u_right_active == 0) return vec2(0.0);
    vec2 force = vec2(0.0);
    if (u_right_tip_count > 0) force += fingertip_force(u_right_tip0_position, u_right_tip0_velocity, u_right_pinch, u_right_openness, u_right_influence);
    if (u_right_tip_count > 1) force += fingertip_force(u_right_tip1_position, u_right_tip1_velocity, u_right_pinch, u_right_openness, u_right_influence);
    if (u_right_tip_count > 2) force += fingertip_force(u_right_tip2_position, u_right_tip2_velocity, u_right_pinch, u_right_openness, u_right_influence);
    if (u_right_tip_count > 3) force += fingertip_force(u_right_tip3_position, u_right_tip3_velocity, u_right_pinch, u_right_openness, u_right_influence);
    if (u_right_tip_count > 4) force += fingertip_force(u_right_tip4_position, u_right_tip4_velocity, u_right_pinch, u_right_openness, u_right_influence);
    return force;
}

void main() {
    vec2 velocity = texture(u_velocity, v_uv).xy;
    vec2 force = left_force() + right_force();
    frag_velocity = velocity + force * u_injection_strength * u_timestep;
}
