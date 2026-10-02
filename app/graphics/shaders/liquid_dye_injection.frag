#version 330 core

in vec2 v_uv;
out vec4 frag_dye;

uniform sampler2D u_dye;
uniform float u_timestep;
uniform float u_source_radius;
uniform float u_injection_strength;
uniform float u_velocity_coupling;

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
uniform vec3 u_left_color;

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
uniform vec3 u_right_color;

vec4 inject_fingertip(
    vec4 dye,
    vec2 source_position,
    vec2 source_velocity,
    float pinch,
    float openness,
    float influence,
    vec3 source_color
) {
    float radius = u_source_radius * mix(0.44, 0.74, openness);
    vec2 delta = v_uv - source_position;
    float falloff = exp(-dot(delta, delta) / max(2.0 * radius * radius, 0.000001));
    float speed_coupling = 1.0 + min(length(source_velocity), 3.0) * u_velocity_coupling;
    float gesture_strength = mix(0.45, 1.0, pinch);
    float amount = clamp(
        falloff * gesture_strength * speed_coupling * influence * u_injection_strength * u_timestep,
        0.0,
        1.0
    );
    dye.rgb = mix(dye.rgb, source_color, amount);
    dye.a = 1.0 - (1.0 - dye.a) * (1.0 - amount);
    return dye;
}

vec4 inject_left(vec4 dye) {
    if (u_left_active == 0 || u_left_influence <= 0.0001) return dye;
    if (u_left_tip_count > 0) dye = inject_fingertip(dye, u_left_tip0_position, u_left_tip0_velocity, u_left_pinch, u_left_openness, u_left_influence, u_left_color);
    if (u_left_tip_count > 1) dye = inject_fingertip(dye, u_left_tip1_position, u_left_tip1_velocity, u_left_pinch, u_left_openness, u_left_influence, u_left_color);
    if (u_left_tip_count > 2) dye = inject_fingertip(dye, u_left_tip2_position, u_left_tip2_velocity, u_left_pinch, u_left_openness, u_left_influence, u_left_color);
    if (u_left_tip_count > 3) dye = inject_fingertip(dye, u_left_tip3_position, u_left_tip3_velocity, u_left_pinch, u_left_openness, u_left_influence, u_left_color);
    if (u_left_tip_count > 4) dye = inject_fingertip(dye, u_left_tip4_position, u_left_tip4_velocity, u_left_pinch, u_left_openness, u_left_influence, u_left_color);
    return dye;
}

vec4 inject_right(vec4 dye) {
    if (u_right_active == 0 || u_right_influence <= 0.0001) return dye;
    if (u_right_tip_count > 0) dye = inject_fingertip(dye, u_right_tip0_position, u_right_tip0_velocity, u_right_pinch, u_right_openness, u_right_influence, u_right_color);
    if (u_right_tip_count > 1) dye = inject_fingertip(dye, u_right_tip1_position, u_right_tip1_velocity, u_right_pinch, u_right_openness, u_right_influence, u_right_color);
    if (u_right_tip_count > 2) dye = inject_fingertip(dye, u_right_tip2_position, u_right_tip2_velocity, u_right_pinch, u_right_openness, u_right_influence, u_right_color);
    if (u_right_tip_count > 3) dye = inject_fingertip(dye, u_right_tip3_position, u_right_tip3_velocity, u_right_pinch, u_right_openness, u_right_influence, u_right_color);
    if (u_right_tip_count > 4) dye = inject_fingertip(dye, u_right_tip4_position, u_right_tip4_velocity, u_right_pinch, u_right_openness, u_right_influence, u_right_color);
    return dye;
}

void main() {
    vec4 dye = texture(u_dye, v_uv);
    dye = inject_left(dye);
    dye = inject_right(dye);
    frag_dye = dye;
}
