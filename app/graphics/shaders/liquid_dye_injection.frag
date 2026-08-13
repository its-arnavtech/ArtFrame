#version 330 core

in vec2 v_uv;
out vec4 frag_dye;

uniform sampler2D u_dye;
uniform float u_timestep;
uniform float u_source_radius;
uniform float u_injection_strength;
uniform float u_velocity_coupling;

uniform int u_left_active;
uniform vec2 u_left_position;
uniform vec2 u_left_velocity;
uniform float u_left_pinch;
uniform float u_left_openness;
uniform vec3 u_left_color;
uniform int u_right_active;
uniform vec2 u_right_position;
uniform vec2 u_right_velocity;
uniform float u_right_pinch;
uniform float u_right_openness;
uniform vec3 u_right_color;

vec4 inject_source(
    vec4 dye,
    int source_active,
    vec2 source_position,
    vec2 source_velocity,
    float pinch,
    float openness,
    vec3 source_color
) {
    if (source_active == 0) {
        return dye;
    }
    float radius = u_source_radius * mix(0.8, 1.65, openness);
    vec2 delta = v_uv - source_position;
    float falloff = exp(-dot(delta, delta) / max(2.0 * radius * radius, 0.000001));
    float speed_coupling = 1.0 + min(length(source_velocity), 3.0) * u_velocity_coupling;
    float gesture_strength = mix(0.3, 1.0, pinch);
    float amount = clamp(
        falloff * gesture_strength * speed_coupling * u_injection_strength * u_timestep,
        0.0,
        1.0
    );
    dye.rgb = mix(dye.rgb, source_color, amount);
    dye.a = 1.0 - (1.0 - dye.a) * (1.0 - amount);
    return dye;
}

void main() {
    vec4 dye = texture(u_dye, v_uv);
    dye = inject_source(
        dye,
        u_left_active,
        u_left_position,
        u_left_velocity,
        u_left_pinch,
        u_left_openness,
        u_left_color
    );
    dye = inject_source(
        dye,
        u_right_active,
        u_right_position,
        u_right_velocity,
        u_right_pinch,
        u_right_openness,
        u_right_color
    );
    frag_dye = dye;
}
