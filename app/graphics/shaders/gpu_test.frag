#version 330 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_camera;
uniform sampler2D u_feedback;
uniform vec2 u_resolution;
uniform float u_time;

uniform int u_left_active;
uniform vec2 u_left_position;
uniform vec2 u_left_velocity;
uniform float u_left_pinch;
uniform int u_right_active;
uniform vec2 u_right_position;
uniform vec2 u_right_velocity;
uniform float u_right_pinch;

vec2 hand_displacement(
    vec2 uv,
    int hand_active,
    vec2 interaction_position,
    vec2 interaction_velocity,
    float strength
) {
    if (hand_active == 0) {
        return vec2(0.0);
    }
    vec2 center = vec2(interaction_position.x, 1.0 - interaction_position.y);
    vec2 velocity = vec2(interaction_velocity.x, -interaction_velocity.y);
    vec2 delta = uv - center;
    float aspect = u_resolution.x / max(u_resolution.y, 1.0);
    float distance_squared = dot(delta * vec2(aspect, 1.0), delta * vec2(aspect, 1.0));
    float falloff = exp(-distance_squared * 38.0);
    vec2 swirl = vec2(-delta.y, delta.x) * sin(u_time * 1.7 + distance_squared * 30.0);
    return falloff * (velocity * 0.018 + swirl * (0.012 + strength * 0.025));
}

void main() {
    vec2 distortion = hand_displacement(
        v_uv, u_left_active, u_left_position, u_left_velocity, 0.35
    );
    distortion += hand_displacement(
        v_uv, u_right_active, u_right_position, u_right_velocity, u_right_pinch
    );

    vec2 camera_uv = vec2(v_uv.x, 1.0 - v_uv.y) + vec2(distortion.x, -distortion.y);
    vec3 camera_bgr = texture(u_camera, clamp(camera_uv, 0.0, 1.0)).rgb;
    vec3 camera_rgb = camera_bgr.bgr;
    vec3 feedback = texture(u_feedback, v_uv - distortion * 0.35).rgb;
    float pulse = 0.5 + 0.5 * sin(u_time * 0.7 + v_uv.x * 4.0 + v_uv.y * 3.0);
    vec3 tint = vec3(0.04, 0.08, 0.13) * pulse;
    vec3 color = camera_rgb + tint * (0.12 + length(distortion) * 8.0);
    color = mix(color, feedback, 0.055);
    frag_color = vec4(color, 1.0);
}
