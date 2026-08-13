#version 330 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_camera;
uniform sampler2D u_dye;
uniform sampler2D u_velocity;
uniform sampler2D u_pressure;
uniform sampler2D u_divergence;
uniform sampler2D u_curl;
uniform int u_visualization_enabled;
uniform int u_debug_view;
uniform int u_left_active;
uniform vec2 u_left_position;
uniform vec3 u_left_color;
uniform int u_right_active;
uniform vec2 u_right_position;
uniform vec3 u_right_color;

vec3 signed_scalar_color(float value, float scale) {
    float amount = clamp(abs(value) * scale, 0.0, 1.0);
    return value >= 0.0 ? vec3(amount, 0.12 * amount, 0.02) : vec3(0.02, 0.24 * amount, amount);
}

float source_marker(vec2 position, int source_active) {
    if (source_active == 0) {
        return 0.0;
    }
    float distance_to_source = length(v_uv - position);
    return smoothstep(0.025, 0.018, distance_to_source);
}

void main() {
    vec2 camera_uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec3 camera_rgb = texture(u_camera, camera_uv).bgr;
    vec4 dye = texture(u_dye, v_uv);
    vec2 velocity = texture(u_velocity, v_uv).xy;
    float pressure = texture(u_pressure, v_uv).r;
    float divergence = texture(u_divergence, v_uv).r;
    float curl = texture(u_curl, v_uv).r;

    if (u_debug_view == 1) {
        frag_color = vec4(dye.rgb, 1.0);
        return;
    }
    if (u_debug_view == 2) {
        float magnitude = clamp(length(velocity) / 1.5, 0.0, 1.0);
        frag_color = vec4(vec3(magnitude), 1.0);
        return;
    }
    if (u_debug_view == 3) {
        float magnitude = clamp(length(velocity), 0.0, 1.0);
        vec2 direction = velocity / max(length(velocity), 0.00001);
        frag_color = vec4(0.5 + 0.5 * direction, magnitude, 1.0);
        return;
    }
    if (u_debug_view == 4) {
        frag_color = vec4(signed_scalar_color(pressure, 8.0), 1.0);
        return;
    }
    if (u_debug_view == 5) {
        frag_color = vec4(signed_scalar_color(divergence, 0.025), 1.0);
        return;
    }
    if (u_debug_view == 6) {
        frag_color = vec4(signed_scalar_color(curl, 12.0), 1.0);
        return;
    }
    if (u_debug_view == 7) {
        float left_marker = source_marker(u_left_position, u_left_active);
        float right_marker = source_marker(u_right_position, u_right_active);
        vec3 source_color = camera_rgb;
        source_color = mix(source_color, u_left_color, left_marker);
        source_color = mix(source_color, u_right_color, right_marker);
        frag_color = vec4(source_color, 1.0);
        return;
    }
    if (u_visualization_enabled == 0) {
        frag_color = vec4(camera_rgb, 1.0);
        return;
    }
    float density = clamp(max(dye.a, max(dye.r, max(dye.g, dye.b))), 0.0, 1.0);
    vec3 liquid = dye.rgb * (0.72 + density * 0.45);
    frag_color = vec4(mix(camera_rgb, liquid, density * 0.78), 1.0);
}
