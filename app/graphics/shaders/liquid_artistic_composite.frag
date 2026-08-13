#version 330 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_base_camera;
uniform sampler2D u_foreground_camera;
uniform sampler2D u_hand_mask;
uniform sampler2D u_material;
uniform sampler2D u_print;
uniform sampler2D u_dye;
uniform sampler2D u_velocity;
uniform sampler2D u_pressure;
uniform sampler2D u_divergence;
uniform sampler2D u_curl;
uniform int u_debug_view;
uniform int u_visualization_enabled;
uniform int u_print_enabled;
uniform int u_left_active;
uniform vec2 u_left_position;
uniform int u_right_active;
uniform vec2 u_right_position;

vec3 signed_scalar(float value, float scale) {
    float amount = clamp(abs(value) * scale, 0.0, 1.0);
    return value >= 0.0 ? vec3(amount, 0.1 * amount, 0.01) : vec3(0.01, 0.2 * amount, amount);
}

float source_marker(vec2 position, int source_active) {
    return source_active == 0 ? 0.0 : smoothstep(0.026, 0.017, length(v_uv - position));
}

void main() {
    vec2 camera_uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec3 base = texture(u_base_camera, camera_uv).bgr;
    vec3 foreground = texture(u_foreground_camera, camera_uv).bgr;
    float mask = texture(u_hand_mask, camera_uv).r;
    vec4 material = texture(u_material, v_uv);
    vec4 dye = texture(u_dye, v_uv);
    vec2 velocity = texture(u_velocity, v_uv).xy;
    float pressure = texture(u_pressure, v_uv).r;
    float divergence = texture(u_divergence, v_uv).r;
    float curl = texture(u_curl, v_uv).r;

    if (u_debug_view == 1) { frag_color = vec4(dye.rgb, 1.0); return; }
    if (u_debug_view == 2) { frag_color = vec4(vec3(clamp(length(velocity) / 1.5, 0.0, 1.0)), 1.0); return; }
    if (u_debug_view == 3) {
        vec2 direction = velocity / max(length(velocity), 0.00001);
        frag_color = vec4(0.5 + 0.5 * direction, clamp(length(velocity), 0.0, 1.0), 1.0);
        return;
    }
    if (u_debug_view == 4) { frag_color = vec4(signed_scalar(pressure, 8.0), 1.0); return; }
    if (u_debug_view == 5) { frag_color = vec4(signed_scalar(divergence, 0.025), 1.0); return; }
    if (u_debug_view == 6) { frag_color = vec4(signed_scalar(curl, 12.0), 1.0); return; }
    if (u_debug_view == 7) {
        float left = source_marker(u_left_position, u_left_active);
        float right = source_marker(u_right_position, u_right_active);
        vec3 sources = mix(base, vec3(0.02, 0.55, 0.78), left);
        frag_color = vec4(mix(sources, vec3(0.90, 0.05, 0.38), right), 1.0);
        return;
    }
    if (u_debug_view == 11) { frag_color = vec4(vec3(mask), 1.0); return; }
    if (u_debug_view == 12) { frag_color = vec4(material.rgb, 1.0); return; }
    if (u_debug_view >= 13 && u_debug_view <= 17) {
        frag_color = vec4(texture(u_print, v_uv).rgb, 1.0);
        return;
    }

    float material_alpha = u_visualization_enabled == 0 ? 0.0 : material.a;
    vec4 print_layer = texture(u_print, v_uv);
    vec3 artistic = u_print_enabled == 1 ? print_layer.rgb : material.rgb;
    float artistic_alpha = u_print_enabled == 1 ? print_layer.a : material_alpha;
    artistic_alpha = u_visualization_enabled == 0 ? 0.0 : artistic_alpha;
    vec3 under_hands = mix(base, artistic, artistic_alpha);
    frag_color = vec4(mix(under_hands, foreground, mask), 1.0);
}
