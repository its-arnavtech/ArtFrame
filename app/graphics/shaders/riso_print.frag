#version 330 core

in vec2 v_uv;
out vec4 frag_print;

uniform sampler2D u_channels;
uniform sampler2D u_material;
uniform sampler2D u_velocity;
uniform sampler2D u_history;
uniform vec2 u_display_size;
uniform vec3 u_primary_ink;
uniform vec3 u_secondary_ink;
uniform vec3 u_paper_color;
uniform vec3 u_accent_ink;
uniform float u_dot_period;
uniform vec2 u_primary_basis;
uniform vec2 u_secondary_basis;
uniform vec2 u_registration_uv;
uniform float u_dot_strength;
uniform float u_threshold;
uniform float u_paper_strength;
uniform float u_grain_strength;
uniform float u_paper_detail;
uniform float u_registration_complexity;
uniform float u_history_mix;
uniform int u_posterization_steps;
uniform int u_debug_mode;

float hash21(vec2 position) {
    vec2 cell = floor(position);
    return fract(sin(dot(cell, vec2(12.9898, 78.233))) * 43758.5453);
}

float prepared_density(float density) {
    float value = clamp((density - u_threshold) / max(1.0 - u_threshold, 0.0001), 0.0, 1.0);
    if (u_posterization_steps >= 2) {
        float steps = float(u_posterization_steps - 1);
        value = floor(value * steps + 0.5) / steps;
    }
    return value;
}

float screen_dot(float density, vec2 basis, vec2 pixel_position) {
    vec2 perpendicular = vec2(-basis.y, basis.x);
    vec2 rotated = vec2(dot(pixel_position, basis), dot(pixel_position, perpendicular));
    vec2 local = fract(rotated / u_dot_period) - 0.5;
    float radius = sqrt(clamp(density, 0.0, 1.0)) * 0.68;
    float antialias = 1.2 / max(u_dot_period, 2.0);
    return smoothstep(radius + antialias, radius - antialias, length(local));
}

vec3 paper_color(vec2 pixel_position, out float paper_value) {
    float coarse = hash21(pixel_position * (0.12 * u_paper_detail));
    float fiber = sin(pixel_position.y * 0.31 + sin(pixel_position.x * 0.047) * 1.8) * 0.5 + 0.5;
    paper_value = (coarse - 0.5) * 0.68 + (fiber - 0.5) * 0.32;
    return clamp(u_paper_color * (1.0 + paper_value * u_paper_strength), 0.0, 1.0);
}

void main() {
    vec2 velocity = texture(u_velocity, v_uv).xy;
    vec2 movement_offset = velocity * u_registration_uv * u_registration_complexity * 0.38;
    vec2 primary_uv = clamp(v_uv + u_registration_uv + movement_offset, 0.0, 1.0);
    vec2 secondary_uv = clamp(v_uv - u_registration_uv - movement_offset, 0.0, 1.0);
    vec4 center_channels = texture(u_channels, v_uv);
    float primary_density = prepared_density(texture(u_channels, primary_uv).r);
    float secondary_density = prepared_density(texture(u_channels, secondary_uv).g);
    float breakup = center_channels.b;

    vec2 pixel = v_uv * u_display_size;
    float primary_dot = screen_dot(primary_density, u_primary_basis, pixel);
    float secondary_dot = screen_dot(secondary_density, u_secondary_basis, pixel);
    float primary_coverage = mix(primary_density, primary_dot, u_dot_strength);
    float secondary_coverage = mix(secondary_density, secondary_dot, u_dot_strength);

    float grain = hash21(pixel * (0.48 + u_paper_detail * 0.22)) - 0.5;
    primary_coverage = clamp(primary_coverage + grain * u_grain_strength - breakup * grain * 0.6, 0.0, 1.0);
    secondary_coverage = clamp(secondary_coverage - grain * u_grain_strength * 0.72 + breakup * grain * 0.4, 0.0, 1.0);

    float paper_value;
    vec3 paper = paper_color(pixel, paper_value);
    vec3 printed = mix(paper, u_primary_ink, primary_coverage);
    vec3 overlap_ink = mix(u_secondary_ink, u_accent_ink, primary_coverage * 0.28);
    printed = mix(printed, overlap_ink * mix(vec3(1.0), u_primary_ink, primary_coverage * 0.16), secondary_coverage);
    float ink_density = max(primary_density, secondary_density);
    float density_alpha = clamp(u_paper_strength * 0.35 + ink_density * (1.0 - u_paper_strength * 0.2), 0.0, 1.0);

    if (u_debug_mode == 1) {
        frag_print = vec4(vec3(center_channels.a), 1.0);
        return;
    }
    if (u_debug_mode == 2) {
        frag_print = vec4(vec3(max(primary_dot, secondary_dot)), 1.0);
        return;
    }
    if (u_debug_mode == 3) {
        frag_print = vec4(primary_coverage, secondary_coverage, abs(primary_coverage - secondary_coverage), 1.0);
        return;
    }
    if (u_debug_mode == 4) {
        frag_print = vec4(paper, 1.0);
        return;
    }

    vec4 history = texture(u_history, v_uv);
    float speed = clamp(length(velocity), 0.0, 1.0);
    float stable_mix = u_history_mix * (1.0 - speed * 0.75);
    vec3 stable_print = mix(printed, history.rgb, stable_mix);
    frag_print = vec4(clamp(stable_print, 0.0, 1.0), density_alpha);
}
