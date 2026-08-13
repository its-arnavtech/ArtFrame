#version 330 core

in vec2 v_uv;
out vec4 frag_material;

uniform sampler2D u_base_camera;
uniform sampler2D u_dye;
uniform sampler2D u_velocity;
uniform sampler2D u_curl;
uniform sampler2D u_pressure;
uniform vec2 u_texel_size;
uniform vec3 u_palette_primary;
uniform vec3 u_palette_secondary;
uniform vec3 u_palette_accent;
uniform vec3 u_palette_shadow;
uniform float u_material_intensity;
uniform float u_texture_strength;
uniform float u_interaction_velocity;
uniform float u_interaction_pinch;
uniform float u_hand_distance;
uniform float u_time;

float density_at(vec2 uv) {
    vec4 dye = texture(u_dye, clamp(uv, 0.0, 1.0));
    return clamp(max(dye.a, max(dye.r, max(dye.g, dye.b))), 0.0, 1.0);
}

vec3 camera_sample(vec2 uv) {
    return texture(u_base_camera, clamp(uv, 0.002, 0.998)).bgr;
}

void main() {
    float density = density_at(v_uv);
    vec2 gradient = vec2(
        density_at(v_uv + vec2(u_texel_size.x, 0.0)) - density_at(v_uv - vec2(u_texel_size.x, 0.0)),
        density_at(v_uv + vec2(0.0, u_texel_size.y)) - density_at(v_uv - vec2(0.0, u_texel_size.y))
    ) * 0.5;
    vec2 velocity = texture(u_velocity, v_uv).xy;
    float speed = min(length(velocity), 3.0);
    float curl = texture(u_curl, v_uv).r;
    float pressure = texture(u_pressure, v_uv).r;
    float edge = smoothstep(0.008, 0.13, length(gradient));
    float body = smoothstep(0.025, 0.34, density);

    vec2 direction = normalize(gradient + velocity * 0.045 + vec2(0.0001));
    vec2 camera_uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec2 screen_direction = vec2(direction.x, -direction.y);
    float motion_energy = min(u_interaction_velocity, 3.0) / 3.0;
    float distortion = body * (0.006 + edge * 0.022 + speed * 0.0035);
    vec2 flow_offset = vec2(velocity.x, -velocity.y) * 0.005;
    vec2 refracted_uv = camera_uv + screen_direction * distortion + flow_offset;

    float dispersion = body * (0.0015 + edge * 0.007 + speed * 0.0015);
    dispersion *= 1.0 + u_interaction_pinch * 0.35 + motion_energy * 0.25;
    vec3 refracted;
    refracted.r = camera_sample(refracted_uv + screen_direction * dispersion).r;
    refracted.g = camera_sample(refracted_uv).g;
    refracted.b = camera_sample(refracted_uv - screen_direction * dispersion).b;

    float film_phase = density * 15.0 + curl * 7.0 + pressure * 4.0;
    film_phase += dot(v_uv, vec2(9.0, -6.0)) + u_time * (0.18 + motion_energy * 0.25);
    vec3 spectrum = 0.54 + 0.46 * cos(film_phase + vec3(0.0, 2.094, 4.188));
    vec3 palette_film = mix(u_palette_primary, u_palette_secondary, spectrum.g);
    palette_film = mix(palette_film, u_palette_accent, spectrum.r * 0.48);
    vec3 iridescence = mix(spectrum, palette_film, 0.42);

    float film_amount = body * (0.055 + edge * 0.36 + abs(curl) * 0.10);
    film_amount += edge * motion_energy * 0.12;
    vec3 color = mix(refracted, iridescence, clamp(film_amount, 0.0, 0.62));
    color += vec3(1.0) * edge * (0.10 + spectrum.b * 0.16);
    color *= 1.0 - body * (1.0 - edge) * 0.035;

    float alpha = body * (0.30 + edge * 0.30 + motion_energy * 0.06);
    alpha *= clamp(u_material_intensity, 0.0, 1.35);
    frag_material = vec4(clamp(color, 0.0, 1.0), clamp(alpha, 0.0, 0.72));
}
