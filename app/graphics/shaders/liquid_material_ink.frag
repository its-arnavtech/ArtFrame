#version 330 core

in vec2 v_uv;
out vec4 frag_material;

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

float paper_grain(vec2 position) {
    vec2 cell = floor(position);
    return fract(sin(dot(cell, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    float density = density_at(v_uv);
    float left = density_at(v_uv - vec2(u_texel_size.x, 0.0));
    float right = density_at(v_uv + vec2(u_texel_size.x, 0.0));
    float bottom = density_at(v_uv - vec2(0.0, u_texel_size.y));
    float top = density_at(v_uv + vec2(0.0, u_texel_size.y));
    vec2 gradient = 0.5 * vec2(right - left, top - bottom);
    float edge = clamp(length(gradient) * 9.0, 0.0, 1.0);
    vec2 velocity = texture(u_velocity, v_uv).xy;
    float speed = clamp(length(velocity), 0.0, 2.0);
    float curl = texture(u_curl, v_uv).r;
    float pressure = texture(u_pressure, v_uv).r;

    float swirl = clamp(abs(curl) * 10.0, 0.0, 1.0);
    float layers = smoothstep(0.08, 0.65, density);
    vec3 pigment = mix(u_palette_accent, u_palette_primary, layers);
    pigment = mix(pigment, u_palette_secondary, clamp(density * density + swirl * 0.18, 0.0, 1.0));
    pigment = mix(pigment, u_palette_shadow, clamp(density * 0.72 + pressure * 1.5, 0.0, 0.86));

    vec2 flow_direction = velocity / max(length(velocity), 0.0001);
    float fiber = sin(dot(v_uv * 850.0, flow_direction + vec2(0.37, 0.61)) + curl * 24.0);
    float grain = paper_grain(gl_FragCoord.xy * 0.72) - 0.5;
    float texture_detail = (grain * 0.7 + fiber * 0.3) * u_texture_strength;
    pigment *= 1.0 + texture_detail * (0.35 + swirl * 0.65);
    pigment += u_palette_accent * edge * (0.05 + speed * 0.05);

    float interaction = 1.0 + min(u_interaction_velocity, 2.0) * 0.04 + u_interaction_pinch * 0.06;
    float alpha = smoothstep(0.015, 0.32, density) * mix(0.48, 0.94, density);
    alpha *= clamp(u_material_intensity * interaction, 0.0, 1.2);
    frag_material = vec4(clamp(pigment, 0.0, 1.0), clamp(alpha, 0.0, 1.0));
}
