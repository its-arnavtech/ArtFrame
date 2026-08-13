#version 330 core

in vec2 v_uv;
out vec4 frag_channels;

uniform sampler2D u_material;
uniform sampler2D u_dye;
uniform sampler2D u_velocity;
uniform sampler2D u_curl;
uniform vec2 u_simulation_size;
uniform float u_density_response;
uniform float u_edge_breakup;
uniform float u_interaction_velocity;
uniform float u_interaction_pinch;
uniform float u_interaction_openness;
uniform float u_hand_distance;

float dye_density(vec2 uv) {
    vec4 dye = texture(u_dye, clamp(uv, 0.0, 1.0));
    return clamp(max(dye.a, max(dye.r, max(dye.g, dye.b))), 0.0, 1.0);
}

void main() {
    vec2 texel = 1.0 / u_simulation_size;
    vec4 material = texture(u_material, v_uv);
    vec4 dye = texture(u_dye, v_uv);
    float density = max(dye_density(v_uv), material.a);
    vec2 gradient = vec2(
        dye_density(v_uv + vec2(texel.x, 0.0)) - dye_density(v_uv - vec2(texel.x, 0.0)),
        dye_density(v_uv + vec2(0.0, texel.y)) - dye_density(v_uv - vec2(0.0, texel.y))
    ) * 0.5;
    float edge = clamp(length(gradient) * 10.0, 0.0, 1.0);
    float speed = clamp(length(texture(u_velocity, v_uv).xy), 0.0, 2.0);
    float curl = clamp(abs(texture(u_curl, v_uv).r) * 10.0, 0.0, 1.0);
    float material_luma = dot(material.rgb, vec3(0.299, 0.587, 0.114));

    float concentration = 1.0 + u_interaction_pinch * 0.20;
    float spread = 0.88 + u_interaction_openness * 0.16 + min(u_hand_distance, 1.0) * 0.08;
    float primary = pow(clamp(density * concentration * spread, 0.0, 1.0), u_density_response);
    float dye_split = clamp(abs(dye.r - dye.b) + abs(dye.g - dye.r) * 0.45, 0.0, 1.0);
    float secondary = pow(clamp(density * (0.36 + material_luma * 0.28 + dye_split * 0.42 + curl * 0.12), 0.0, 1.0), u_density_response);
    float structure = clamp(edge + curl * 0.42 + speed * 0.12 + min(u_interaction_velocity, 3.0) * 0.025, 0.0, 1.0);
    float breakup = clamp(structure * u_edge_breakup, 0.0, 1.0);
    frag_channels = vec4(primary, secondary, breakup, density);
}
