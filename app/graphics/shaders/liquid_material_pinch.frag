#version 330 core

in vec2 v_uv;
out vec4 frag_material;

uniform sampler2D u_base_camera;
uniform sampler2D u_dye;
uniform sampler2D u_velocity;
uniform sampler2D u_curl;
uniform sampler2D u_pressure;
uniform vec2 u_texel_size;
uniform vec2 u_display_size;
uniform float u_material_intensity;
uniform float u_texture_strength;
uniform float u_glass_refraction;
uniform float u_glass_dispersion;
uniform float u_glass_roughness;
uniform float u_glass_edge_brightness;
uniform float u_interaction_velocity;

uniform int u_left_active;
uniform vec2 u_left_position;
uniform vec2 u_left_velocity;
uniform float u_left_pinch;
uniform float u_left_openness;
uniform float u_left_influence;
uniform int u_right_active;
uniform vec2 u_right_position;
uniform vec2 u_right_velocity;
uniform float u_right_pinch;
uniform float u_right_openness;
uniform float u_right_influence;

float aspect_ratio() {
    return u_display_size.x / max(u_display_size.y, 1.0);
}

vec2 metric(vec2 point) {
    return vec2(point.x * aspect_ratio(), point.y);
}

float segment_distance(vec2 point, vec2 start, vec2 end) {
    vec2 p = metric(point);
    vec2 a = metric(start);
    vec2 b = metric(end);
    vec2 segment = b - a;
    float projection = clamp(
        dot(p - a, segment) / max(dot(segment, segment), 0.000001),
        0.0,
        1.0
    );
    return length(p - (a + segment * projection));
}

float smooth_union(float first, float second, float softness) {
    float blend = clamp(0.5 + 0.5 * (first - second) / softness, 0.0, 1.0);
    return mix(second, first, blend) + softness * blend * (1.0 - blend);
}

vec2 trail_vector(vec2 velocity) {
    float speed = length(velocity);
    return speed > 0.0001
        ? velocity / speed * min(speed, 2.8) * 0.060
        : vec2(0.0);
}

float pinch_source(
    vec2 point,
    int source_active,
    vec2 position,
    vec2 velocity,
    float pinch,
    float openness,
    float influence
) {
    if (source_active == 0 || influence <= 0.0001) {
        return 0.0;
    }
    float radius = mix(0.055, 0.092, openness) * mix(1.08, 0.90, pinch);
    vec2 tail = position - trail_vector(velocity);
    float distance_to_trail = segment_distance(point, position, tail);
    float volume = 1.0 - smoothstep(radius * 0.72, radius, distance_to_trail);
    float membrane = 1.0 - smoothstep(radius, radius * 1.23, distance_to_trail);
    return smooth_union(volume, membrane * 0.32, 0.12) * influence;
}

float pinch_field(vec2 uv) {
    float left = pinch_source(
        uv,
        u_left_active,
        u_left_position,
        u_left_velocity,
        u_left_pinch,
        u_left_openness,
        u_left_influence
    );
    float right = pinch_source(
        uv,
        u_right_active,
        u_right_position,
        u_right_velocity,
        u_right_pinch,
        u_right_openness,
        u_right_influence
    );
    return clamp(smooth_union(left, right, 0.14), 0.0, 1.0);
}

vec3 camera_sample(vec2 uv) {
    return texture(u_base_camera, clamp(uv, 0.002, 0.998)).bgr;
}

void main() {
    float center = pinch_field(v_uv);
    vec2 step_uv = u_texel_size * 1.45;
    float positive_x = pinch_field(v_uv + vec2(step_uv.x, 0.0));
    float negative_x = pinch_field(v_uv - vec2(step_uv.x, 0.0));
    float positive_y = pinch_field(v_uv + vec2(0.0, step_uv.y));
    float negative_y = pinch_field(v_uv - vec2(0.0, step_uv.y));
    vec2 gradient = 0.5 * vec2(
        positive_x - negative_x,
        positive_y - negative_y
    );
    float curvature = positive_x + negative_x + positive_y + negative_y - 4.0 * center;
    float coverage = smoothstep(0.018, 0.22, center);
    float edge = smoothstep(0.008, 0.115, length(gradient)) * coverage;

    vec2 velocity = texture(u_velocity, v_uv).xy;
    float speed = min(length(velocity), 3.0);
    float curl = clamp(texture(u_curl, v_uv).r, -2.5, 2.5);
    float pressure = clamp(texture(u_pressure, v_uv).r, -1.0, 1.0);
    vec2 flow_normal = velocity * (0.020 + u_texture_strength * 0.018);
    flow_normal += vec2(-velocity.y, velocity.x) * curl * 0.005;
    vec3 normal = normalize(vec3(
        -gradient * (15.0 + edge * 9.0) + flow_normal * coverage,
        0.76 + coverage * 0.50
    ));

    vec2 camera_uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec2 screen_normal = vec2(normal.x, -normal.y);
    float motion_energy = min(u_interaction_velocity, 3.0) / 3.0;
    float refraction_strength = u_glass_refraction * (0.52 + coverage * 0.92);
    refraction_strength *= 1.0 + edge * 0.42 + motion_energy * 0.12;
    vec2 refraction = screen_normal * refraction_strength;
    refraction += vec2(velocity.x, -velocity.y) * 0.0055 * coverage;

    vec2 optical_axis = normalize(screen_normal + vec2(0.0001));
    float dispersion = u_glass_dispersion * (0.35 + edge * 0.88);
    vec2 refracted_uv = camera_uv + refraction;
    vec3 transmitted;
    transmitted.r = camera_sample(refracted_uv + optical_axis * dispersion).r;
    transmitted.g = camera_sample(refracted_uv).g;
    transmitted.b = camera_sample(refracted_uv - optical_axis * dispersion).b;

    vec4 dye = texture(u_dye, v_uv);
    float dye_density = clamp(max(dye.a, max(dye.r, max(dye.g, dye.b))), 0.0, 1.0);
    vec3 dye_color = dye.rgb / max(max(dye.r, max(dye.g, dye.b)), 0.08);
    vec2 flow_axis = velocity / max(length(velocity), 0.0001);
    float flow_coordinate = dot(metric(v_uv), metric(flow_axis));
    float internal_flow = 0.5 + 0.5 * sin(
        flow_coordinate * 48.0 + pressure * 13.0 + curl * 3.2
    );
    float liquid_mix = coverage * (0.14 + dye_density * 0.18 + internal_flow * speed * 0.018);
    vec3 liquid_tint = mix(vec3(0.72, 0.88, 0.98), dye_color, 0.34);
    vec3 glass = mix(transmitted, transmitted * 0.78 + liquid_tint * 0.22, liquid_mix);

    float fresnel = 0.04 + 0.96 * pow(1.0 - clamp(normal.z, 0.0, 1.0), 5.0);
    vec3 view_direction = vec3(0.0, 0.0, 1.0);
    vec3 light = normalize(vec3(-0.42, 0.68, 0.72));
    float highlight_power = mix(112.0, 30.0, u_glass_roughness);
    float highlight = pow(
        max(dot(normal, normalize(light + view_direction)), 0.0),
        highlight_power
    );
    float caustic = edge * smoothstep(0.008, 0.11, -curvature);
    glass = mix(glass, vec3(0.90, 0.95, 1.0), fresnel * 0.18);
    glass += vec3(1.0, 0.985, 0.95) * (
        highlight * 0.76 + edge * u_glass_edge_brightness * 0.32
    );
    glass += vec3(0.60, 0.82, 1.0) * caustic * 0.18;

    // A high interior alpha makes this a visible liquid volume rather than a
    // transparent hole bounded only by a ring.
    float alpha = coverage * (0.88 + edge * 0.05 + fresnel * 0.04);
    alpha *= clamp(u_material_intensity, 0.0, 1.25);
    frag_material = vec4(clamp(glass, 0.0, 1.0), clamp(alpha, 0.0, 0.97));
}
