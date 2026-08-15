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
uniform float u_interaction_pinch;
uniform float u_hand_distance;

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

float capsule(vec2 point, vec2 start, vec2 end, float radius, float feather) {
    return 1.0 - smoothstep(radius, radius + feather, segment_distance(point, start, end));
}

float smooth_union(float first, float second, float softness) {
    float blend = clamp(0.5 + 0.5 * (first - second) / softness, 0.0, 1.0);
    return mix(second, first, blend) + softness * blend * (1.0 - blend);
}

float dye_density(vec2 uv) {
    vec4 dye = texture(u_dye, clamp(uv, 0.0, 1.0));
    return clamp(max(dye.a, max(dye.r, max(dye.g, dye.b))), 0.0, 1.0);
}

vec2 limited_trail(vec2 velocity) {
    float speed = length(velocity);
    return speed > 0.0001
        ? velocity / speed * min(speed, 2.2) * 0.050
        : vec2(0.0);
}

float hand_shape(
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
    float radius = mix(0.044, 0.088, openness) * mix(1.10, 0.86, pinch);
    vec2 tail = position - limited_trail(velocity);
    float core = capsule(point, position, tail, radius, 0.020);
    float membrane = capsule(point, position, tail, radius * 1.34, 0.034) * 0.28;
    return smooth_union(core, membrane, 0.12) * influence;
}

float bridge_shape(vec2 point) {
    if (
        u_left_active == 0 || u_right_active == 0
        || u_left_influence <= 0.0001 || u_right_influence <= 0.0001
    ) {
        return 0.0;
    }
    float average_open = 0.5 * (u_left_openness + u_right_openness);
    float average_pinch = 0.5 * (u_left_pinch + u_right_pinch);
    float radius = mix(0.050, 0.108, average_open) * mix(0.84, 1.12, average_pinch);
    float core = capsule(point, u_left_position, u_right_position, radius, 0.026);
    float membrane = capsule(
        point,
        u_left_position,
        u_right_position,
        radius * 1.28,
        0.042
    ) * 0.25;
    return smooth_union(core, membrane, 0.13) * min(u_left_influence, u_right_influence);
}

vec2 source_lens(
    vec2 point,
    int source_active,
    vec2 position,
    float openness,
    float influence
) {
    if (source_active == 0 || influence <= 0.0001) {
        return vec2(0.0);
    }
    vec2 delta = metric(point) - metric(position);
    float radius = mix(0.064, 0.110, openness);
    float normalized_distance = clamp(length(delta) / radius, 0.0, 1.0);
    float lens_profile = sin(normalized_distance * 3.14159265) * influence;
    vec2 direction = delta / max(length(delta), 0.00001);
    return vec2(direction.x / aspect_ratio(), direction.y) * lens_profile;
}

vec2 field_warp(vec2 uv) {
    vec2 flow = texture(u_velocity, clamp(uv, 0.0, 1.0)).xy;
    float curl = clamp(texture(u_curl, clamp(uv, 0.0, 1.0)).r, -2.0, 2.0);
    vec2 transverse = vec2(-flow.y, flow.x);
    // Field motion comes only from the solver. Avoid independent animated noise,
    // which makes a refractive surface crawl while the fluid is stationary.
    vec2 displacement = flow * 0.0042 + transverse * curl * 0.00055;
    return clamp(uv + displacement, 0.0, 1.0);
}

float glass_field(vec2 uv) {
    vec2 shaped_uv = field_warp(uv);
    float transported = smoothstep(0.018, 0.50, dye_density(shaped_uv));
    float left = hand_shape(
        shaped_uv,
        u_left_active,
        u_left_position,
        u_left_velocity,
        u_left_pinch,
        u_left_openness,
        u_left_influence
    );
    float right = hand_shape(
        shaped_uv,
        u_right_active,
        u_right_position,
        u_right_velocity,
        u_right_pinch,
        u_right_openness,
        u_right_influence
    );
    float source_surface = smooth_union(left, right, 0.16);
    source_surface = smooth_union(source_surface, bridge_shape(shaped_uv), 0.18);
    return clamp(smooth_union(transported * 0.90, source_surface, 0.14), 0.0, 1.0);
}

vec3 camera_sample(vec2 uv) {
    return texture(u_base_camera, clamp(uv, 0.002, 0.998)).bgr;
}

vec3 refracted_camera(
    vec2 camera_uv,
    vec2 refraction,
    vec2 optical_axis,
    vec2 tangent,
    float dispersion,
    float roughness
) {
    vec2 center = camera_uv + refraction;
    vec3 transmitted;
    transmitted.r = camera_sample(center + optical_axis * dispersion).r;
    transmitted.g = camera_sample(center).g;
    transmitted.b = camera_sample(center - optical_axis * dispersion).b;

    vec2 rough_offset = tangent * (0.45 + roughness * 1.65) / u_display_size;
    vec3 shoulder = 0.5 * (
        camera_sample(center + rough_offset) + camera_sample(center - rough_offset)
    );
    return mix(transmitted, shoulder, roughness * 0.22);
}

vec3 environment_reflection(vec3 normal, vec2 camera_uv, vec2 refraction) {
    // A neutral studio environment gives readable glass highlights without
    // turning the surface into an opaque, flipped chrome mirror.
    vec3 reflected_direction = reflect(vec3(0.0, 0.0, -1.0), normal);
    float upper_softbox = smoothstep(0.10, 0.82, reflected_direction.y);
    float side_softbox = pow(max(reflected_direction.x * 0.5 + 0.5, 0.0), 5.0);
    vec3 studio = mix(vec3(0.025, 0.032, 0.043), vec3(0.92, 0.96, 1.0), upper_softbox);
    studio += vec3(1.0, 0.94, 0.86) * side_softbox * 0.16;

    vec3 local_scene = camera_sample(camera_uv - refraction * 0.34);
    return mix(studio, local_scene, 0.16);
}

void main() {
    float center = glass_field(v_uv);
    vec2 step_uv = u_texel_size * 1.55;
    float positive_x = glass_field(v_uv + vec2(step_uv.x, 0.0));
    float negative_x = glass_field(v_uv - vec2(step_uv.x, 0.0));
    float positive_y = glass_field(v_uv + vec2(0.0, step_uv.y));
    float negative_y = glass_field(v_uv - vec2(0.0, step_uv.y));

    vec2 gradient = 0.5 * vec2(
        positive_x - negative_x,
        positive_y - negative_y
    );
    float curvature = positive_x + negative_x + positive_y + negative_y - 4.0 * center;
    float gradient_magnitude = length(gradient);
    float coverage = smoothstep(0.025, 0.24, center);
    float membrane = smoothstep(0.006, 0.12, gradient_magnitude) * coverage;

    vec2 velocity = texture(u_velocity, v_uv).xy;
    float speed = min(length(velocity), 2.5);
    float curl = clamp(texture(u_curl, v_uv).r, -2.0, 2.0);
    float pressure = clamp(texture(u_pressure, v_uv).r, -1.0, 1.0);
    vec2 flow_detail = velocity * (0.012 + u_texture_strength * 0.022);
    flow_detail += vec2(-velocity.y, velocity.x) * curl * 0.004;
    vec2 volume_lens = source_lens(
        v_uv,
        u_left_active,
        u_left_position,
        u_left_openness,
        u_left_influence
    );
    volume_lens += source_lens(
        v_uv,
        u_right_active,
        u_right_position,
        u_right_openness,
        u_right_influence
    );

    vec3 normal = normalize(vec3(
        -gradient * (12.0 + membrane * 10.0)
            + volume_lens * 0.74
            + flow_detail * coverage,
        0.82 + coverage * 0.54
    ));
    vec2 screen_normal = vec2(normal.x, -normal.y);
    vec2 optical_axis = normalize(screen_normal + vec2(0.0001));
    vec2 tangent = vec2(-optical_axis.y, optical_axis.x);

    float interaction_energy = min(u_interaction_velocity, 3.0) / 3.0;
    float lens_depth = mix(0.22, 1.0, coverage);
    float refraction_amount = u_glass_refraction * lens_depth;
    refraction_amount *= 1.0 + membrane * 0.36 + u_interaction_pinch * 0.10;
    vec2 flow_refraction = vec2(velocity.x, -velocity.y) * 0.0038;
    vec2 refraction = screen_normal * refraction_amount + flow_refraction * coverage;

    vec2 camera_uv = vec2(v_uv.x, 1.0 - v_uv.y);
    float dispersion = u_glass_dispersion * (0.30 + membrane * 0.92);
    dispersion *= 1.0 + interaction_energy * 0.12;
    vec3 transmission = refracted_camera(
        camera_uv,
        refraction,
        optical_axis,
        tangent,
        dispersion,
        u_glass_roughness
    );

    float view_facing = clamp(normal.z, 0.0, 1.0);
    float fresnel = 0.04 + 0.96 * pow(1.0 - view_facing, 5.0);
    vec3 reflection = environment_reflection(normal, camera_uv, refraction);
    vec3 glass = mix(transmission, reflection, fresnel);
    vec3 transported_color = texture(u_dye, v_uv).rgb;
    vec3 volume_tint = mix(vec3(0.86, 0.93, 0.98), transported_color, 0.20);
    glass = mix(glass, glass * 0.94 + volume_tint * 0.06, coverage * 0.18);

    vec3 view_direction = vec3(0.0, 0.0, 1.0);
    vec3 light_one = normalize(vec3(-0.42, 0.66, 0.72));
    vec3 light_two = normalize(vec3(0.68, -0.24, 0.70));
    float highlight_power = mix(118.0, 34.0, u_glass_roughness);
    float highlight = pow(
        max(dot(normal, normalize(light_one + view_direction)), 0.0),
        highlight_power
    );
    highlight += pow(
        max(dot(normal, normalize(light_two + view_direction)), 0.0),
        highlight_power * 0.72
    ) * 0.45;

    float outer_rim = membrane * (0.45 + fresnel * 1.8);
    float inner_shadow = membrane * smoothstep(-0.015, 0.055, curvature);
    float caustic = membrane * smoothstep(0.010, 0.12, -curvature);
    glass *= 1.0 - inner_shadow * 0.18;
    glass += vec3(1.0, 0.985, 0.955) * (
        highlight * 0.72 + outer_rim * u_glass_edge_brightness * 0.46
    );
    glass += vec3(0.78, 0.90, 1.0) * caustic * 0.20;

    // Pressure and curl color only the deepest optical path very slightly.
    // They should describe the flow, not read as an opaque blue material.
    glass += vec3(0.055, 0.085, 0.12) * max(pressure, 0.0) * coverage * 0.05;
    glass += vec3(0.08, 0.05, 0.09) * abs(curl) * coverage * 0.008;

    float alpha = coverage * (0.80 + fresnel * 0.14 + membrane * 0.05);
    alpha *= clamp(u_material_intensity, 0.0, 1.35);
    frag_material = vec4(clamp(glass, 0.0, 1.0), clamp(alpha, 0.0, 0.96));
}
