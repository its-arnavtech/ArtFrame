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
uniform float u_interaction_velocity;
uniform float u_interaction_pinch;
uniform float u_hand_distance;
uniform float u_time;

uniform int u_left_active;
uniform vec2 u_left_position;
uniform vec2 u_left_velocity;
uniform float u_left_pinch;
uniform float u_left_openness;
uniform int u_right_active;
uniform vec2 u_right_position;
uniform vec2 u_right_velocity;
uniform float u_right_pinch;
uniform float u_right_openness;

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
    float projection = clamp(dot(p - a, segment) / max(dot(segment, segment), 0.000001), 0.0, 1.0);
    return length(p - (a + segment * projection));
}

float capsule(vec2 point, vec2 start, vec2 end, float radius, float feather) {
    return 1.0 - smoothstep(radius, radius + feather, segment_distance(point, start, end));
}

float dye_density(vec2 uv) {
    vec4 dye = texture(u_dye, clamp(uv, 0.0, 1.0));
    return clamp(max(dye.a, max(dye.r, max(dye.g, dye.b))), 0.0, 1.0);
}

vec2 limited_trail(vec2 velocity) {
    float speed = length(velocity);
    return speed > 0.0001 ? velocity / speed * min(speed, 2.2) * 0.052 : vec2(0.0);
}

float hand_shape(
    vec2 point,
    int source_active,
    vec2 position,
    vec2 velocity,
    float pinch,
    float openness
) {
    if (source_active == 0) {
        return 0.0;
    }
    float radius = mix(0.042, 0.086, openness) * mix(1.12, 0.84, pinch);
    vec2 tail = position - limited_trail(velocity);
    float core = capsule(point, position, tail, radius, 0.018);
    float halo = capsule(point, position, tail, radius * 1.42, 0.032) * 0.34;
    return max(core, halo);
}

vec2 field_warp(vec2 uv) {
    vec2 flow = texture(u_velocity, clamp(uv, 0.0, 1.0)).xy;
    float first = sin(uv.y * 19.0 + uv.x * 7.0 + u_time * 0.42);
    float second = cos(uv.x * 23.0 - uv.y * 5.0 - u_time * 0.31);
    vec2 wave = vec2(first, second) * (0.0065 + min(length(flow), 2.0) * 0.0022);
    return clamp(uv + wave + flow * 0.0035, 0.0, 1.0);
}

float bridge_shape(vec2 point) {
    if (u_left_active == 0 || u_right_active == 0) {
        return 0.0;
    }
    float average_open = 0.5 * (u_left_openness + u_right_openness);
    float average_pinch = 0.5 * (u_left_pinch + u_right_pinch);
    float radius = mix(0.048, 0.105, average_open) * mix(0.82, 1.14, average_pinch);
    float body = capsule(point, u_left_position, u_right_position, radius, 0.025);
    float outer_membrane = capsule(
        point,
        u_left_position,
        u_right_position,
        radius * 1.32,
        0.040
    ) * 0.28;
    return max(body, outer_membrane);
}

float glass_field(vec2 uv) {
    vec2 shaped_uv = field_warp(uv);
    float transported = smoothstep(0.025, 0.56, dye_density(shaped_uv));
    float left = hand_shape(
        shaped_uv,
        u_left_active,
        u_left_position,
        u_left_velocity,
        u_left_pinch,
        u_left_openness
    );
    float right = hand_shape(
        shaped_uv,
        u_right_active,
        u_right_position,
        u_right_velocity,
        u_right_pinch,
        u_right_openness
    );
    return clamp(max(transported * 0.88, max(bridge_shape(shaped_uv), max(left, right))), 0.0, 1.0);
}

vec3 camera_sample(vec2 uv) {
    return texture(u_base_camera, clamp(uv, 0.002, 0.998)).bgr;
}

void main() {
    float field = glass_field(v_uv);
    vec2 gradient_step = u_texel_size * 1.65;
    vec2 gradient = vec2(
        glass_field(v_uv + vec2(gradient_step.x, 0.0)) - glass_field(v_uv - vec2(gradient_step.x, 0.0)),
        glass_field(v_uv + vec2(0.0, gradient_step.y)) - glass_field(v_uv - vec2(0.0, gradient_step.y))
    ) * 0.5;

    vec2 velocity = texture(u_velocity, v_uv).xy;
    float speed = min(length(velocity), 2.5);
    float curl = texture(u_curl, v_uv).r;
    float pressure = texture(u_pressure, v_uv).r;
    float edge = smoothstep(0.012, 0.20, length(gradient));
    float body = smoothstep(0.025, 0.28, field);
    vec2 surface_ripple = vec2(
        sin(v_uv.y * 31.0 + v_uv.x * 9.0 + u_time * 0.68 + curl * 5.0),
        cos(v_uv.x * 37.0 - v_uv.y * 7.0 - u_time * 0.54 + pressure * 8.0)
    );
    surface_ripple *= body * (0.11 + speed * 0.045 + min(u_interaction_velocity, 3.0) * 0.025);
    vec3 normal = normalize(vec3(
        -gradient * (10.0 + edge * 13.0) + surface_ripple,
        0.42 + body * 0.55
    ));

    vec2 camera_uv = vec2(v_uv.x, 1.0 - v_uv.y);
    vec2 screen_normal = vec2(normal.x, -normal.y);
    float interaction_energy = min(u_interaction_velocity, 3.0) / 3.0;
    float refraction_strength = mix(0.010, 0.052, body) * (1.0 + edge * 0.75);
    refraction_strength *= 1.0 + u_interaction_pinch * 0.20 + interaction_energy * 0.12;
    vec2 flow_offset = vec2(velocity.x, -velocity.y) * 0.0065;
    vec2 refraction = screen_normal * refraction_strength + flow_offset;

    float spectral_width = (0.0012 + edge * 0.0055) * body;
    vec2 spectral_axis = normalize(screen_normal + vec2(0.0001, 0.0001));
    vec3 refracted;
    refracted.r = camera_sample(camera_uv + refraction + spectral_axis * spectral_width).r;
    refracted.g = camera_sample(camera_uv + refraction).g;
    refracted.b = camera_sample(camera_uv + refraction - spectral_axis * spectral_width).b;

    vec2 reflection_uv = vec2(
        1.0 - camera_uv.x + screen_normal.x * 0.075,
        1.0 - camera_uv.y + screen_normal.y * 0.12
    );
    vec3 reflected_scene = camera_sample(reflection_uv);
    float reflected_luma = dot(reflected_scene, vec3(0.299, 0.587, 0.114));
    reflected_luma = smoothstep(0.10, 0.88, reflected_luma);

    float warped_height = v_uv.y + normal.y * 0.20 + curl * 0.025 + pressure * 0.018;
    float broad_band = 0.5 + 0.5 * sin(warped_height * 18.0 + normal.x * 4.5);
    float fine_band = 0.5 + 0.5 * sin(warped_height * 47.0 - normal.x * 9.0);
    vec3 studio_light = mix(vec3(0.035, 0.045, 0.060), vec3(1.0, 0.98, 0.93), smoothstep(0.20, 0.82, broad_band));
    studio_light = mix(studio_light, vec3(0.18, 0.22, 0.27), fine_band * 0.24);

    vec3 chrome = mix(vec3(reflected_luma), reflected_scene, 0.34);
    chrome = mix(chrome, studio_light, 0.48 + edge * 0.16);
    float fresnel = pow(clamp(1.0 - normal.z, 0.0, 1.0), 2.15);
    float metal_amount = clamp(0.38 + fresnel * 0.52 + edge * 0.20 + u_interaction_pinch * 0.10, 0.0, 0.95);
    vec3 glass = mix(refracted, chrome, metal_amount);

    vec3 light_one = normalize(vec3(-0.46, 0.72, 0.52));
    vec3 light_two = normalize(vec3(0.72, -0.18, 0.66));
    float highlight = pow(max(dot(normal, light_one), 0.0), 28.0);
    highlight += pow(max(dot(normal, light_two), 0.0), 54.0) * 0.72;
    float rim_light = edge * (0.52 + fresnel * 0.75);
    glass *= 1.0 - edge * 0.18;
    glass += vec3(1.0, 0.985, 0.95) * (highlight * 0.82 + rim_light * 0.42);
    glass += vec3(0.16, 0.30, 0.42) * max(curl, 0.0) * 0.06;
    glass += vec3(0.34, 0.12, 0.24) * max(-curl, 0.0) * 0.05;

    float micro_ripple = sin((v_uv.x + v_uv.y) * 410.0 + curl * 16.0) * u_texture_strength;
    glass *= 1.0 + micro_ripple * edge * 0.08;
    float alpha = body * (0.64 + fresnel * 0.18 + edge * 0.18);
    alpha *= clamp(u_material_intensity, 0.0, 1.35);
    frag_material = vec4(clamp(glass, 0.0, 1.0), clamp(alpha, 0.0, 0.98));
}
