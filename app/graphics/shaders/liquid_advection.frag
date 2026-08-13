#version 330 core

in vec2 v_uv;
out vec4 frag_quantity;

uniform sampler2D u_quantity;
uniform sampler2D u_velocity;
uniform vec2 u_texel_size;
uniform float u_timestep;
uniform float u_decay;
uniform float u_diffusion;

void main() {
    vec2 velocity = texture(u_velocity, v_uv).xy;
    vec2 minimum_uv = 0.5 * u_texel_size;
    vec2 maximum_uv = vec2(1.0) - minimum_uv;
    vec2 previous_position = clamp(
        v_uv - velocity * u_timestep,
        minimum_uv,
        maximum_uv
    );
    // Linear texture filtering provides bilinear semi-Lagrangian sampling.
    vec4 quantity = texture(u_quantity, previous_position);
    if (u_diffusion > 0.0) {
        vec4 neighbors = texture(u_quantity, clamp(previous_position - vec2(u_texel_size.x, 0.0), minimum_uv, maximum_uv));
        neighbors += texture(u_quantity, clamp(previous_position + vec2(u_texel_size.x, 0.0), minimum_uv, maximum_uv));
        neighbors += texture(u_quantity, clamp(previous_position - vec2(0.0, u_texel_size.y), minimum_uv, maximum_uv));
        neighbors += texture(u_quantity, clamp(previous_position + vec2(0.0, u_texel_size.y), minimum_uv, maximum_uv));
        quantity = mix(quantity, neighbors * 0.25, clamp(u_diffusion * u_timestep * 60.0, 0.0, 1.0));
    }
    frag_quantity = quantity * u_decay;
}
