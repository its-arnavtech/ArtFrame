#version 330 core

in vec2 v_uv;
out vec2 frag_velocity;

uniform sampler2D u_velocity;
uniform vec2 u_texel_size;
uniform float u_maximum_velocity;

void main() {
    vec2 velocity = texture(u_velocity, v_uv).xy;
    if (any(isnan(velocity)) || any(isinf(velocity))) {
        velocity = vec2(0.0);
    }
    float speed = length(velocity);
    if (speed > u_maximum_velocity) {
        velocity *= u_maximum_velocity / speed;
    }
    vec2 wall = 1.5 * u_texel_size;
    bool at_wall = v_uv.x <= wall.x || v_uv.x >= 1.0 - wall.x;
    at_wall = at_wall || v_uv.y <= wall.y || v_uv.y >= 1.0 - wall.y;
    frag_velocity = at_wall ? vec2(0.0) : velocity;
}
