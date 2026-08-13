#version 330 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_dye;
uniform sampler2D u_flow;
uniform float u_delta_seconds;

void main() {
    vec2 velocity = texture(u_flow, v_uv).xy;
    vec2 source_uv = v_uv - velocity * u_delta_seconds;
    frag_color = texture(u_dye, source_uv);
}
