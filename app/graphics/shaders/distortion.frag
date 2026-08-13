#version 330 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_source;
uniform sampler2D u_flow;
uniform float u_amount;

void main() {
    vec2 displacement = texture(u_flow, v_uv).xy * u_amount;
    frag_color = texture(u_source, v_uv + displacement);
}
