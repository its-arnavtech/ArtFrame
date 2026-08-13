#version 330 core

in vec2 v_uv;
out vec4 frag_color;

uniform sampler2D u_base;
uniform sampler2D u_effect;

void main() {
    vec4 base = texture(u_base, v_uv);
    vec4 effect = texture(u_effect, v_uv);
    frag_color = vec4(mix(base.rgb, effect.rgb, effect.a), 1.0);
}
