#version 460

layout(location=0)in vec2 uv;
layout(location=1)in vec3 color;

layout(location=0)out vec4 fragmentColor;

void main()
{
    fragmentColor=vec4(color,1);
}