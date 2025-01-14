#version 460
#extension GL_GOOGLE_include_directive:enable
#include "common.h"

layout(location=0)in vec3 position;
layout(location=1)in vec3 normal;
layout(location=2)in vec2 texcoord;

layout(location=0)out vec2 uv;
layout(location=1)out vec3 color;

layout(push_constant)uniform _PushContent{
    PushContent pc;
};

void main(){
    gl_Position=pc.mvp*vec4(position,1.);
    
    color=normal;
    uv=texcoord;
}

