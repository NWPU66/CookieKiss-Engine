#version 460
#extension GL_GOOGLE_include_directive:enable
#include "common.h"

layout(location=0)in vec3 position;

layout(location=0)out vec2 uv;
layout(location=1)out vec3 color;

layout(push_constant)uniform PushContent_
{
    PushContent pc;
};

struct NormalBufferStruct{
    vec3 normal;
};
struct UVBufferStruct{
    vec2 uv;
};
layout(set=0,binding=0)buffer NormalBuffer{
    NormalBufferStruct normals[];
};
layout(set=0,binding=1)buffer UVBuffer{
    UVBufferStruct uvs[];
};

void main(){
    gl_Position=pc.mvp*vec4(position,1.);
    
    color=normals[gl_VertexIndex].normal;
    uv=uvs[gl_VertexIndex].uv;
}
