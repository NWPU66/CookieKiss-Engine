#version 450
#extension GL_GOOGLE_include_directive:enable
#include "common.h"

layout(location=0)in vec3 inPos;
layout(location=1)in vec3 inNormal;

layout(location=0)out vec3 outWorldPos;
layout(location=1)out vec3 outWorldNormal;
layout(location=2)out vec3 outWorldPosPrev;

layout(push_constant)uniform NodeInfo_
{
    NodeInfo nodeInfo;
};

layout(set=0,binding=0)uniform FrameInfo_
{
    FrameInfo frameInfo;
};

void main(){
    vec4 pos=nodeInfo.model*vec4(inPos,1.);
    gl_Position=frameInfo.viewProj*pos;
    
    // Apply jitter offset
    gl_Position.xy+=frameInfo.jitterOffset*gl_Position.w;
    
    outWorldPos=pos.xyz;
    outWorldNormal=inNormal;//REVIEW - ???这个是local空间的法向吧
    
    vec4 posPrev=nodeInfo.modelPrev*vec4(inPos,1.);
    outWorldPosPrev=posPrev.xyz;
}
