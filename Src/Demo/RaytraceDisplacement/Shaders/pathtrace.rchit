#version 460
#extension GL_EXT_ray_tracing:require
#extension GL_EXT_nonuniform_qualifier:enable
#extension GL_EXT_scalar_block_layout:enable
#extension GL_GOOGLE_include_directive:enable
#extension GL_EXT_shader_explicit_arithmetic_types_int32:require
#extension GL_EXT_shader_explicit_arithmetic_types_int64:require
#extension GL_EXT_buffer_reference2:require
#extension GL_NV_displacement_micromap:require
#extension GL_EXT_ray_tracing_position_fetch:require

#include "device_host.h"
#include "animate_heightmap.h"
#include "dh_bindings.h"
#include "payload.h"
#include "nvvkhl/shaders/dh_sky.h"

// Barycentric coordinates of hit location relative to triangle vertices
hitAttributeEXT vec2 attribs;

// clang-format off
layout(location=0)rayPayloadInEXT HitPayload payload;
layout(set=0,binding=BRtTlas)uniform accelerationStructureEXT topLevelAS;
layout(set=0,binding=BRtFrameInfo)uniform FrameInfo_{FrameInfo frameInfo;};
layout(set=0,binding=BRtSkyParam)uniform SkyInfo_{SimpleSkyParameters skyInfo;};
layout(push_constant)uniform RtxPushConstant_{PushConstant pc;};
// clang-format on

// Return true if there is no occluder, meaning that the light is visible from P toward L
bool shadowRay(vec3 P,vec3 L){
    const uint rayFlags=gl_RayFlagsTerminateOnFirstHitEXT|gl_RayFlagsSkipClosestHitShaderEXT|gl_RayFlagsCullBackFacingTrianglesEXT;
    HitPayload savedP=payload;
    traceRayEXT(topLevelAS,rayFlags,0xFF,0,0,0,P,.0001,L,100.,0);
    bool visible=(payload.depth==MISS_DEPTH);
    payload=savedP;
    return visible;
}

float fresnelSchlickApprox(vec3 incident,vec3 normal,float ior){
    float r0=(ior-1.)/(ior+1.);
    r0*=r0;
    float cosX=-dot(normal,incident);
    if(ior>1.)
    {
        float sinT2=ior*ior*(1.-cosX*cosX);
        // Total internal reflection
        if(sinT2>1.){return 1.;}
        cosX=sqrt(1.-sinT2);
    }
    float x=1.-cosX;
    float ret=r0+(1.-r0)*x*x*x*x*x;
    return ret;
}

// utility for temperature
float fade(float low,float high,float value){
    float mid=(low+high)*.5;
    float range=(high-low)*.5;
    float x=1.-clamp(abs(mid-value)/range,0.,1.);
    return smoothstep(0.,1.,x);
}

// Return a cold-hot color based on intensity [0-1]
vec3 temperature(float intensity)
{
    const vec3 water=vec3(0.,0.,.5);
    const vec3 sand=vec3(.8,.7,.4);
    const vec3 green=vec3(.1,.4,.1);
    const vec3 rock=vec3(.4,.4,.4);
    const vec3 snow=vec3(1.,1.,1.);
    
    vec3 color=(fade(-.25,.25,intensity)*water//
    +fade(0.,.5,intensity)*sand//
    +fade(.25,.75,intensity)*green//
    +fade(.5,1.,intensity)*rock//
    +smoothstep(.75,1.,intensity)*snow);
    return color;
}

vec2 baseToMicro(vec2 barycentrics[3],vec2 p)
{
    vec2 ap=p-barycentrics[0];
    vec2 ab=barycentrics[1]-barycentrics[0];
    vec2 ac=barycentrics[2]-barycentrics[0];
    float rdet=1.f/(ab.x*ac.y-ab.y*ac.x);
    return vec2(ap.x*ac.y-ap.y*ac.x,ap.y*ab.x-ap.x*ab.y)*rdet;
    
    /**NOTE - 函数解释
    大三角形面片A1、A2、A3，一级细分出4个小三角形面片，6个顶点B1、B2、B3、B4、B5、B6
    p是击中点相对于大三角形面片A1、A2、A3的重心坐标
    barycentrics[3]是击中的micro三角形面片顶点（B1、B3、B5）相对于大三角形面片A1、A2、A3的重心坐标
    
    返回值是p点相对于micro三角形面片B1、B3、B5的重心坐标
    */
}

void wireframe(inout float wire,float width,vec3 bary){
    float minBary=min(bary.x,min(bary.y,bary.z));
    wire=min(wire,smoothstep(width,width+.002F,minBary));
    //NOTE - 线是黑色的
}

void main()
{
    // We hit our max depth
    if(payload.depth>=pc.maxDepth)
    {
        return;
    }
}

