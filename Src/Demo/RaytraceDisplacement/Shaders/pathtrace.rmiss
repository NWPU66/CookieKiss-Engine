#version 460
#extension GL_EXT_ray_tracing:require
#extension GL_GOOGLE_include_directive:enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64:require

#include "device_host.h"
#include "payload.h"
#include "dh_bindings.h"
#include "nvvkhl/shaders/dh_sky.h"

layout(location=0)rayPayloadInEXT HitPayload payload;

layout(set=0,binding=BRtSkyParam)uniform SkyInfo_
{
    SimpleSkyParameters skyInfo;
};

void main()
{
    vec3 sky_color=evalSimpleSky(skyInfo,gl_WorldRayDirectionEXT);
    payload.color+=sky_color*payload.weight;
    payload.depth=MISS_DEPTH;// Stop
}
