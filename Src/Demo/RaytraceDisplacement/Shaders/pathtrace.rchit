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
    
    // NOTE - 对于非微网格三角形，可能无法访问 gl_HitMicroTriangleVertexPositionsNV 或
    // gl_HitMicroTriangleVertexBarycentricsNV。
    bool isMicromesh=gl_HitKindEXT==gl_HitKindFrontFacingMicroTriangleNV||gl_HitKindEXT==gl_HitKindBackFacingMicroTriangleNV;
    bool isFront=gl_HitKindEXT==gl_HitKindFrontFacingTriangleEXT||gl_HitKindEXT==gl_HitKindFrontFacingMicroTriangleNV;
    
    vec3 wPos=gl_WorldRayOriginEXT+gl_WorldRayDirectionEXT*gl_HitTEXT;
    //NOTE - gl_HitTEXT是gl_RayTmaxEXT的别名，会在运行的过程中改变
    //其中，在closest hit shader中，gl_HitTEXT是最近击中点相对于射线原点的距离
    vec3 wDir=normalize(gl_WorldRayDirectionEXT);
    vec3 wEye=-wDir;
    vec3 wLight=normalize(skyInfo.directionToLight);
    vec3 wNorm=isMicromesh?
    normalize(cross(gl_HitMicroTriangleVertexPositionsNV[2]-gl_HitMicroTriangleVertexPositionsNV[0],
        gl_HitMicroTriangleVertexPositionsNV[2]-gl_HitMicroTriangleVertexPositionsNV[1])):
        normalize(cross(gl_HitTriangleVertexPositionsEXT[2]-gl_HitTriangleVertexPositionsEXT[0],
            gl_HitTriangleVertexPositionsEXT[2]-gl_HitTriangleVertexPositionsEXT[1]));
            
            float height=(wPos.y/pc.heightmapScale)*2.f+.5f;
            
            wNorm=isFront?wNorm:-wNorm;
            
            vec3 albedo=vec3(.2,.2,.8);
            
            // Add wireframe
            float opacity=pc.opacity;
            if(isMicromesh){
                // Color based on the height
                albedo=temperature(height);
                
                float wire=1.;
                const vec2 microBary2=baseToMicro(gl_HitMicroTriangleVertexBarycentricsNV,attribs);
                const vec3 microBary=vec3(1.F-microBary2.x-microBary2.y,microBary2.xy);
                wireframe(wire,.002F*pc.wireframeScale,microBary);//小三角形的线绘制
                
                const vec3 baseBary=vec3(1.-attribs.x-attribs.y,attribs.xy);
                wireframe(wire,.008F,baseBary);//大三角形的线绘制
                
                const vec3 wireColor=vec3(.3F,.3F,.3F);
                albedo=mix(wireColor,albedo,wire);
                opacity=mix(1.,opacity,wire);
            }
            
            float ior=isFront?(1./pc.refractiveIndex):pc.refractiveIndex;
            float reflectivity=fresnelSchlickApprox(wDir,wNorm,ior);
            vec3 reflectionWeight=payload.weight*reflectivity;
            vec3 refractionWeight=payload.weight*(1.-reflectivity);
            int newDepth=payload.depth+1;
            
            if(isFront){
                // Add light contribution unless in shadow
                bool visible=shadowRay(wPos,wLight);
                if(visible){
                    float diffuse=clamp(dot(wNorm,wLight),0.,1.);
                    payload.color+=payload.weight*albedo*diffuse*opacity;
                }
                //NOTE - 正面接收来自天光的漫反射（直接漫反射）
            }else{
                // Absorption - Beer's law
                vec3 density=vec3(.8,.8,.4);
                vec3 absorption=exp(-density*pc.density*gl_HitTEXT);
                //NOTE - 背面代表几何体的内部，是参与介质（半透明材质），会吸收光线
                reflectionWeight*=absorption;
                refractionWeight*=absorption;
            }
            refractionWeight*=(1.-opacity);
            
            // Note: the following follows both sides of the branch, which is slow
            
            // Reflection
            if(max(max(reflectionWeight.x,reflectionWeight.y),reflectionWeight.z)>.01){
                //NOTE - 任何一个色彩分量的反射权重 > 0，则进行反射
                vec3 reflectDir=reflect(wDir,wNorm);
                payload.weight=reflectionWeight;
                payload.depth=newDepth;
                traceRayEXT(topLevelAS,gl_RayFlagsCullBackFacingTrianglesEXT,0xFF,0,0,0,wPos,.0001,reflectDir,100.,0);
            }
            
            // Refraction
            if(max(max(refractionWeight.x,refractionWeight.y),refractionWeight.z)>.01){
                //NOTE - 任何一个色彩分量的折射权重 > 0，则进行折射
                vec3 refractDir=refract(wDir,wNorm,ior);
                payload.weight=refractionWeight;
                payload.depth=newDepth;
                traceRayEXT(topLevelAS,gl_RayFlagsCullBackFacingTrianglesEXT,0xFF,0,0,0,wPos,.0001,refractDir,100.,0);
            }
            
        }
        
        /**NOTE - 材质总结
        1. payload是一个共享存储区（对于每个像素而言，1spp），光线追踪traceRayEXT()是递归进行的，不存在对共享存储区的数据竞争
        
        2. payload.color是这个像素的最终颜色，是所有光线贡献的累加
        3. payload.weight是当前光线对最终颜色的贡献，payload.depth是当前光线递归的深度
        
        4. 主要材质：表面是漫反射，内部是有色玻璃
        5. 正面击中时，接收来自天光的漫反射，背面击中时，接收来自玻璃内部的散射（参与介质）
        
        6. 下一层射线由反射和折射构成。设置好权重和深度后，调用traceRayEXT()进入下一层射线
        7. 随着程序遍历所有光线的分支路径（路径追踪），payload.color会逐渐积累所有当前光线的颜色
        */
        