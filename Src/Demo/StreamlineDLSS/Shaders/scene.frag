#version 450
#extension GL_GOOGLE_include_directive:enable
#include "common.h"

layout(location=0)in vec3 inWorldPos;
layout(location=1)in vec3 inWorldNormal;
layout(location=2)in vec3 inWorldPosPrev;

layout(location=0)out vec4 outColor;
layout(location=1)out vec2 outMotion;

layout(push_constant)uniform NodeInfo_
{
  NodeInfo nodeInfo;
};

layout(set=0,binding=0)uniform FrameInfo_
{
  FrameInfo frameInfo;
};

vec4 getColor(vec3 viewDir,vec3 normal)
{
  vec3 lightDir=viewDir;
  
  vec4 color=vec4(nodeInfo.color,1.);
  // Diffuse
  color.rgb*=abs(dot(normal,lightDir));
  // Specular
  color.rgb+=pow(max(0.,dot(normal,normalize(lightDir+viewDir))),16.);
  
  return color;//Phong Lighting
}

vec2 getMotion(vec3 worldPos,vec3 worldPosPrev){
  vec4 clipPos=frameInfo.viewProj*vec4(worldPos,1.);
  vec4 clipPosPrev=frameInfo.viewProjPrev*vec4(worldPosPrev,1.);
  return((clipPosPrev.xy/clipPosPrev.w)-(clipPos.xy/clipPos.w))*.5;
  //REVIEW - ???为什么要乘以0.5
}

void main(){
  outColor=getColor(normalize(frameInfo.camPos-inWorldPos),inWorldNormal);
  outMotion=getMotion(inWorldPos,inWorldPosPrev);
}
