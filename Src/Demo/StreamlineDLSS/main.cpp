/**
 * @file main.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-11-28
 *
 * @copyright Copyright (c) 2024
 *
 */

// c
#include <cstdint>
#include <cstdlib>

// cpp
#include <algorithm>
#include <exception>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// 3rdparty
#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#include "backends/imgui_impl_glfw.h"
#include "backends/imgui_impl_vulkan.h"
#include "glm/glm.hpp"
#include "imgui.h"
#include <vulkan/vulkan_core.h>

static const std::string PROJECT_NAME = "StreamlineDLSS";

// 3rdparty - nvvk
#define VMA_IMPLEMENTATION
#include "nvh/cameramanipulator.hpp"
#include "nvh/primitives.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/memallocator_vma_vk.hpp"
#include "nvvk/pipeline_vk.hpp"

// 3rdparty - nvvkhl
#include "nvvkhl/appbase_vk.hpp"
#include "nvvkhl/gbuffer.hpp"

// users
#define SL_MANUAL_HOOKING true
#include "streamline_wrapper.hpp"
// NOTE - streamline_wrapper负责动态加载sl的函数，main.cpp不需要再include sl的头文件
#include "Shaders/common.h"
#include "Shaders/spirv/generated_spirv/post_frag.h"
#include "Shaders/spirv/generated_spirv/post_vert.h"
#include "Shaders/spirv/generated_spirv/scene_frag.h"
#include "Shaders/spirv/generated_spirv/scene_vert.h"

// global variables
static const int SAMPLE_WIDTH  = 1920;
static const int SAMPLE_HEIGHT = 1080;
// Streamline features to enable
static const sl::Feature SL_FEATURES[] = {
    sl::kFeatureReflex,  // Reflex is required for DLSS Frame Generation
    sl::kFeatureDLSS,    // DLSS Super Resolution
    sl::kFeatureDLSS_G,  // DLSS Frame Generation
};

static const VkFormat SAMPLE_DEPTH_FORMAT =
    VK_FORMAT_D24_UNORM_S8_UINT;  // Streamline only supports VK_FORMAT_D24_UNORM_S8_UINT and
                                  // VK_FORMAT_D32_SFLOAT currently
static const VkFormat SAMPLE_COLOR_FORMATS[] = {
    VK_FORMAT_R16G16B16A16_SFLOAT,  // Color
    VK_FORMAT_R16G16_SFLOAT         // Motion vectors (must be in format VK_FORMAT_R16G16_SFLOAT or
                                    // VK_FORMAT_R32G32_SFLOAT)
};

#define USE_D3D_CLIP_SPACE 0

class StreamlineSample : public nvvkhl::AppBaseVk {
public:
    StreamlineSample()           = default;
    ~StreamlineSample() override = default;

    void create(const nvvkhl::AppBaseVkCreateInfo& info,
                sl::Result                         dlssSupported,
                sl::Result                         dlssgSupported)
    {
        AppBaseVk::create(info);

        m_dlssSupported  = dlssSupported;
        m_dlssgSupported = dlssgSupported;

        // This sample only operates with a single viewport, so create a handle for viewport index
        // zero
        m_viewportHandle = sl::ViewportHandle(0);

        // Limit to 180 FPS by default
        m_reflexOptions.frameLimitUs = 5555;

        // This sample does simulation and rendering in a single thread, so can't use markers to
        // optimize
        m_reflexOptions.useMarkersToOptimize = false;

        // Set Streamline default options
        slDLSSSetOptions(m_viewportHandle, m_dlssOptions);
        slDLSSGSetOptions(m_viewportHandle, m_dlssgOptions);
        slReflexSetOptions(m_reflexOptions);

        m_dset  = std::make_unique<nvvk::DescriptorSetContainer>(info.device);
        m_alloc = std::make_unique<nvvk::ResourceAllocatorVma>(info.instance, info.device,
                                                               info.physicalDevice);

        const VkSamplerCreateInfo samplerCreateInfo = nvvk::makeSamplerCreateInfo();
        m_defaultSampler                            = m_alloc->acquireSampler(samplerCreateInfo);

        createScene();
        createPipelines();
        createImages(getSize());
    }

    void destroy() override
    {
        vkDeviceWaitIdle(m_device);

        destroyImages();
        destroyPipelines();
        destroyScene();

        m_alloc->releaseSampler(m_defaultSampler);
        m_defaultSampler = VK_NULL_HANDLE;

        m_dset.reset();
        m_alloc.reset();

        AppBaseVk::destroy();
    }

    /**
     * @brief onResize
     *
     * @note 交换链的重建由父类onFramebufferSize负责
     * onFramebufferSize函数内会调用虚函数onResize，它基本上是onFramebufferSize的后处理函数
     */
    void onResize(int width, int height) override
    {
        createImages(VkExtent2D{static_cast<uint32_t>(width), static_cast<uint32_t>(height)});
    }

    void prepareFrame() override
    {
        AppBaseVk::prepareFrame();

        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        drawGui();
        ImGui::Render();
        // REVIEW - 这里为什么突然用到imgui，整个流程中没有看到imgui的初始化
        // 有没有可能父类nvvkhl::AppBaseVk已经初始化过了。

        m_animationTimePrev = m_animationTime;
        m_animationTime += ImGui::GetIO().DeltaTime;
    }

    void renderFrame(VkCommandBuffer cmd, sl::FrameToken* frame)
    {
        const VkExtent2D renderSize  = m_gBuffers->getSize();  // BGuffer中图像的大小
        const VkExtent2D outputSize  = getSize();              // 窗口大小
        const float      aspectRatio = m_gBuffers->getAspectRatio();
        const float      scalingRatio =
            static_cast<float>(outputSize.width) /
            static_cast<float>(renderSize.width);  // 窗口大小和图像大小的比例

        // Get view information from camera
        const glm::mat4 viewMatrix = CameraManip.getMatrix();      // WS->CameraSapce
        const glm::vec2 clipPlanes = CameraManip.getClipPlanes();  // 近平面和远平面
        glm::mat4       projMatrix = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()),
                                                           aspectRatio, clipPlanes.x, clipPlanes.y);
#if !USE_D3D_CLIP_SPACE
        projMatrix[1][1] *= -1;  // Flip the Y axis to convert from D3D to Vulkan:
#endif

        glm::vec3 geye, gcenter, gup;
        CameraManip.getLookat(geye, gcenter, gup);
        glm::vec3 eye     = geye;
        glm::vec3 center  = gcenter;
        glm::vec3 up      = gup;
        glm::vec3 right   = glm::normalize(glm::cross(center - eye, up));
        glm::vec3 forward = glm::normalize(center - eye);

        // Calculate pixel jitter offset using Halton sequence
        glm::vec2 jitterOffset = {};  // 抖动抗锯齿

        if (m_dlssOptions.mode != sl::DLSSMode::eOff)
        {
            const std::function<float(uint32_t, uint32_t)> halton = [](uint32_t index,
                                                                       uint32_t base) -> float {
                float f = 1.0f;
                float r = 0.0f;
                while (index > 0)
                {
                    f *= static_cast<float>(base);
                    r += static_cast<float>(index % base) / f;
                    index /= base;
                }
                return r;
            };
            const auto jitterPhases = static_cast<uint32_t>(8 * scalingRatio * scalingRatio + 0.5f);
            const uint32_t jitterCurrentIndex = (*frame % jitterPhases) + 1;
            jitterOffset.x                    = halton(jitterCurrentIndex, 2) - 0.5f;
            jitterOffset.y                    = halton(jitterCurrentIndex, 3) - 0.5f;
        }

        // Update Streamline constants
        {
            const std::function<sl::float4x4(const glm::mat4&)> matrixToSL =
                [](const glm::mat4& m) -> sl::float4x4 {
                // Streamline expects row-major matrices
                sl::float4x4 res;
                res.row[0].x = m[0][0];
                res.row[0].y = m[1][0];  // 转置
                res.row[0].z = m[2][0];
                res.row[0].w = m[3][0];
                res.row[1].x = m[0][1];
                res.row[1].y = m[1][1];
                res.row[1].z = m[2][1];
                res.row[1].w = m[3][1];
                res.row[2].x = m[0][2];
                res.row[2].y = m[1][2];
                res.row[2].z = m[2][2];
                res.row[2].w = m[3][2];
                res.row[3].x = m[0][3];
                res.row[3].y = m[1][3];
                res.row[3].z = m[2][3];
                res.row[3].w = m[3][3];
                return res;
            };

            sl::Constants constants;
            constants.jitterOffset = sl::float2(jitterOffset.x, jitterOffset.y);
            constants.mvecScale    = sl::float2(1.0f, 1.0f);  // 运动向量的缩放比例
            // getMotion in scene.frag calculates normalized motion
            // vectors already, so do not need to scale

            glm::mat4 projMatrixD3D     = projMatrix;
            glm::mat4 projMatrixPrevD3D = m_projMatrixPrev;
#if USE_D3D_CLIP_SPACE
            constants.mvecScale.y *= -1.0f;  // getMotion in scene.frag does not flip Y coordinate
                                             // for D3D clip space, so do it via scaling
#else
            // Streamline expects clip space as defined in Direct3D (X left->right, Y bottom->top, Z
            // front->back) But clip space in Vulkan is upside down (X left->right, Y top->bottom, Z
            // front-back), so need to adjust projection matrices accordingly
            projMatrixD3D[1][1] *= -1.0f;
            projMatrixPrevD3D[1][1] *= -1.0f;
#endif
            const glm::mat4 reprojectionMatrix = projMatrixPrevD3D *
                                                 (m_viewMatrixPrev * glm::inverse(viewMatrix)) *
                                                 glm::inverse(projMatrixD3D);
            // NOTE - 第n-1帧的NDC坐标 = reprojectionMatrix * 第n帧的NDC坐标

            // camera space & clip space
            constants.cameraViewToClip = matrixToSL(projMatrix);
            constants.clipToCameraView = matrixToSL(glm::inverse(projMatrix));
            constants.clipToLensClip   = matrixToSL(glm::mat4(1));
            constants.clipToPrevClip   = matrixToSL(reprojectionMatrix);
            constants.prevClipToClip   = matrixToSL(glm::inverse(reprojectionMatrix));

            // camera
            constants.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
            constants.cameraPos           = sl::float3(eye.x, eye.y, eye.z);
            constants.cameraUp            = sl::float3(up.x, up.y, up.z);
            constants.cameraRight         = sl::float3(right.x, right.y, right.z);
            constants.cameraFwd           = sl::float3(forward.x, forward.y, forward.z);
            constants.cameraNear          = clipPlanes.x;
            constants.cameraFar           = clipPlanes.y;
            constants.cameraFOV           = glm::radians(CameraManip.getFov());
            constants.cameraAspectRatio   = aspectRatio;

            constants.depthInverted =
                (projMatrix[2][2] == 0.0f) ? sl::Boolean::eTrue : sl::Boolean::eFalse;
            // depthInverted：z深度是否离相机越近越大
            constants.cameraMotionIncluded  = sl::Boolean::eTrue;
            constants.motionVectors3D       = sl::Boolean::eFalse;
            constants.reset                 = sl::Boolean::eFalse;
            constants.motionVectorsJittered = sl::Boolean::eFalse;

            if (SL_FAILED(res, slSetConstants(constants, *frame, m_viewportHandle)))
            {
                LOGE("Streamline: Failed to set constants (%d)\n", res);
                return;
            }
        }

        // Update frame info uniform buffer
        {
            FrameInfo frameInfo;
            frameInfo.viewProj     = projMatrix * viewMatrix;
            frameInfo.viewProjPrev = m_projMatrixPrev * m_viewMatrixPrev;
            frameInfo.jitterOffset =
                jitterOffset * 2.0f / glm::vec2(renderSize.width, renderSize.height);
            frameInfo.camPos = eye;
            vkCmdUpdateBuffer(cmd, m_frameInfo.buffer, 0, sizeof(frameInfo), &frameInfo);

            VkBufferMemoryBarrier barrier{
                .sType         = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
                .pNext         = nullptr,
                .srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT,
                .dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT,
                .buffer        = m_frameInfo.buffer,
                .offset        = 0,
                .size          = VK_WHOLE_SIZE,
            };
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                 VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                                     VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                                 0, 0, nullptr, 1, &barrier, 0, nullptr);
        }

        // Scene rendering
        {
            std::vector<VkImageView> colorImageViews;
            for (uint32_t i = 0; i < std::size(SAMPLE_COLOR_FORMATS); i++)
            {
                colorImageViews.push_back(m_gBuffers->getColorImageView(i));
                nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getColorImage(i),
                                            VK_IMAGE_LAYOUT_UNDEFINED,
                                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            }
            nvvk::cmdBarrierImageLayout(cmd, m_gBuffers->getDepthImage(), VK_IMAGE_LAYOUT_UNDEFINED,
                                        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                                        VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT);

            nvvk::createRenderingInfo renderingInfo(
                VkRect2D{
                    .offset = {0, 0},
                    .extent = renderSize,
                },
                colorImageViews, m_gBuffers->getDepthImageView(), VK_ATTACHMENT_LOAD_OP_CLEAR,
                VK_ATTACHMENT_LOAD_OP_CLEAR);
            renderingInfo.colorAttachments[0].clearValue = {1.0f, 0.0f, 1.0f, 1.0f};  // Color
            renderingInfo.colorAttachments[1].clearValue = {0.0f, 0.0f, 0.0f,
                                                            0.0f};  // Motion vectors
            renderingInfo.pStencilAttachment             = nullptr;
            vkCmdBeginRendering(cmd, &renderingInfo);
            // NOTE - vkCmdBeginRendering可以设置渲染期间用到的Attachment
            // 好像不需要renderpass和帧缓存，单pass应该可以这么搞，多pass的话老老实实用renderpass和帧缓存吧

#if USE_D3D_CLIP_SPACE
            // Flip viewport to simulate D3D clip space (this is supported since the
            // VK_KHR_maintenance1 extension)
            const VkViewport viewport{
                .x        = 0,
                .y        = static_cast<float>(renderSize.height),
                .width    = static_cast<float>(renderSize.width),
                .height   = static_cast<float>(renderSize.height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            };
#else
            const VkViewport viewport{
                .x        = 0,
                .y        = 0,
                .width    = static_cast<float>(renderSize.width),
                .height   = static_cast<float>(renderSize.height),
                .minDepth = 0.0f,
                .maxDepth = 1.0f,
            };
#endif
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            const VkRect2D scissor{
                .offset = {0, 0},
                .extent = {renderSize.width, renderSize.height},
            };
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_scenePipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dset->getPipeLayout(),
                                    0, m_dset->getSetsCount(), m_dset->getSets(), 0, nullptr);
            // NOTE - 图形管线的资源有三：1. 附着Attachment（帧缓存）；2. 描述符集和pushConstant；
            // 4. 顶点输入和输入顶点组装

            for (const Node& node : m_sceneNodes)
            {
                NodeInfo nodeInfo;
                nodeInfo.model = nodeInfo.modelPrev = node.localMatrix();
                if (node.motion)
                {
                    nodeInfo.model *= glm::translate(
                        glm::mat4(1.f),
                        {-sinf(fmodf(m_animationTime, glm::two_pi<float>())), 0.0f, 0.0f});
                    nodeInfo.modelPrev *= glm::translate(
                        glm::mat4(1.f),
                        {-sinf(fmodf(m_animationTimePrev, glm::two_pi<float>())), 0.0f, 0.0f});
                    // 这里设置了物体的运动
                }
                nodeInfo.color = glm::vec3(sin(glm::vec3(1.33333f, 2.33333f, 3.33333f) *
                                               static_cast<float>(node.material)) *
                                               0.5f +
                                           0.5f);
                vkCmdPushConstants(cmd, m_dset->getPipeLayout(),
                                   VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                                   sizeof(nodeInfo), &nodeInfo);
                // NOTE - FrameInfo作为UniformBuffer设置进管线中，作为描述符集
                // 而NodeInfo作为PushContent设置进管线中

                const VkDeviceSize offsets = {0};
                vkCmdBindVertexBuffers(cmd, 0, 1, &m_sceneMeshes[node.mesh].verticesBuffer.buffer,
                                       &offsets);
                vkCmdBindIndexBuffer(cmd, m_sceneMeshes[node.mesh].trianglesBuffer.buffer, 0,
                                     VK_INDEX_TYPE_UINT32);
                vkCmdDrawIndexed(
                    cmd, static_cast<uint32_t>(m_sceneMeshes[node.mesh].triangles.size() * 3), 1, 0,
                    0, 0);
            }

            vkCmdEndRendering(cmd);
        }

        // Tag current state of resources needed for DLSS Super Resolution
        {
            sl::Resource colorInputResource(sl::ResourceType::eTex2d, m_gBuffers->getColorImage(0),
                                            nullptr, m_gBuffers->getColorImageView(0),
                                            VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            colorInputResource.width        = renderSize.width;
            colorInputResource.height       = renderSize.height;
            colorInputResource.nativeFormat = m_gBuffers->getColorFormat(0);
            colorInputResource.mipLevels    = 1;
            colorInputResource.arrayLayers  = 1;
            colorInputResource.usage =
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT;  // See nvvkhl::GBuffer::create

            sl::Resource colorOutputResource(sl::ResourceType::eTex2d, m_outputImage.image, nullptr,
                                             m_outputImageView,
                                             VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            colorOutputResource.width        = outputSize.width;
            colorOutputResource.height       = outputSize.height;
            colorOutputResource.nativeFormat = SAMPLE_COLOR_FORMATS[0];
            colorOutputResource.mipLevels    = 1;
            colorOutputResource.arrayLayers  = 1;
            colorOutputResource.usage =
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;  // See createImages below

            sl::Resource depthResource(sl::ResourceType::eTex2d, m_gBuffers->getDepthImage(),
                                       nullptr, m_gBuffers->getDepthImageView(),
                                       VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
            depthResource.width        = renderSize.width;
            depthResource.height       = renderSize.height;
            depthResource.nativeFormat = m_gBuffers->getDepthFormat();
            depthResource.mipLevels    = 1;
            depthResource.arrayLayers  = 1;
            depthResource.usage        = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                  VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                  VK_IMAGE_USAGE_TRANSFER_DST_BIT;  // See nvvkhl::GBuffer::create

            sl::Resource motionVectorsResource(
                sl::ResourceType::eTex2d, m_gBuffers->getColorImage(1), nullptr,
                m_gBuffers->getColorImageView(1), VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
            motionVectorsResource.width        = renderSize.width;
            motionVectorsResource.height       = renderSize.height;
            motionVectorsResource.nativeFormat = m_gBuffers->getColorFormat(1);
            motionVectorsResource.mipLevels    = 1;
            motionVectorsResource.arrayLayers  = 1;
            motionVectorsResource.usage =
                VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                VK_IMAGE_USAGE_TRANSFER_DST_BIT;  // See nvvkhl::GBuffer::create

            const sl::ResourceTag tags[] = {
                sl::ResourceTag(&colorInputResource, sl::kBufferTypeScalingInputColor,
                                sl::ResourceLifecycle::eValidUntilPresent),
                sl::ResourceTag(&colorOutputResource, sl::kBufferTypeScalingOutputColor,
                                sl::ResourceLifecycle::eValidUntilPresent),
                sl::ResourceTag(&depthResource, sl::kBufferTypeDepth,
                                sl::ResourceLifecycle::eValidUntilPresent),
                sl::ResourceTag(&motionVectorsResource, sl::kBufferTypeMotionVectors,
                                sl::ResourceLifecycle::eValidUntilPresent),
            };

            slSetTag(m_viewportHandle, tags, static_cast<uint32_t>(std::size(tags)), cmd);
        }

        if (m_dlssOptions.mode != sl::DLSSMode::eOff)
        {
            nvvk::cmdBarrierImageLayout(cmd, m_outputImage.image, VK_IMAGE_LAYOUT_UNDEFINED,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            const sl::BaseStructure* inputs[] = {&m_viewportHandle};

            if (SL_FAILED(res, slEvaluateFeature(sl::kFeatureDLSS, *frame, inputs,
                                                 static_cast<uint32_t>(std::size(inputs)), cmd)))
            {
                LOGE("Streamline: Failed to evaluate DLSS Super Resolution (%d)\n", res);
            }
        }
        else
        {
            nvvk::cmdBarrierImageLayout(cmd, m_outputImage.image,
                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

            // When DLSS Super Resolution is off, 'm_outputImage' already points to
            // 'm_gBuffers->getColorImage()', so no additional processing needed here
        }

        // Post-processing
        {
            nvvk::cmdBarrierImageLayout(cmd, m_swapChain.getActiveImage(),
                                        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                                        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);

            const nvvk::createRenderingInfo renderingInfo(
                {{0, 0}, getSize()}, {m_swapChain.getActiveImageView()}, VK_NULL_HANDLE,
                VK_ATTACHMENT_LOAD_OP_CLEAR, VK_ATTACHMENT_LOAD_OP_DONT_CARE);
            vkCmdBeginRendering(cmd, &renderingInfo);

            const VkViewport viewport = {0.0f,
                                         0.0f,
                                         static_cast<float>(outputSize.width),
                                         static_cast<float>(outputSize.height),
                                         0.0f,
                                         1.0f};
            vkCmdSetViewport(cmd, 0, 1, &viewport);
            const VkRect2D scissor = {{0, 0}, {outputSize.width, outputSize.height}};
            vkCmdSetScissor(cmd, 0, 1, &scissor);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_postPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_dset->getPipeLayout(),
                                    0, m_dset->getSetsCount(), m_dset->getSets(), 0, nullptr);

            vkCmdDraw(cmd, 3, 1, 0, 0);

            vkCmdEndRendering(cmd);
        }

        // Tag current state of additional resources needed for DLSS Frame Generation
        {
            sl::Resource colorResource(sl::ResourceType::eTex2d, m_outputImage.image, nullptr,
                                       m_outputImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            colorResource.width        = outputSize.width;
            colorResource.height       = outputSize.height;
            colorResource.nativeFormat = SAMPLE_COLOR_FORMATS[0];
            colorResource.mipLevels    = 1;
            colorResource.arrayLayers  = 1;
            colorResource.usage =
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;  // See createImages below

            const sl::ResourceTag tags[] = {
                sl::ResourceTag(&colorResource, sl::kBufferTypeHUDLessColor,
                                sl::ResourceLifecycle::eValidUntilPresent),
            };

            slSetTag(m_viewportHandle, tags, static_cast<uint32_t>(std::size(tags)), cmd);
        }

        // ImGui rendering
        if (ImDrawData* const drawData = ImGui::GetDrawData();
            drawData != nullptr && drawData->TotalVtxCount != 0 && drawData->TotalIdxCount != 0)
        {
            const nvvk::createRenderingInfo renderingInfo(
                {{0, 0}, getSize()}, {m_swapChain.getActiveImageView()}, VK_NULL_HANDLE,
                VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_LOAD_OP_DONT_CARE);
            vkCmdBeginRendering(cmd, &renderingInfo);

            ImGui_ImplVulkan_RenderDrawData(drawData, cmd);

            vkCmdEndRendering(cmd);
        }

        nvvk::cmdBarrierImageLayout(cmd, m_swapChain.getActiveImage(),
                                    VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);

        // Update previous matrices for next frame
        m_projMatrixPrev = projMatrix;
        m_viewMatrixPrev = viewMatrix;
    }

    void drawGui()
    {
        ImGui::SetNextWindowSize(ImVec2(200, 500), ImGuiCond_FirstUseEver);
        ImGui::Begin("Settings");

        // The framerate calculated by the application does not account for the additional present
        // calls introduced by DLSS Frame Generation, so need to adjust it accordingly
        uint32_t numFramesActuallyPresented = 1;
        if (m_dlssgSupported == sl::Result::eOk)
        {
            sl::DLSSGState dlssgState;
            if (SL_FAILED(res, slDLSSGGetState(m_viewportHandle, dlssgState,
                                               &m_dlssgOptions)))  // Only call this once per frame!
            {
                LOGE("Streamline: Failed to get DLSS Frame Generation state (%d)\n", res);
            }
            else if (dlssgState.status != sl::DLSSGStatus::eOk)
            {
                LOGE("Streamline: DLSS Frame Generation status is %u\n", dlssgState.status);

                // Turn off DLSS Frame Generation if something went wrong
                m_dlssgOptions.mode = sl::DLSSGMode::eOff;
                slDLSSGSetOptions(m_viewportHandle, m_dlssgOptions);
            }
            numFramesActuallyPresented = max(dlssgState.numFramesActuallyPresented, 1U);
        }
        ImGui::Text("FPS: %f\n",
                    static_cast<double>(ImGui::GetIO().Framerate * numFramesActuallyPresented));
        // NOTE - ImGui::GetIO().Framerate是标准的一帧，而DLSS技术能在一帧内生成多个额外帧

        if (ImGui::CollapsingHeader("Reflex"))
        {
            ImGui::PushID("Reflex");

            bool modified = false;

            modified |= ImGui::Combo("Mode", reinterpret_cast<int*>(&m_reflexOptions.mode),
                                     "Off\0Low Latency\0Low Latency with Boost\0");

            if (int fps = m_reflexOptions.frameLimitUs != 0 ?
                              static_cast<int>((1000.0f / m_reflexOptions.frameLimitUs) * 1000.0f) :
                              0;
                ImGui::InputInt("FPS Limit", &fps, 1, 100, ImGuiInputTextFlags_EnterReturnsTrue))
            {
                // Clamp possible FPS range
                fps = std::clamp(fps, 1, 200);

                m_reflexOptions.frameLimitUs = static_cast<uint32_t>((1000.0f / fps) * 1000.0f);
                modified                     = true;
            }

            if (modified)
            {
                if (SL_FAILED(res, slReflexSetOptions(m_reflexOptions)))
                {
                    LOGE("Streamline: Failed to set Reflex options (%d)\n", res);
                }
            }

            ImGui::PopID();
        }

        if (ImGui::CollapsingHeader("DLSS Super Resolution"))
        {
            ImGui::PushID("DLSS Super Resolution");

            if (m_dlssSupported == sl::Result::eOk)
            {
                bool modified = false;

                modified |= ImGui::Combo("Mode", reinterpret_cast<int*>(&m_dlssOptions.mode),
                                         "Off\0Performance\0Balanced\0Quality\0Ultra "
                                         "Performance\0Ultra Quality\0DLAA\0");
                modified |= ImGui::SliderFloat("Sharpness", &m_dlssOptions.sharpness, 0.0f, 1.0f);

                if (modified) { createImages(getSize()); }
            }
            else
            {
                ImGui::TextWrapped(
                    "DLSS Super Resolution is not supported on this adapter (return code %d)",
                    m_dlssSupported);
            }

            ImGui::PopID();
        }

        if (ImGui::CollapsingHeader("DLSS Frame Generation"))
        {
            ImGui::PushID("DLSS Frame Generation");

            if (m_dlssgSupported == sl::Result::eOk)
            {
                bool modified = false;
                modified |=
                    ImGui::Checkbox("Enabled", reinterpret_cast<bool*>(&m_dlssgOptions.mode));

                if (modified)
                {
                    // DLSS Frame Generation requires Reflex to be on
                    if (m_dlssgOptions.mode != sl::DLSSGMode::eOff &&
                        m_reflexOptions.mode == sl::ReflexMode::eOff)
                    {
                        m_reflexOptions.mode = sl::ReflexMode::eLowLatency;
                        slReflexSetOptions(m_reflexOptions);
                    }

                    if (SL_FAILED(res, slDLSSGSetOptions(m_viewportHandle, m_dlssgOptions)))
                    {
                        LOGE("Streamline: Failed to set DLSS Frame Generation options (%d)\n", res);
                    }
                }
            }
            else if (m_dlssgSupported == sl::Result::eErrorOSDisabledHWS)
            {
                ImGui::TextWrapped("DLSS Frame Generation is not supported because "
                                   "hardware-accelerated GPU scheduling is not enabled: "
                                   "https://devblogs.microsoft.com/directx/"
                                   "hardware-accelerated-gpu-scheduling/");
            }
            else
            {
                ImGui::TextWrapped(
                    "DLSS Frame Generation is not supported on this adapter (return code %d)",
                    m_dlssgSupported);
            }

            ImGui::PopID();
        }

        ImGui::End();
    }

private:
    sl::Result m_dlssSupported  = sl::Result::eOk;
    sl::Result m_dlssgSupported = sl::Result::eOk;

    sl::DLSSOptions   m_dlssOptions;
    sl::DLSSGOptions  m_dlssgOptions;
    sl::ReflexOptions m_reflexOptions;

    sl::ViewportHandle m_viewportHandle;

    std::unique_ptr<nvvk::DescriptorSetContainer> m_dset;
    std::unique_ptr<nvvk::ResourceAllocator>      m_alloc;
    VkSampler                                     m_defaultSampler = VK_NULL_HANDLE;

    struct PrimitiveMeshVk : nvh::PrimitiveMesh
    {
        explicit PrimitiveMeshVk(const nvh::PrimitiveMesh& mesh) : nvh::PrimitiveMesh(mesh) {}

        nvvk::Buffer verticesBuffer;
        nvvk::Buffer trianglesBuffer;
    };
    std::vector<PrimitiveMeshVk> m_sceneMeshes;

    struct Node : nvh::Node
    {
        bool motion = false;
    };
    std::vector<Node> m_sceneNodes;

    nvvk::Buffer m_frameInfo;  // Pipeline Push Content

    VkPipeline m_scenePipeline = VK_NULL_HANDLE;
    VkPipeline m_postPipeline  = VK_NULL_HANDLE;

    std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;
    nvvk::Image                      m_outputImage;
    VkImageView                      m_outputImageView = VK_NULL_HANDLE;

    float     m_animationTime     = 0.0F;
    float     m_animationTimePrev = 0.0F;
    glm::mat4 m_projMatrixPrev;
    glm::mat4 m_viewMatrixPrev;

    void createScene()
    {
        // Create meshes
        m_sceneMeshes.emplace_back(nvh::createSphereUv());
        m_sceneMeshes.emplace_back(nvh::createCube());
        m_sceneMeshes.emplace_back(nvh::createTetrahedron());
        m_sceneMeshes.emplace_back(nvh::createOctahedron());
        m_sceneMeshes.emplace_back(nvh::createIcosahedron());
        m_sceneMeshes.emplace_back(nvh::createConeMesh());
        const int num_meshes = static_cast<int>(m_sceneMeshes.size());

        const VkCommandBuffer cmd = createTempCmdBuffer();
        for (int i = 0; i < num_meshes; i++)
        {
            PrimitiveMeshVk& mesh = m_sceneMeshes[i];
            mesh.verticesBuffer =
                m_alloc->createBuffer(cmd, mesh.vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
            mesh.trianglesBuffer =
                m_alloc->createBuffer(cmd, mesh.triangles, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
        }
        m_frameInfo = m_alloc->createBuffer(sizeof(FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        submitTempCmdBuffer(cmd);

        // Create instances/nodes
        for (int i = 0; i < num_meshes; i++)
        {
            Node& node       = m_sceneNodes.emplace_back();
            node.mesh        = i;
            node.material    = i;
            node.translation = glm::vec3(
                -(static_cast<float>(num_meshes) * 0.5f) + static_cast<float>(i), 0.0f, 0.0f);
            node.motion = true;
        }
        Node& background       = m_sceneNodes.emplace_back();
        background.mesh        = 1;
        background.translation = {0.0f, 0.0f, -5.0f};
        background.scale       = {50, 50, 50};

        CameraManip.setClipPlanes({0.1f, 100.0f});
        CameraManip.setLookat({0.0f, 0.0f, 5.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f});
    }

    void createPipelines()
    {
        // init descriptor set
        m_dset->addBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        m_dset->addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);
        m_dset->initLayout();
        m_dset->initPool(1);
        const VkPushConstantRange pushConstantRange{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset     = 0,
            .size       = sizeof(NodeInfo),
        };
        m_dset->initPipeLayout(1, &pushConstantRange);

        // Update descriptor set
        {
            VkDescriptorBufferInfo frameInfoBufferInfo{
                .buffer = m_frameInfo.buffer,
                .offset = 0,
                .range  = VK_WHOLE_SIZE,
            };
            VkWriteDescriptorSet frameInfoWrite = m_dset->makeWrite(0, 0, &frameInfoBufferInfo);
            vkUpdateDescriptorSets(m_device, 1, &frameInfoWrite, 0, nullptr);
        }

        // Create the scene rendering pipeline
        {
            // pipelineState
            nvvk::GraphicsPipelineState pipelineState{};
            pipelineState.rasterizationState.cullMode = VK_CULL_MODE_NONE;
            pipelineState.addBindingDescriptions({
                VkVertexInputBindingDescription{
                    .binding   = 0,
                    .stride    = sizeof(nvh::PrimitiveVertex),
                    .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
                },
            });
            pipelineState.addAttributeDescriptions({
                VkVertexInputAttributeDescription{
                    .location = 0,
                    .binding  = 0,
                    .format   = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset   = offsetof(nvh::PrimitiveVertex, p),
                },
                VkVertexInputAttributeDescription{
                    .location = 0,
                    .binding  = 0,
                    .format   = VK_FORMAT_R32G32B32_SFLOAT,
                    .offset   = offsetof(nvh::PrimitiveVertex, n),
                },
            });
            pipelineState.clearBlendAttachmentStates();
            for (uint32_t i = 0; i < std::size(SAMPLE_COLOR_FORMATS); i++)
            {
                pipelineState.addBlendAttachmentState(
                    nvvk::GraphicsPipelineState::makePipelineColorBlendAttachmentState());
                // 这里加入了两个ColorBlendAttachment
            }

            // renderingInfo
            VkPipelineRenderingCreateInfo renderingInfo{
                .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                .pNext                   = nullptr,
                .colorAttachmentCount    = std::size(SAMPLE_COLOR_FORMATS),
                .pColorAttachmentFormats = &SAMPLE_COLOR_FORMATS[0],
                .depthAttachmentFormat   = SAMPLE_DEPTH_FORMAT,
            };

            nvvk::GraphicsPipelineGenerator pipelineGenerator(m_device, m_dset->getPipeLayout(),
                                                              renderingInfo, pipelineState);
            pipelineGenerator.addShader(
                std::vector<uint32_t>{std::begin(scene_vert), std::end(scene_vert)},
                VK_SHADER_STAGE_VERTEX_BIT);
            pipelineGenerator.addShader(
                std::vector<uint32_t>{std::begin(scene_frag), std::end(scene_frag)},
                VK_SHADER_STAGE_FRAGMENT_BIT);

            m_scenePipeline = pipelineGenerator.createPipeline();
            // NOTE - 这个pipelineGenerator好像不能指定subpass
            // 它renderpass都有，为什么不能指定subpass？
        }

        // Create the post-processing pipeline
        {
            nvvk::GraphicsPipelineState pipelineState;

            const VkFormat                colorFormat = m_swapChain.getFormat();
            VkPipelineRenderingCreateInfo renderingInfo{
                .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                .pNext                   = nullptr,
                .colorAttachmentCount    = 1,
                .pColorAttachmentFormats = &colorFormat,
            };

            nvvk::GraphicsPipelineGenerator pipelineGenerator(m_device, m_dset->getPipeLayout(),
                                                              renderingInfo, pipelineState);
            pipelineGenerator.addShader(
                std::vector<uint32_t>{std::begin(post_vert), std::end(post_vert)},
                VK_SHADER_STAGE_VERTEX_BIT);
            pipelineGenerator.addShader(
                std::vector<uint32_t>{std::begin(post_frag), std::end(post_frag)},
                VK_SHADER_STAGE_FRAGMENT_BIT);

            m_postPipeline = pipelineGenerator.createPipeline();
        }
    }

    void createImages(const VkExtent2D& size)
    {
        vkDeviceWaitIdle(m_device);

        destroyImages();

        // Update output dimensions for DLSS Super Resolution
        m_dlssOptions.outputWidth  = size.width;
        m_dlssOptions.outputHeight = size.height;
        m_dlssOptions.colorBuffersHDR =
            (SAMPLE_COLOR_FORMATS[0] >= VK_FORMAT_A2R10G10B10_UNORM_PACK32) ? sl::Boolean::eTrue :
                                                                              sl::Boolean::eFalse;
        if (SL_FAILED(res, slDLSSSetOptions(m_viewportHandle, m_dlssOptions)))
        {
            LOGE("Streamline: Failed to set DLSS Super Resolution options (%d)\n", res);
        }

        sl::DLSSOptimalSettings dlssSettings;
        if (SL_FAILED(res, slDLSSGetOptimalSettings(m_dlssOptions, dlssSettings)) ||
            m_dlssOptions.mode == sl::DLSSMode::eOff || dlssSettings.optimalRenderWidth == 0 ||
            dlssSettings.optimalRenderHeight == 0)
        {
            // Default render dimensions to the window dimensions if DLSS is not enabled
            dlssSettings.optimalRenderWidth  = size.width;
            dlssSettings.optimalRenderHeight = size.height;
        }

        const VkExtent2D renderSize{dlssSettings.optimalRenderWidth,
                                    dlssSettings.optimalRenderHeight};

        m_gBuffers = std::make_unique<nvvkhl::GBuffer>(
            m_device, m_alloc.get(), renderSize,
            std::vector<VkFormat>(std::begin(SAMPLE_COLOR_FORMATS), std::end(SAMPLE_COLOR_FORMATS)),
            SAMPLE_DEPTH_FORMAT);

        if (m_dlssOptions.mode != sl::DLSSMode::eOff)
        {
            // The image DLSS Super Resolution outputs to must have the VK_IMAGE_USAGE_STORAGE_BIT
            // usage flag, or else nothing will be rendered to it!
            const VkImageCreateInfo outputCreateInfo = nvvk::makeImage2DCreateInfo(
                renderSize, SAMPLE_COLOR_FORMATS[0],
                VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
            m_outputImage = m_alloc->createImage(outputCreateInfo);

            const VkImageViewCreateInfo outputViewCreateInfo =
                nvvk::makeImage2DViewCreateInfo(m_outputImage.image, outputCreateInfo.format);
            NVVK_CHECK(
                vkCreateImageView(m_device, &outputViewCreateInfo, nullptr, &m_outputImageView));
        }
        else
        {
            m_outputImage.image = m_gBuffers->getColorImage();
            m_outputImageView   = m_gBuffers->getColorImageView();
        }

        // Update descriptor set with the new output image
        {
            const VkDescriptorImageInfo outputImageInfo{
                .sampler     = m_defaultSampler,
                .imageView   = m_outputImageView,
                .imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            };
            const VkWriteDescriptorSet outputImageWrite = m_dset->makeWrite(0, 1, &outputImageInfo);
            vkUpdateDescriptorSets(m_device, 1, &outputImageWrite, 0, nullptr);
        }
    }

    void destroyScene()
    {
        for (PrimitiveMeshVk& mesh : m_sceneMeshes)
        {
            m_alloc->destroy(mesh.trianglesBuffer);
            m_alloc->destroy(mesh.verticesBuffer);
        }

        m_sceneNodes.clear();
        m_sceneMeshes.clear();

        m_alloc->destroy(m_frameInfo);
    }

    void destroyPipelines()
    {
        vkDestroyPipeline(m_device, m_scenePipeline, nullptr);
        m_scenePipeline = VK_NULL_HANDLE;

        vkDestroyPipeline(m_device, m_postPipeline, nullptr);
        m_postPipeline = VK_NULL_HANDLE;
    }

    void destroyImages()
    {
        if (m_outputImage.memHandle != nullptr)
        {
            vkDestroyImageView(m_device, m_outputImageView, nullptr);
            m_alloc->destroy(m_outputImage);
            // NOTE - 两种情况
            // 不拿GBuffer的时候，把自己创建的东西删掉
            // 拿GBuffer的时候，reset GBuffer就好了
        }
        else { m_outputImage.image = VK_NULL_HANDLE; }
        m_outputImageView = VK_NULL_HANDLE;
        m_gBuffers.reset();
    }
};

/**
 * @brief Setup GLFW window
 *
 * @return GLFWwindow*
 */
GLFWwindow* initGLFW()
{
    if (glfwInit() == 0)
    {
        LOGE("GLFW: Initialization failed\n");
        throw std::runtime_error(
            "error code: " + std::to_string(static_cast<int>(sl::Result::eErrorNotInitialized)));
    }

    if (glfwVulkanSupported() == 0)
    {
        LOGE("GLFW: Vulkan not supported\n");
        throw std::runtime_error("error code: " + std::to_string(static_cast<int>(
                                                      sl::Result::eErrorAdapterNotSupported)));
    }

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // 使用Vulkan API
    GLFWwindow* const window =
        glfwCreateWindow(SAMPLE_WIDTH, SAMPLE_HEIGHT, PROJECT_NAME.c_str(), nullptr, nullptr);
    if (window == nullptr)
    {
        LOGE("GLFW: Failed to create window\n");
        throw std::runtime_error(
            "error code: " + std::to_string(static_cast<int>(sl::Result::eErrorNotInitialized)));
    }

    return window;
}

std::tuple<sl::Result, sl::Result> checkFeatureSupport(VkPhysicalDevice physicalDevice)
{
    // Verify that the requested features are actually supported on the created device
    sl::AdapterInfo adapter;
    adapter.vkPhysicalDevice = physicalDevice;
    // This sample can optionally run without DLSS, but it requires Reflex
    sl::Result dlssSupported  = sl::Result::eOk;
    sl::Result dlssgSupported = sl::Result::eOk;

    for (const sl::Feature feature : SL_FEATURES)
    {
        if (SL_FAILED(res, slIsFeatureSupported(feature, adapter)))
        {
            if (res == sl::Result::eErrorOSDisabledHWS)
            {
                LOGW("Streamline: Feature %u is not supported because hardware-accelerated GPU "
                     "scheduling is not enabled: "
                     "https://devblogs.microsoft.com/directx/"
                     "hardware-accelerated-gpu-scheduling/.\n",
                     feature);
            }
            else
            {
                LOGE("Streamline: Feature %u is not supported on this adapter (return code %d). "
                     "The log messages from Streamline may include more information.\n",
                     feature, res);
            }

            switch (feature)
            {
                case sl::kFeatureDLSS: {
                    LOGW("This sample can still run, but DLSS Super Resolution will not be "
                         "available\n");
                    dlssSupported = res;
                    break;
                }
                case sl::kFeatureDLSS_G: {
                    LOGW("This sample can still run, but DLSS Frame Generation will not be "
                         "available\n");
                    dlssgSupported = res;
                    break;
                }
                default: {
                    throw std::runtime_error("error code: " +
                                             std::to_string(static_cast<int>(res)));
                }
            }
        }
    }

    return std::make_tuple(dlssSupported, dlssgSupported);
}

/**
 * @brief Add Vulkan extensions required by Streamline features
 *
 * @param contextInfo
 * @param useVsync
 */
void addStreamlineVulkanExtensions(nvvk::ContextCreateInfo& contextInfo, bool& useVsync)
{
#if SL_MANUAL_HOOKING
    VkPhysicalDeviceOpticalFlowFeaturesNV opticalFlowFeatures = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPTICAL_FLOW_FEATURES_NV,
    };

    // Add Vulkan extensions required by Streamline features
    for (const sl::Feature feature : SL_FEATURES)
    {
        sl::FeatureRequirements requirements;
        if (SL_FAILED(res, slGetFeatureRequirements(feature, requirements)))
        {
            LOGE("Streamline: Failed to get feature requirements for feature %u (%d)\n", feature,
                 res);
            throw std::runtime_error("error code: " + std::to_string(static_cast<int>(res)));
        }
        if ((requirements.flags & sl::FeatureRequirementFlags::eVulkanSupported) == 0)
        {
            LOGE("Streamline: Feature %u is not supported on Vulkan\n", feature);
            throw std::runtime_error("error code: " + std::to_string(static_cast<int>(
                                                          sl::Result::eErrorFeatureNotSupported)));
        }

        // set vsync off if required
        if ((requirements.flags & sl::FeatureRequirementFlags::eVSyncOffRequired) != 0)
        {
            useVsync = false;
        }

        // add required queue
        if (requirements.vkNumComputeQueuesRequired != 0)
        {
            contextInfo.addRequestedQueue(VK_QUEUE_COMPUTE_BIT,
                                          requirements.vkNumComputeQueuesRequired);
            // NOTE - 这里好像只是添加了队列，并没有实际的创建队列
        }
        if (requirements.vkNumGraphicsQueuesRequired != 0)
        {
            contextInfo.addRequestedQueue(VK_QUEUE_GRAPHICS_BIT,
                                          requirements.vkNumGraphicsQueuesRequired);
        }
        if (requirements.vkNumOpticalFlowQueuesRequired != 0)
        {
            contextInfo.addRequestedQueue(VK_QUEUE_OPTICAL_FLOW_BIT_NV,
                                          requirements.vkNumOpticalFlowQueuesRequired);
        }

        // add device extensions
        for (uint32_t i = 0; i < requirements.vkNumDeviceExtensions; i++)
        {
            void* featureStruct = nullptr;
            if (strcmp(requirements.vkDeviceExtensions[i], VK_NV_OPTICAL_FLOW_EXTENSION_NAME) == 0)
            {
                featureStruct                   = &opticalFlowFeatures;
                opticalFlowFeatures.opticalFlow = VK_TRUE;
            }
            contextInfo.addDeviceExtension(requirements.vkDeviceExtensions[i], false,
                                           featureStruct);
        }

        // add instance extensions
        for (uint32_t i = 0; i < requirements.vkNumInstanceExtensions; i++)
        {
            contextInfo.addInstanceExtension(requirements.vkInstanceExtensions[i]);
        }
    }
#endif
}

/**
 * @brief 将Vulkan信息分配给Streamline
 *
 */
void assignVulkanInfoToStreamline(nvvk::Context& context)
{
#if SL_MANUAL_HOOKING
    StreamlineWrapper::get().initVulkanHooks(context.m_device);

    // assign vulkan infomation to streamline
    //  Inform Streamline about the main Vulkan device and queues
    sl::VulkanInfo vulkanInfo;
    vulkanInfo.device         = context.m_device;
    vulkanInfo.instance       = context.m_instance;
    vulkanInfo.physicalDevice = context.m_physicalDevice;

    // 创建队列
    if (const auto queueC = context.createQueue(VK_QUEUE_COMPUTE_BIT, "queueC");
        queueC.queue != VK_NULL_HANDLE)
    {
        vulkanInfo.computeQueueIndex  = queueC.queueIndex;
        vulkanInfo.computeQueueFamily = queueC.familyIndex;
    }
    if (const auto queueG = context.createQueue(VK_QUEUE_GRAPHICS_BIT, "queueG");
        queueG.queue != VK_NULL_HANDLE)
    {
        vulkanInfo.graphicsQueueIndex  = queueG.queueIndex;
        vulkanInfo.graphicsQueueFamily = queueG.familyIndex;
    }
    if (const auto queueOF = context.createQueue(VK_QUEUE_OPTICAL_FLOW_BIT_NV, "queueOF");
        queueOF.queue != VK_NULL_HANDLE)
    {
        vulkanInfo.opticalFlowQueueIndex    = queueOF.queueIndex;
        vulkanInfo.opticalFlowQueueFamily   = queueOF.familyIndex;
        vulkanInfo.useNativeOpticalFlowMode = true;
    }

    if (SL_FAILED(res, slSetVulkanInfo(vulkanInfo)))
    {
        LOGE("Streamline: Failed to set Vulkan info (%d)\n", res);
        throw std::runtime_error("error code: " + std::to_string(static_cast<int>(res)));
    }
#endif
}

/**
 * @brief Create Window Surface
 *
 */
VkSurfaceKHR createWindowSurface(nvvk::Context& context, GLFWwindow* window)
{
    VkSurfaceKHR surface = VK_NULL_HANDLE;

#if 1
    if (NVVK_CHECK(glfwCreateWindowSurface(context.m_instance, window, nullptr, &surface)))
#else
    VkWin32SurfaceCreateInfoKHR surfaceInfo = {
        .sType = VK_STRUCTURE_TYPE_WIN32_SURFACE_CREATE_INFO_KHR,
    };
    GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                       reinterpret_cast<LPCWSTR>(&glfwGetWin32Window), &surfaceInfo.hinstance);
    surfaceInfo.hwnd = glfwGetWin32Window(window);

    // Call vkCreateWin32SurfaceKHR directly so that it is routed through Streamline
    if (NVVK_CHECK(vkCreateWin32SurfaceKHR(context.m_instance, &surfaceInfo, nullptr, &surface)))
#endif
    {
        throw std::runtime_error("error code: " +
                                 std::to_string(static_cast<int>(sl::Result::eErrorVulkanAPI)));
    }

    return surface;
}

int main(int argc, char** argv)
{
    try
    {
        // Setup GLFW window
        GLFWwindow* const window = initGLFW();

        // Initialize Streamline
        // (this must happen before any Vulkan calls are made)
        sl::Preferences pref;
        pref.showConsole = true;
        pref.logLevel    = sl::LogLevel::eVerbose;
#if SL_MANUAL_HOOKING
        pref.flags |= sl::PreferenceFlags::eUseManualHooking;
#endif
        pref.featuresToLoad    = SL_FEATURES;
        pref.numFeaturesToLoad = static_cast<uint32_t>(std::size(SL_FEATURES));
        pref.applicationId     = 231313132;
        pref.engine            = sl::EngineType::eCustom;
        pref.engineVersion     = nullptr;
        pref.renderAPI         = sl::RenderAPI::eVulkan;

        if (SL_FAILED(res, slInit(pref)))
        {
            LOGE("Streamline: Initialization failed (%d)\n", res);
            return static_cast<int>(res);
        }

        nvvk::ContextCreateInfo contextInfo(/* bUseValidation = */ false);
        contextInfo.setVersion(1, 3);
        // Add Vulkan extensions required by GLFW
        {
            contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
            uint32_t                 vkNumInstanceExtensions = 0;
            const char* const* const vkInstanceExtensions =
                glfwGetRequiredInstanceExtensions(&vkNumInstanceExtensions);
            // NOTE - vkInstanceExtensions持有的资源由GLFW自动管理，你不应该自己释放它
            for (uint32_t i = 0; i < vkNumInstanceExtensions; i++)
            {
                contextInfo.addInstanceExtension(vkInstanceExtensions[i]);
            }
        }

        // Add Vulkan extensions required by Streamline features
        bool useVsync = false;
        addStreamlineVulkanExtensions(contextInfo, useVsync);

        // Create Vulkan device and queues
        nvvk::Context context;
        if (!context.init(contextInfo)) { return static_cast<int>(sl::Result::eErrorVulkanAPI); }

        // check if all the features(dlss(optional), dlssg(optional), reflex) are supported
        auto [dlssSupported, dlssgSupported] = checkFeatureSupport(context.m_physicalDevice);

        // 将Vulkan信息分配给Streamline
        assignVulkanInfoToStreamline(context);

        // Create window surface
        VkSurfaceKHR surface = createWindowSurface(context, window);

        // Create main application
        nvvkhl::AppBaseVkCreateInfo appInfo{
            .instance            = context.m_instance,
            .device              = context.m_device,
            .physicalDevice      = context.m_physicalDevice,
            .queueIndices        = {context.m_queueGCT},
            .surface             = surface,
            .size                = VkExtent2D{SAMPLE_WIDTH, SAMPLE_HEIGHT},
            .window              = window,
            .useDynamicRendering = true,
            .useVsync            = useVsync,
        };
        StreamlineSample app;
        app.create(appInfo, dlssSupported, dlssgSupported);

        // Main loop
        while (glfwWindowShouldClose(window) == 0)
        {
            if (app.isMinimized())
            {
                glfwPollEvents();
                continue;
            }

            sl::FrameToken* frame = nullptr;
            if (SL_FAILED(res, slGetNewFrameToken(frame)))
            {
                LOGE("Streamline: Failed to get new frame token (%d)\n", res);
                break;
            }

            // Perform sleep before any input is processed for optimal frame pacing
            slReflexSleep(*frame);

            // Input
            {
                // 模拟标记应该捕捉读取新的输入和所有基于这些输入更新世界所做的工作，但不包括上面的休眠
                slPCLSetMarker(sl::PCLMarker::eSimulationStart, *frame);

                glfwPollEvents();
                app.prepareFrame();

                slPCLSetMarker(sl::PCLMarker::eSimulationEnd, *frame);
                // NOTE - sl::ReflexMarker has change to sl::PCLMarker
                // and slReflexSetMarker has been changed to slPCLSetMarker
            }

            // Rendering
            {
                const uint32_t        curFrame = app.getCurFrame();
                const VkCommandBuffer cmd      = app.getCommandBuffers()[curFrame];

                slPCLSetMarker(sl::PCLMarker::eRenderSubmitStart, *frame);

                VkCommandBufferBeginInfo beginInfo = {
                    .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
                };
                vkBeginCommandBuffer(cmd, &beginInfo);
                app.renderFrame(cmd, frame);
                vkEndCommandBuffer(cmd);

                slPCLSetMarker(sl::PCLMarker::eRenderSubmitEnd, *frame);
            }

            // Presentation
            {
                // This marker is required for DLSS-G to work
                slPCLSetMarker(sl::PCLMarker::ePresentStart, *frame);

                app.submitFrame();

                slPCLSetMarker(sl::PCLMarker::ePresentEnd, *frame);
            }
        }

        // clean up
        app.destroy();

        // 在销毁 Vulkan 设备之前关闭 Streamline，以便它能正确清理资源
        slShutdown();

        vkDestroySurfaceKHR(context.m_instance, surface, nullptr);

        context.deinit();

        glfwDestroyWindow(window);

        glfwTerminate();
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

/**NOTE - Nvidia Reflex技术
主要解决系统延迟的问题(游戏延迟分为三部分:网络延迟和系统延迟)
通过Reflex技术降低系统延迟
 */

/**NOTE - Nvidia DLSS 深度学习超级采样
1. DLSS 超分辨率
2. DLSS 帧生成
 */
