/**
 * @file main.cpp
 * @author your name (you@domain.com)
 * @brief
 * @version 0.1
 * @date 2024-11-22
 *
 * @copyright Copyright (c) 2024
 *
 */

// c
#include <cassert>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

// cpp
#include <algorithm>
#include <array>
#include <chrono>
#include <exception>
#include <functional>
#include <glm/matrix.hpp>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <vulkan/vulkan_beta.h>

// 3rdparty
#define VK_ENABLE_BETA_EXTENSIONS
#define VMA_IMPLEMENTATION
#include "heightmap_rtx/include/heightmap_rtx.h"
#include "imgui/backends/imgui_impl_vulkan.h"
#include "imgui/imgui.h"
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "vma/include/vk_mem_alloc.h"
#include <glm/vec4.hpp>
#include <vulkan/vulkan_core.h>

// 3rdparty - nvvk
#include "nvh/cameramanipulator.hpp"
#include "nvh/primitives.hpp"
#include "nvvk/acceleration_structures.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/error_vk.hpp"
#include "nvvk/images_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/sbtwrapper_vk.hpp"
#include "nvvk/shaders_vk.hpp"

// 3rdparty - nvvkhl
#include "nvvkhl/alloc_vma.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_testing.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/pipeline_container.hpp"
#include "nvvkhl/shaders/dh_sky.h"

// users
#include "Shaders/animate_heightmap.h"
#include "Shaders/device_host.h"
#include "Shaders/dh_bindings.h"
#include "raytracing_vk.hpp"

// global variables
constexpr std::string_view PROJECT_NAME = "RaytraceDisplacement";
#define HEIGHTMAP_RESOLUTION 256

/**
 * @brief Move-only VkShaderModule constructed from SPIR-V data
 *
 */
class ShaderModule {
public:
    ShaderModule() = default;
    ~ShaderModule() { destroy(); }

    // 模板构造函数，使用SPIR-V数据创建VkShaderModule
    template <size_t N>
    ShaderModule(VkDevice device, const std::array<uint32_t, N>& spirv)
        : m_device(device),
          m_module(nvvk::createShaderModule(device, spirv.data(), spirv.size() * sizeof(uint32_t)))
    {
    }

    ShaderModule(const ShaderModule& other) = delete;

    ShaderModule(ShaderModule&& other) noexcept : m_device(other.m_device), m_module(other.m_module)
    {
        other.destroy();
    }

    ShaderModule& operator=(const ShaderModule& other) = delete;

    ShaderModule& operator=(ShaderModule&& other) noexcept
    {
        if (this != &other)
        {
            destroy();
            std::swap(m_module, other.m_module);
            std::swap(m_device, other.m_device);
        }
        return *this;
    }

    // 显式类型转换运算符，将ShaderModule对象转换为VkShaderModule
    explicit operator VkShaderModule() const { return m_module; }

private:
    // 销毁VkShaderModule并重置成员变量
    void destroy()
    {
        if (m_module != VK_NULL_HANDLE) { vkDestroyShaderModule(m_device, m_module, nullptr); }
        m_device = VK_NULL_HANDLE;
        m_module = VK_NULL_HANDLE;
    }

    VkDevice       m_device = VK_NULL_HANDLE;
    VkShaderModule m_module = VK_NULL_HANDLE;
};

/**
 * @brief Container to run a compute shader with only a single instance of bindings.
 *
 * @tparam PushConstants
 */
template <class PushConstants> struct SingleComputePipeline
{
public:
    /**
     * @brief Slightly ugly callback for declaring and writing shader bindings
     *
     */
    struct BindingsCB
    {
        std::function<void(nvvk::DescriptorSetBindings&)> declare;
        std::function<std::vector<VkWriteDescriptorSet>(nvvk::DescriptorSetBindings&,
                                                        VkDescriptorSet)>
            create;
    };

    void create(VkDevice device, const BindingsCB& bindingsCB, ShaderModule&& module)
    {
        shaderModule = std::move(module);

        // 创建描述符集
        nvvk::DescriptorSetBindings bindings;
        bindingsCB.declare(bindings);

        descriptorSetLayout = bindings.createLayout(device);
        descriptorPool      = bindings.createPool(device);
        descriptorSet = nvvk::allocateDescriptorSet(device, descriptorPool, descriptorSetLayout);

        std::vector<VkWriteDescriptorSet> bindingsDescWrites =
            bindingsCB.create(bindings, descriptorSet);
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(bindingsDescWrites.size()),
                               bindingsDescWrites.data(), 0, nullptr);

        // 创建PushContent和管线布局
        VkPushConstantRange pushConstantRange{
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset     = 0,
            .size       = sizeof(PushConstants),
        };
        VkPipelineLayoutCreateInfo pipelineLayoutCreate{
            .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .pNext                  = nullptr,
            .flags                  = 0,
            .setLayoutCount         = 1,
            .pSetLayouts            = &descriptorSetLayout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges    = &pushConstantRange,
        };
        vkCreatePipelineLayout(device, &pipelineLayoutCreate, nullptr, &pipelineLayout);

        // shader stage
        VkPipelineShaderStageCreateInfo shaderStageCreate{
            .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext               = nullptr,
            .flags               = 0,
            .stage               = VK_SHADER_STAGE_COMPUTE_BIT,
            .module              = static_cast<VkShaderModule>(shaderModule),
            .pName               = "main",
            .pSpecializationInfo = nullptr,
        };

        // create compute pipline
        VkComputePipelineCreateInfo computePipelineCreate{
            .sType              = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .pNext              = nullptr,
            .flags              = 0,
            .stage              = shaderStageCreate,
            .layout             = pipelineLayout,
            .basePipelineHandle = VK_NULL_HANDLE,
            .basePipelineIndex  = 0,
        };
        vkCreateComputePipelines(device, {}, 1, &computePipelineCreate, nullptr, &pipeline);
    }

    void dispatch(VkCommandBuffer     cmd,
                  const PushConstants pushConstants,
                  uint32_t            groupCountX,
                  uint32_t            groupCountY = 1,
                  uint32_t            groupCountZ = 1)
    {
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                                &descriptorSet, 0, nullptr);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(PushConstants), &pushConstants);
        vkCmdDispatch(cmd, groupCountX, groupCountY, groupCountZ);
    }

    void destroy(VkDevice device)
    {
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        shaderModule = ShaderModule();
    }

private:
    VkDescriptorSet       descriptorSet{VK_NULL_HANDLE};
    VkDescriptorSetLayout descriptorSetLayout{VK_NULL_HANDLE};
    VkDescriptorPool      descriptorPool{VK_NULL_HANDLE};
    VkPipeline            pipeline{VK_NULL_HANDLE};
    VkPipelineLayout      pipelineLayout{VK_NULL_HANDLE};
    ShaderModule          shaderModule;
};

struct AnimatedHeightmap
{
public:
    void create(nvvkhl::AllocVma& alloc, nvvk::DebugUtil& dutil, uint32_t resolution)
    {
        m_resolution = resolution;
        createHeightmaps(alloc, dutil);

        SingleComputePipeline<shaders::AnimatePushConstants>::BindingsCB bindingsCallback{
            .declare = [](nvvk::DescriptorSetBindings& bindings) -> void {
                bindings.addBinding(BINDING_ANIM_IMAGE_A_HEIGHT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                    1, VK_SHADER_STAGE_COMPUTE_BIT);
                bindings.addBinding(BINDING_ANIM_IMAGE_B_HEIGHT, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                    1, VK_SHADER_STAGE_COMPUTE_BIT);
                bindings.addBinding(BINDING_ANIM_IMAGE_A_VELOCITY, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                    1, VK_SHADER_STAGE_COMPUTE_BIT);
                bindings.addBinding(BINDING_ANIM_IMAGE_B_VELOCITY, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
                                    1, VK_SHADER_STAGE_COMPUTE_BIT);
            },
            .create = [this](nvvk::DescriptorSetBindings& bindings,
                             VkDescriptorSet descriptorSet) -> std::vector<VkWriteDescriptorSet> {
                return std::vector<VkWriteDescriptorSet>{
                    bindings.makeWrite(descriptorSet, BINDING_ANIM_IMAGE_A_HEIGHT,
                                       &(this->m_heightmapA.descriptor)),
                    bindings.makeWrite(descriptorSet, BINDING_ANIM_IMAGE_B_HEIGHT,
                                       &(this->m_heightmapB.descriptor)),
                    bindings.makeWrite(descriptorSet, BINDING_ANIM_IMAGE_A_VELOCITY,
                                       &(this->m_velocityA.descriptor)),
                    bindings.makeWrite(descriptorSet, BINDING_ANIM_IMAGE_B_VELOCITY,
                                       &(this->m_velocityB.descriptor)),
                };
            },
        };
        m_animatePipeline.create(alloc.getDevice(), bindingsCallback,
                                 ShaderModule(alloc.getDevice(), animate_heightmap_comp));
    }

    void destroy(nvvkhl::AllocVma& alloc)
    {
        destroyHeightmaps(alloc);
        m_animatePipeline.destroy(alloc.getDevice());
    }

    void clear(VkCommandBuffer cmd)
    {
        VkClearColorValue       heightValue{.float32 = {0.5f}};
        VkClearColorValue       velocityValue{.float32 = {0.0f}};
        VkImageSubresourceRange range{
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        };

        // Perform the initial transition from VK_IMAGE_LAYOUT_UNDEFINED
        imageBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL);
        m_currentIsA = !m_currentIsA;
        imageBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL);

        vkCmdClearColorImage(cmd, height().image, height().descriptor.imageLayout, &heightValue, 1,
                             &range);
        vkCmdClearColorImage(cmd, velocity().image, velocity().descriptor.imageLayout,
                             &velocityValue, 1, &range);
        imageBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL);
        m_currentIsA = !m_currentIsA;
    }

    void animate(VkCommandBuffer cmd)
    {
        shaders::AnimatePushConstants pushConstants{
            .mouse      = m_mouse * glm::vec2(m_resolution),
            .writeToA   = m_currentIsA ? 0U : 1U,
            .resolution = int(m_resolution),
            .deltaTime  = 1.0F,
        };

        // Add some raindrops if the user doesn't draw for a few seconds
        const double timeout      = 5.0;
        const double dropDelay    = 0.5;
        const double dropDuration = 0.05;
        auto         now          = std::chrono::system_clock::now();
        static auto  lastDraw     = std::chrono::system_clock::time_point();
        if (m_mouse.x >= 0.0f) { lastDraw = now; }
        if (std::chrono::duration<double>(now - lastDraw).count() > timeout)
        {
            static std::random_device                    rd;
            static std::mt19937                          mt(rd());
            static std::uniform_real_distribution<float> dist(0.0, double(m_resolution));
            static auto                                  lastDrop = now;
            static glm::vec2                             dropPos  = {};
            if (std::chrono::duration<double>(now - lastDrop).count() > dropDelay)
            {
                lastDrop = now;
                dropPos  = {dist(mt), dist(mt)};
            }
            if (std::chrono::duration<double>(now - lastDrop).count() < dropDuration)
            {
                pushConstants.mouse = dropPos;
            }
        }

        assert(m_resolution % ANIMATION_WORKGROUP_SIZE == 0);
        m_animatePipeline.dispatch(cmd, pushConstants, m_resolution / ANIMATION_WORKGROUP_SIZE,
                                   m_resolution / ANIMATION_WORKGROUP_SIZE);

        imageBarrier(cmd, VK_IMAGE_LAYOUT_GENERAL);

        m_currentIsA = !m_currentIsA;
    }

    const nvvk::Texture& height() const { return m_currentIsA ? m_heightmapB : m_heightmapA; }
    const nvvk::Texture& velocity() const { return m_currentIsA ? m_velocityB : m_velocityA; }
    void                 setMouse(const glm::vec2& position) { m_mouse = position; }

private:
    SingleComputePipeline<shaders::AnimatePushConstants> m_animatePipeline;

    uint32_t      m_resolution;
    nvvk::Texture m_heightmapA;
    nvvk::Texture m_heightmapB;
    nvvk::Texture m_velocityA;
    nvvk::Texture m_velocityB;
    bool          m_currentIsA = true;
    glm::vec2     m_mouse;
    VkImageLayout m_currentImageLayoutsA = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageLayout m_currentImageLayoutsB = VK_IMAGE_LAYOUT_UNDEFINED;

    VkImageLayout& imageLayouts()
    {
        return m_currentIsA ? m_currentImageLayoutsB : m_currentImageLayoutsA;
    }

    void imageBarrier(VkCommandBuffer cmd, VkImageLayout newLayout)
    {
        std::array<VkImageMemoryBarrier, 2> barriers{
            nvvk::makeImageMemoryBarrier(height().image, VK_ACCESS_SHADER_WRITE_BIT,
                                         VK_ACCESS_SHADER_READ_BIT, imageLayouts(), newLayout),
            nvvk::makeImageMemoryBarrier(velocity().image, VK_ACCESS_SHADER_WRITE_BIT,
                                         VK_ACCESS_SHADER_READ_BIT, imageLayouts(), newLayout),
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                                 VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0, 0, nullptr, 0, nullptr, barriers.size(), barriers.data());
        imageLayouts() = newLayout;
    }

    void createHeightmaps(nvvkhl::AllocVma& alloc, nvvk::DebugUtil& dutil)
    {
        VkImageCreateInfo imageInfo = nvvk::makeImage2DCreateInfo(
            VkExtent2D{m_resolution, m_resolution}, VkFormat{VK_FORMAT_R32_SFLOAT},
            VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT);
        nvvk::Image imageA = alloc.createImage(imageInfo);
        nvvk::Image imageB = alloc.createImage(imageInfo);
        nvvk::Image imageC = alloc.createImage(imageInfo);
        nvvk::Image imageD = alloc.createImage(imageInfo);

        VkSamplerCreateInfo samplerInfo{
            .sType                   = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext                   = nullptr,
            .flags                   = 0,
            .magFilter               = VK_FILTER_LINEAR,
            .minFilter               = VK_FILTER_LINEAR,
            .mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeV            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .addressModeW            = VK_SAMPLER_ADDRESS_MODE_REPEAT,
            .mipLodBias              = 0,
            .anisotropyEnable        = false,
            .maxAnisotropy           = 0,
            .compareEnable           = false,
            .compareOp               = VK_COMPARE_OP_NEVER,
            .minLod                  = 0,
            .maxLod                  = std::numeric_limits<float>::max(),
            .borderColor             = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        };
        m_heightmapA = alloc.createTexture(
            imageA, nvvk::makeImage2DViewCreateInfo(imageA.image, VK_FORMAT_R32_SFLOAT),
            samplerInfo);
        m_heightmapB = alloc.createTexture(
            imageB, nvvk::makeImage2DViewCreateInfo(imageB.image, VK_FORMAT_R32_SFLOAT),
            samplerInfo);
        m_velocityA = alloc.createTexture(
            imageC, nvvk::makeImage2DViewCreateInfo(imageC.image, VK_FORMAT_R32_SFLOAT),
            samplerInfo);
        m_velocityB = alloc.createTexture(
            imageD, nvvk::makeImage2DViewCreateInfo(imageD.image, VK_FORMAT_R32_SFLOAT),
            samplerInfo);
        dutil.setObjectName(m_heightmapA.descriptor.imageView, "HeightmapA");
        dutil.setObjectName(m_heightmapB.descriptor.imageView, "HeightmapB");
        dutil.setObjectName(m_velocityA.descriptor.imageView, "VelocityA");
        dutil.setObjectName(m_velocityB.descriptor.imageView, "VelocityB");

        // 图像布局可能会随着时间的推移而变化。尽管如此，nvvk::Texture
        // 在持久的描述符中保持一个布局，但这并不总是保持最新。 nvvk::ResourceAllocator 默认为
        // VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL， 但由于我们知道将过渡到
        // VK_IMAGE_LAYOUT_GENERAL， 因此在创建管道描述符集布局之前进行了设置。
        m_heightmapA.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        m_heightmapB.descriptor.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        m_velocityA.descriptor.imageLayout  = VK_IMAGE_LAYOUT_GENERAL;
        m_velocityB.descriptor.imageLayout  = VK_IMAGE_LAYOUT_GENERAL;
    }

    void destroyHeightmaps(nvvkhl::AllocVma& alloc)
    {
        alloc.destroy(m_heightmapA);
        alloc.destroy(m_heightmapB);
        alloc.destroy(m_velocityA);
        alloc.destroy(m_velocityB);
    }
};

class RaytracingSample : public nvvkhl::IAppElement {
public:
    RaytracingSample()           = default;
    ~RaytracingSample() override = default;

    void onAttach(nvvkhl::Application* app) override
    {
        m_app    = app;
        m_device = app->getDevice();

        VmaAllocatorCreateInfo allocator_info{
            .flags          = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT,
            .physicalDevice = app->getPhysicalDevice(),
            .device         = app->getDevice(),
            .instance       = app->getInstance(),
        };

        m_dutil = std::make_unique<nvvk::DebugUtil>(m_device);         // Debug utility
        m_alloc = std::make_unique<nvvkhl::AllocVma>(allocator_info);  // Allocator
        m_staticCommandPool =
            std::make_unique<nvvk::CommandPool>(m_device, m_app->getQueue(0).queueIndex);
        m_rtContext       = rt::Context{m_device, m_alloc.get(), nullptr,
                                  [](VkResult result) { NVVK_CHECK(result); }};
        m_rtScratchBuffer = std::make_unique<rt::ScratchBuffer>(m_rtContext);

        m_rtSet.init(m_device);

        // Requesting ray tracing properties
        VkPhysicalDeviceProperties2 prop2{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
            .pNext = &m_rtProperties,
        };
        vkGetPhysicalDeviceProperties2(m_app->getPhysicalDevice(), &prop2);

        // Create resources
        createScene();
        createVkBuffers();
        createHrtxPipeline();
        static_assert(HEIGHTMAP_RESOLUTION % ANIMATION_WORKGROUP_SIZE == 0,
                      "currently, resolution must match compute workgroup size");
        m_heightmap.create(*m_alloc, *m_dutil, HEIGHTMAP_RESOLUTION);
        const VkDescriptorImageInfo& heightmapHeightDesc =
            m_heightmap.height().descriptor;  // same for both buffers A/B
        m_heightmapImguiDesc =
            ImGui_ImplVulkan_AddTexture(heightmapHeightDesc.sampler, heightmapHeightDesc.imageView,
                                        heightmapHeightDesc.imageLayout);
        // ANCHOR - ???

        // Initialize the heightmap textures before referencing one in
        // createHrtxMap().
        {
            auto* cmd = m_app->createTempCmdBuffer();
            m_heightmap.clear(cmd);
            m_app->submitAndWaitTempCmdBuffer(cmd);
        }

        {
            VkCommandBuffer cmd = m_app->createTempCmdBuffer();
            createBottomLevelAS(cmd);
            createTopLevelAS(cmd);
            m_app->submitAndWaitTempCmdBuffer(cmd);
        }

        createRtxPipeline();
        createGbuffers(m_viewSize);
    }

    void onDetach() override { destroyResources(); }

    void onResize(uint32_t width, uint32_t height) override
    {
        createGbuffers(glm::vec2{width, height});
        writeRtDesc();
    }

    void onUIRender() override
    {
        // Setting menu
        {
            ImGui::Begin("Settings");
            ImGuiH::CameraWidget();

            using namespace ImGuiH;

            bool recreateAS = false;

            // #MICROMESH - begin
            ImGui::Text("Heightmap Displacement");

            PropertyEditor::begin();
            recreateAS |= PropertyEditor::entry(
                "Enable", [&] { return ImGui::Checkbox("##ll", &m_settings.enableDisplacement); });
            PropertyEditor::entry(
                "Animation", [&] { return ImGui::Checkbox("##ll", &m_settings.enableAnimation); });
            recreateAS |= PropertyEditor::entry("Subdivision Level", [&]() -> bool {
                return ImGui::SliderInt("#1", &m_settings.subdivlevel, 0, 5);
            });
            recreateAS |= PropertyEditor::entry("Heightmap Scale", [&]() -> bool {
                return ImGui::SliderFloat("#1", &m_settings.heightmapScale, 0.05F, 2.0F);
            });

            if (recreateAS)
            {
                vkDeviceWaitIdle(m_device);

                // HrtxMap objects need to be re-created when the input attributes
                // change
                destroyHrtxMaps();

                // Recreate the acceleration structure
                auto* initCmd = m_app->createTempCmdBuffer();
                createBottomLevelAS(initCmd);
                createTopLevelAS(initCmd);
                m_app->submitAndWaitTempCmdBuffer(initCmd);
                writeRtDesc();
            }
            // #MICROMESH - end
            PropertyEditor::end();

            // material
            ImGui::Text("Material");
            PropertyEditor::begin();
            PropertyEditor::entry("Opacity", [&]() -> bool {
                return ImGui::SliderFloat("#1", &m_settings.opacity, 0.0F, 1.0F);
            });
            PropertyEditor::entry("Refractive Index", [&]() -> bool {
                return ImGui::SliderFloat("#1", &m_settings.refractiveIndex, 0.5F, 4.0F);
            });
            PropertyEditor::entry("Density", [&]() -> bool {
                return ImGui::SliderFloat("#1", &m_settings.density, 0.0F, 5.0F);
            });
            PropertyEditor::end();

            ImGui::Separator();

            // sum light
            PropertyEditor::begin();
            glm::vec3 dir = m_skyParams.directionToLight;
            ImGuiH::azimuthElevationSliders(dir, false);
            m_skyParams.directionToLight = dir;
            PropertyEditor::end();

            ImGui::End();
        }

        // Rendering Viewport
        {
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0, 0));
            ImGui::Begin("Viewport");

            // Display the G-Buffer image
            ImGui::Image(m_gBuffer->getDescriptorSet(), ImGui::GetContentRegionAvail());

            ImGui::End();
            ImGui::PopStyleVar();
        }

        // Heightmap preview and mouse interaction
        {
            ImGui::Begin("Heightmap");
            ImGui::Text("Animated heightmap. Click to draw.");

            ImVec2 windowPos       = ImGui::GetCursorScreenPos();
            ImVec2 windowSize      = ImGui::GetContentRegionAvail();
            ImVec2 previewSize     = {std::min(windowSize.x, windowSize.y),
                                      std::min(windowSize.x, windowSize.y)};
            ImVec2 marginTotal     = {windowSize.x - previewSize.x, windowSize.y - previewSize.y};
            ImVec2 heightmapOffset = {marginTotal.x / 2, marginTotal.y / 2};
            auto   mouseAbs        = ImGui::GetIO().MousePos;
            ImVec2 mouse           = {mouseAbs.x - heightmapOffset.x - windowPos.x,
                                      mouseAbs.y - heightmapOffset.y - windowPos.y};
            auto   mouseNorm       = glm::vec2{mouse.x, mouse.y} /
                             glm::vec2{previewSize.x, previewSize.y};  // 归一化到0-1

            // Update the heightmap mouse position when dragging the mouse in the
            // heightmap window. If not clicking, moving the mouse off-screen will
            // stop it affecting the animation
            if (ImGui::GetIO().MouseDown[0]                     //
                && mouseNorm.x >= 0.0f && mouseNorm.x <= 1.0f   //
                && mouseNorm.y >= 0.0f && mouseNorm.y <= 1.0f)  //
            {
                m_heightmap.setMouse(mouseNorm);
            }
            else { m_heightmap.setMouse(glm::vec2(-0.5f)); }

            // Display the heightmap
            ImVec2 drawPos = ImGui::GetCursorPos();
            ImGui::SetCursorPos({heightmapOffset.x + drawPos.x, heightmapOffset.y + drawPos.y});
            // cursor与用户的鼠标指针（系统光标）无关，是 ImGui 内部用于布局的工具
            ImGui::Image(m_heightmapImguiDesc, previewSize);

            ImGui::End();
        }
    }

    void onRender(VkCommandBuffer cmd) override
    {
        if (m_settings.enableAnimation)
        {
            // 在提交 m_cmdHrtxUpdate 之前推进高度图动画。注意
            // animate() 被调用两次，以便将双缓冲的结果返回到原始状态。理想情况下，这里只需要调用
            // submitTempCmdBuffer()， 以避免在渲染循环中出现 GPU 停顿，但该函数尚未实现。
            VkCommandBuffer animCmd = m_app->createTempCmdBuffer();
            m_heightmap.animate(animCmd);
            m_heightmap.animate(animCmd);
            m_app->submitAndWaitTempCmdBuffer(animCmd);

            // Update the raytracing displacement from the heightmap. m_cmdHrtxUpdate
            // already includes an image barrier for compute shader writes.
            VkSubmitInfo submit = {
                .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
                .commandBufferCount = 1,
                .pCommandBuffers    = &m_cmdHrtxUpdate,
            };
            vkQueueSubmit(m_app->getQueue(0).queue, 1, &submit, VK_NULL_HANDLE);

            // 为高度图重建 BLAS。然后需要更新 TLAS，
            // 但不需要更新其他静态几何体。请注意，上述动画和
            // heightmap_rtx 命令在 onRender() 返回之前在同一队列上提交，
            // 并且临时的 'cmd' 也被提交，因此事件的顺序仍然被保留。
            m_rtBlas[0].update(m_rtContext, m_rtBlasInput[0], *m_rtScratchBuffer, cmd);

            // 对于 TLAS，使用 PREFER_FAST_TRACE 标志，并仅执行重建，
            m_rtTlas->rebuild(m_rtContext, m_rtTlasInput, *m_rtScratchBuffer, cmd);

            // NOTE - 为什么blas是更新而tlas是重建？
        }

        auto sdbg = m_dutil->DBG_SCOPE(cmd);

        // 摄像机操作
        float     view_aspect_ratio = m_viewSize.x / m_viewSize.y;
        glm::vec3 eye;
        glm::vec3 center;
        glm::vec3 up;
        CameraManip.getLookat(eye, center, up);
        // Update the uniform buffer containing frame info
        const auto& clip = CameraManip.getClipPlanes();
        auto proj_mat = glm::perspectiveRH_ZO(glm::radians(CameraManip.getFov()), view_aspect_ratio,
                                              clip.x, clip.y);
        proj_mat[1][1] *= -1;  // flip y axis
        auto               view_mat = CameraManip.getMatrix();
        shaders::FrameInfo finfo{
            .proj    = proj_mat,
            .view    = view_mat,
            .projInv = glm::inverse(proj_mat),
            .viewInv = glm::inverse(view_mat),
            .camPos  = eye,
        };
        vkCmdUpdateBuffer(cmd, m_bFrameInfo.buffer, 0, sizeof(shaders::FrameInfo), &finfo);

        // Update the sky
        vkCmdUpdateBuffer(cmd, m_bSkyParams.buffer, 0, sizeof(nvvkhl_shaders::SimpleSkyParameters),
                          &m_skyParams);

        // Ray trace
        std::vector<VkDescriptorSet> desc_sets{m_rtSet.getSet()};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.plines[0]);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtPipe.layout, 0,
                                static_cast<uint32_t>(desc_sets.size()), desc_sets.data(), 0,
                                nullptr);

        m_pushConst.opacity         = m_settings.opacity;
        m_pushConst.refractiveIndex = m_settings.refractiveIndex;
        m_pushConst.density         = m_settings.density;
        m_pushConst.heightmapScale  = m_settings.heightmapScale;
        m_pushConst.maxDepth        = m_settings.maxDepth;
        m_pushConst.wireframeScale  = 1 << m_settings.subdivlevel;
        vkCmdPushConstants(cmd, m_rtPipe.layout, VK_SHADER_STAGE_ALL, 0,
                           sizeof(shaders::PushConstant), &m_pushConst);
        // NOTE - 设置少量的管线运行时常量

        const auto& regions = m_sbt.getRegions();
        const auto& size    = m_app->getViewportSize();
        vkCmdTraceRaysKHR(cmd, &regions[0], &regions[1], &regions[2], &regions[3], size.width,
                          size.height, 1);
    }

private:
    struct settings  // ANCHOR - 渲染器的设置
    {
        float opacity{0.25F};
        float refractiveIndex{1.03F};
        float density{1.0F};
        float heightmapScale{0.2f};
        int   maxDepth{5};
        bool  enableAnimation{true};
        bool  enableDisplacement{true};
        int   subdivlevel{3};
    } m_settings;

    struct PrimitiveMeshVk
    {
        nvvk::Buffer vertices, indices;
    };

    // vulkan应用与实例
    nvvkhl::Application*               m_app{nullptr};
    std::unique_ptr<nvvk::DebugUtil>   m_dutil;
    nvvkhl::AllocVma                   m_alloc;
    std::unique_ptr<nvvk::CommandPool> m_staticCommandPool;

    // 基本渲染设置
    glm::vec2 m_viewSize    = {1, 1};
    VkFormat  m_colorFormat = VK_FORMAT_R8G8B8A8_UNORM;       // Color format of the image
    VkFormat  m_depthFormat = VK_FORMAT_X8_D24_UNORM_PACK32;  // Depth format of the depth buffer
    VkClearColorValue m_clearColor = {.float32 = {0.3F, 0.3F, 0.3F, 1.0F}};  // Clear color
    VkDevice          m_device     = VK_NULL_HANDLE;                         // Convenient
    std::unique_ptr<nvvkhl::GBuffer>    m_gBuffer;  // G-Buffers: color + depth
    nvvkhl_shaders::SimpleSkyParameters m_skyParams{};

    // GPU scene buffers
    std::vector<PrimitiveMeshVk> m_bMeshes;  // on gpu
    nvvk::Buffer                 m_bFrameInfo, m_bSkyParams;

    // Data and settings
    std::vector<nvh::PrimitiveMesh> m_meshes;  // on cpu
    std::vector<nvh::Node>          m_nodes;

    // Raytracing pipeline
    nvvk::DescriptorSetContainer m_rtSet;                  // Descriptor set
    shaders::PushConstant        m_pushConst{};            // Information sent to the shader
    VkPipelineLayout m_pipelineLayout   = VK_NULL_HANDLE;  // The description of the pipeline
    VkPipeline       m_graphicsPipeline = VK_NULL_HANDLE;  // The graphic pipeline to render
    int              m_frame            = 0;

    // ray tracing
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProperties{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR,
    };
    ShaderModule                                           m_rtShaderRgen;
    ShaderModule                                           m_rtShaderRmiss;
    ShaderModule                                           m_rtShaderRchit;
    nvvk::SBTWrapper                                       m_sbt;  // Shader binding table wrapper
    rt::Context                                            m_rtContext;
    std::unique_ptr<rt::ScratchBuffer>                     m_rtScratchBuffer;
    VkAccelerationStructureTrianglesDisplacementMicromapNV m_rtDisplacement;
    std::vector<rt::AccelerationStructureInput>            m_rtBlasInput;
    std::vector<rt::BuiltAccelerationStructure>            m_rtBlas;
    std::unique_ptr<rt::InstanceBuffer>                    m_rtInstances;
    rt::AccelerationStructureInput                         m_rtTlasInput;
    std::unique_ptr<rt::BuiltAccelerationStructure>        m_rtTlas;
    nvvkhl::PipelineContainer                              m_rtPipe;

    // height map
    HrtxPipeline      m_hrtxPipeline{};
    HrtxMap           m_hrtxMap{};
    AnimatedHeightmap m_heightmap;
    VkDescriptorSet   m_heightmapImguiDesc = VK_NULL_HANDLE;
    VkCommandBuffer   m_cmdHrtxUpdate      = VK_NULL_HANDLE;

    void createScene()
    {
        float cubeHeight = 0.5f;
        m_meshes         = std::vector<nvh::PrimitiveMesh>{
            nvh::createPlane(HEIGHTMAP_RESOLUTION / 32, 1.0F, 1.0F),
            nvh::createCube(1.0F, cubeHeight, 1.0F),
        };
        m_nodes = std::vector<nvh::Node>{
            nvh::Node{.mesh = 0},
            nvh::Node{.mesh = 1},
        };

        // Remove the top face of the cube and move down so it's flush with the
        // heightmap-displaced plane.
        m_meshes[1].triangles.pop_back();
        m_meshes[1].triangles.pop_back();
        std::for_each(m_meshes[1].vertices.begin(), m_meshes[1].vertices.end(),
                      [&](nvh::PrimitiveVertex& v) { v.p.y -= cubeHeight * 0.5f; });

        // Setting camera to see the scene
        CameraManip.setClipPlanes({0.01F, 100.0F});
        CameraManip.setLookat({0.5F, 0.2F, 1.0F}, {0.0F, -0.2F, 0.0F}, {0.0F, 1.0F, 0.0F});

        // Default Sky values
        m_skyParams = nvvkhl_shaders::initSimpleSkyParameters();
    }

    void createGbuffers(const glm::vec2& size)
    {
        vkDeviceWaitIdle(m_device);

        // Rendering image targets
        m_viewSize = size;
        m_gBuffer  = std::make_unique<nvvkhl::GBuffer>(
            m_device, m_alloc.get(),
            VkExtent2D{static_cast<uint32_t>(size.x), static_cast<uint32_t>(size.y)}, m_colorFormat,
            m_depthFormat);
    }

    void createVkBuffers()
    {
        auto* cmd = m_app->createTempCmdBuffer();
        m_bMeshes.resize(m_meshes.size());

        auto rt_usage_flag = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                             VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;

        // Create a buffer of Vertex and Index per mesh
        for (size_t i = 0; i < m_meshes.size(); i++)
        {
            auto& m = m_bMeshes[i];

            // 调整纹理坐标以准确落在纹理素中心。这是必要的，因为 heightmap_rtx 使用 GLSL 的
            // texture() 对高度图进行采样，像素值位于纹理素中心，例如 {0.5 / 宽度, 0.5 / 高度}
            // 采样像素 {0, 0}。然而，nvh::createPlane() 生成的纹理坐标范围为
            // [0.0, 1.0]。此外，为了匹配 Imgui 预览图像，还翻转了 Y 坐标。
            float scale  = (float(HEIGHTMAP_RESOLUTION) - 1.0f) / float(HEIGHTMAP_RESOLUTION);
            float offset = 0.5f / float(HEIGHTMAP_RESOLUTION);
            std::for_each(m_meshes[i].vertices.begin(), m_meshes[i].vertices.end(),
                          [&](nvh::PrimitiveVertex& v) {
                              v.t   = v.t * scale + offset;
                              v.t.y = 1.0f - v.t.y;
                          });

            m.vertices = m_alloc->createBuffer(cmd, m_meshes[i].vertices, rt_usage_flag);
            m.indices  = m_alloc->createBuffer(cmd, m_meshes[i].triangles, rt_usage_flag);
            m_dutil->DBG_NAME_IDX(m.vertices.buffer, i);
            m_dutil->DBG_NAME_IDX(m.indices.buffer, i);
        }

        // Create the buffer of the current frame, changing at each frame
        m_bFrameInfo = m_alloc->createBuffer(
            sizeof(shaders::FrameInfo), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        m_dutil->DBG_NAME(m_bFrameInfo.buffer);

        // Create the buffer of sky parameters, updated at each frame
        m_bSkyParams = m_alloc->createBuffer(
            sizeof(nvvkhl_shaders::SimpleSkyParameters), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        m_dutil->DBG_NAME(m_bSkyParams.buffer);

        m_app->submitAndWaitTempCmdBuffer(cmd);
    }

    void createHrtxPipeline()
    {
        HrtxAllocatorCallbacks allocatorCallbacks{
            .createBuffer = [](const VkBufferCreateInfo    bufferCreateInfo,
                               const VkMemoryPropertyFlags memoryProperties,
                               void*                       userPtr) -> VkBuffer* {
                auto alloc  = reinterpret_cast<nvvkhl::AllocVma*>(userPtr);
                auto result = new nvvk::Buffer();
                *result = alloc->createBuffer(bufferCreateInfo, memoryProperties);  // 复制赋值
                return &result->buffer;  // return pointer to member
            },
            .destroyBuffer = [](VkBuffer* bufferPtr, void* userPtr) -> void {
                auto alloc = reinterpret_cast<nvvkhl::AllocVma*>(userPtr);
                // reconstruct from pointer to member
                auto nvvkBuffer = reinterpret_cast<nvvk::Buffer*>(
                    reinterpret_cast<char*>(bufferPtr) - offsetof(nvvk::Buffer, buffer));
                // NOTE -
                // bufferPtr转成1B偏移的指针，然后减去buffer成员的偏移量，得到nvvk::Buffer的头指针
                alloc->destroy(*nvvkBuffer);
                delete nvvkBuffer;
            },
            .userPtr         = nullptr,
            .systemAllocator = m_alloc.get(),
        };

        // Create a HrtxPipeline object. This holds the shader and resources for baking
        HrtxPipelineCreate hrtxPipelineCreate{
            .physicalDevice      = m_app->getPhysicalDevice(),
            .device              = m_app->getDevice(),
            .allocator           = allocatorCallbacks,
            .instance            = VK_NULL_HANDLE,
            .getInstanceProcAddr = nullptr,
            .getDeviceProcAddr   = nullptr,
            .pipelineCache       = VK_NULL_HANDLE,
            .checkResultCallback = [](VkResult result) -> void {
                nvvk::checkResult(result, "HRTX");
            },
        };

        auto* cmd = m_app->createTempCmdBuffer();
        if (hrtxCreatePipeline(cmd, &hrtxPipelineCreate, &m_hrtxPipeline) != VK_SUCCESS)
        {
            LOGW("Warning: Failed to create HrtxPipeline. Raytracing heightmaps will not work.\n");
        }
        m_app->submitAndWaitTempCmdBuffer(cmd);
    }

    HrtxMap createHrtxMap(const VkAccelerationStructureGeometryKHR& geometry,
                          uint32_t                                  triangleCount,
                          const PrimitiveMeshVk&                    mesh,
                          const nvvk::Texture&                      texture,
                          VkCommandBuffer                           cmd)
    {
        // Barrier to make sure writes by the compute shader are visible when creating the micromap
        auto imageBarrier = nvvk::makeImageMemoryBarrier(
            texture.image, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL);
        hrtxBarrierFlags(nullptr, nullptr, nullptr, nullptr, &imageBarrier.newLayout);
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &imageBarrier);

        HrtxMapCreate mapCreate{
            .triangles           = &geometry.geometry.triangles,
            .primitiveCount      = triangleCount,
            .textureCoordsBuffer = {.deviceAddress = nvvk::getBufferDeviceAddress(
                                                         m_device, mesh.vertices.buffer) +
                                                     offsetof(nvh::PrimitiveVertex, t)},
            .textureCoordsFormat = VK_FORMAT_R32G32_SFLOAT,
            .textureCoordsStride = sizeof(nvh::PrimitiveVertex),
            .directionsBuffer    = {.deviceAddress =
                                        nvvk::getBufferDeviceAddress(m_device, mesh.vertices.buffer) +
                                        offsetof(nvh::PrimitiveVertex, n)},
            .directionsFormat    = VK_FORMAT_R32G32B32_SFLOAT,
            .directionsStride    = sizeof(nvh::PrimitiveVertex),
            .heightmapImage      = texture.descriptor,
            .heightmapBias       = -m_settings.heightmapScale * 0.5f,
            .heightmapScale      = m_settings.heightmapScale,
            .subdivisionLevel    = static_cast<uint32_t>(m_settings.subdivlevel),
        };

        HrtxMap hrtxMap{};
        if (hrtxCmdCreateMap(cmd, m_hrtxPipeline, &mapCreate, &hrtxMap) != VK_SUCCESS)
        {
            LOGW("Warning: Failed to create HrtxMap for mesh %p. Raytracing heightmaps will not "
                 "work.\n",
                 &mesh);
        }
        return hrtxMap;
    }

    void destroyHrtxMaps()
    {
        m_staticCommandPool->destroy(m_cmdHrtxUpdate);
        m_cmdHrtxUpdate = VK_NULL_HANDLE;

        hrtxDestroyMap(m_hrtxMap);
        m_hrtxMap = nullptr;
    }

    /**
     * @brief Build a bottom level acceleration structure (BLAS) for each mesh
     *
     * @param cmd
     */
    void createBottomLevelAS(VkCommandBuffer cmd)
    {
        // Prepare to create one BLAS per mesh
        m_rtBlasInput.clear();
        for (size_t i = 0; i < m_meshes.size(); i++)
        {
            // Each BLAS has only one geometry input.
            assert(!m_meshes[i].vertices.empty());

            std::vector<rt::SimpleGeometryInput> simpleGeometryInputs{
                rt::SimpleGeometryInput{
                    .triangleCount = static_cast<uint32_t>(m_meshes[i].triangles.size()),
                    .maxVertex     = static_cast<uint32_t>(m_meshes[i].vertices.size()) - 1,
                    .indexAddress =
                        nvvk::getBufferDeviceAddress(m_device, m_bMeshes[i].indices.buffer),
                    .vertexAddress =
                        nvvk::getBufferDeviceAddress(m_device, m_bMeshes[i].vertices.buffer),
                    .vertexStride = sizeof(nvh::PrimitiveVertex),
                },
            };
            m_rtBlasInput.push_back(rt::createBlasInput(
                simpleGeometryInputs, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR |
                                          VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR |
                                          VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_DATA_ACCESS_KHR));
        }

        // 为第一个网格创建一个 HrtxMap。大多数 nvpro_core
        // 命令缓冲区是临时的/一次性的。在这种情况下，相同的命令缓冲区可以简单地重新提交以重建所有
        // HrtxMap 对象。 注意：这会记录一个仅引用双缓冲高度图之一的命令缓冲区。
        assert(!m_cmdHrtxUpdate);
        assert(!m_hrtxMap);
        m_cmdHrtxUpdate = m_staticCommandPool->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                                                   true, 0, nullptr);
        m_hrtxMap       = createHrtxMap(m_rtBlasInput[0].geometries[0],
                                        m_rtBlasInput[0].rangeInfos[0].primitiveCount, m_bMeshes[0],
                                        m_heightmap.height(), m_cmdHrtxUpdate);
        // NOTE - [0]是plane，[1]是cube

        if (m_hrtxMap == nullptr)
        {
            LOGE("ERROR: createHrtxMap() failed");
            exit(1);
        }
        m_rtDisplacement = hrtxMapDesc(m_hrtxMap);

        m_staticCommandPool->submit(1, &m_cmdHrtxUpdate);

        // Apply the heightmap to the first mesh. The pNext pointer is reused for
        // build updates, so the object is stored in m_rtDisplacement.
        m_rtBlasInput[0].geometries[0].geometry.triangles.pNext =
            m_settings.enableDisplacement ? &m_rtDisplacement : nullptr;

        // Create the bottom level acceleration structures
        m_rtBlas.clear();
        for (auto& blasInput : m_rtBlasInput)
        {
            rt::AccelerationStructureSizes blasSizes(m_rtContext, blasInput);
            m_rtBlas.emplace_back(
                m_rtContext, rt::AccelerationStructure(m_rtContext, blasInput.type, *blasSizes, 0),
                blasInput, *m_rtScratchBuffer, cmd);
        }
    }

    /**
     * @brief Create the top level acceleration structures, referencing all BLAS
     *
     * @param cmd
     */
    void createTopLevelAS(VkCommandBuffer cmd)
    {
        // Update the cube's scale so that the heightmap cannot intersect it
        m_nodes[1].scale.y = m_settings.heightmapScale * 1.01f;

        VkGeometryInstanceFlagsKHR instanceFlags =
            VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR |
            VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
        std::vector<VkAccelerationStructureInstanceKHR> instances(m_nodes.size());
        for (auto& node : m_nodes)
        {
            instances.push_back(VkAccelerationStructureInstanceKHR{
                .transform =
                    nvvk::toTransformMatrixKHR(node.localMatrix()),  // Position of the instance
                .instanceCustomIndex =
                    static_cast<uint32_t>(node.mesh) & 0x00FFFFFF,  // gl_InstanceCustomIndexEXT
                .mask = 0xFF,  // 前八位是mask，后24位才是索引
                .instanceShaderBindingTableRecordOffset =
                    0,  // We will use the same hit group for all objects
                .flags                          = instanceFlags & 0xFFU,
                .accelerationStructureReference = m_rtBlas[node.mesh].address(),
            });
        }
        m_rtInstances = std::make_unique<rt::InstanceBuffer>(m_rtContext, instances, cmd);
        m_rtTlasInput =
            rt::createTlasInput(instances.size(), m_rtInstances->address(),
                                VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR |
                                    VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR);

        // Create the top level acceleration structure
        rt::AccelerationStructureSizes tlasSizes(m_rtContext, m_rtTlasInput);
        m_rtTlas = std::make_unique<rt::BuiltAccelerationStructure>(
            m_rtContext, rt::AccelerationStructure(m_rtContext, m_rtTlasInput.type, *tlasSizes, 0),
            m_rtTlasInput, *m_rtScratchBuffer, cmd);
    }

    /**
     * @brief Pipeline for the ray tracer: all shaders, raygen, chit, miss
     *
     */
    void createRtxPipeline()
    {
        auto& p = m_rtPipe;
        auto& d = m_rtSet;
        p.plines.resize(1);

        // This descriptor set, holds the top level acceleration structure and the output image
        // Create Binding Set
        d.addBinding(BRtTlas, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                     VK_SHADER_STAGE_ALL);
        d.addBinding(BRtOutImage, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_ALL);
        d.addBinding(BRtFrameInfo, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
        d.addBinding(BRtSkyParam, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_ALL);
        d.initLayout();
        d.initPool(1);

        m_dutil->DBG_NAME(d.getLayout());
        m_dutil->DBG_NAME(d.getSet(0));

        // Creating all shaders
        enum StageIndices { eRaygen, eMiss, eClosestHit, eShaderGroupCount };
        std::array<VkPipelineShaderStageCreateInfo, 3> stages{
            VkPipelineShaderStageCreateInfo{
                .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext               = nullptr,
                .flags               = 0,
                .stage               = VK_SHADER_STAGE_RAYGEN_BIT_KHR,
                .module              = ShaderModule(),
                .pName               = "main",
                .pSpecializationInfo = nullptr,
            },  // RAYGEN
            VkPipelineShaderStageCreateInfo{
                .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext               = nullptr,
                .flags               = 0,
                .stage               = VK_SHADER_STAGE_MISS_BIT_KHR,
                .module              = ShaderModule(),
                .pName               = "main",
                .pSpecializationInfo = nullptr,
            },  // MISS
            VkPipelineShaderStageCreateInfo{
                .sType               = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext               = nullptr,
                .flags               = 0,
                .stage               = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR,
                .module              = ShaderModule(),
                .pName               = "main",
                .pSpecializationInfo = nullptr,
            },  // CLOSEST_HIT
        };
        m_dutil->setObjectName(stages[eRaygen].module, "Raygen");
        m_dutil->setObjectName(stages[eMiss].module, "Miss");
        m_dutil->setObjectName(stages[eClosestHit].module, "Closest Hit");

        // Shader groups
        std::array<VkRayTracingShaderGroupCreateInfoKHR, eShaderGroupCount> shader_groups{
            VkRayTracingShaderGroupCreateInfoKHR{
                .sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                .pNext              = nullptr,
                .type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                .generalShader      = eRaygen,
                .closestHitShader   = VK_SHADER_UNUSED_KHR,
                .anyHitShader       = VK_SHADER_UNUSED_KHR,
                .intersectionShader = VK_SHADER_UNUSED_KHR,
                .pShaderGroupCaptureReplayHandle = nullptr,
            },  // RAYGEN
            VkRayTracingShaderGroupCreateInfoKHR{
                .sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                .pNext              = nullptr,
                .type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR,
                .generalShader      = eMiss,
                .closestHitShader   = VK_SHADER_UNUSED_KHR,
                .anyHitShader       = VK_SHADER_UNUSED_KHR,
                .intersectionShader = VK_SHADER_UNUSED_KHR,
                .pShaderGroupCaptureReplayHandle = nullptr,
            },  // MISS
            VkRayTracingShaderGroupCreateInfoKHR{
                .sType              = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR,
                .pNext              = nullptr,
                .type               = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR,
                .generalShader      = VK_SHADER_UNUSED_KHR,
                .closestHitShader   = eClosestHit,
                .anyHitShader       = VK_SHADER_UNUSED_KHR,
                .intersectionShader = VK_SHADER_UNUSED_KHR,
                .pShaderGroupCaptureReplayHandle = nullptr,
            },  // CLOSEST_HIT
        };

        // Push constant: we want to be able to update constants used by the shaders
        VkPushConstantRange push_constant{
            .stageFlags = VK_SHADER_STAGE_ALL,
            .offset     = 0,
            .size       = sizeof(shaders::PushConstant),
        };

        // Descriptor sets
        std::vector<VkDescriptorSetLayout> rt_desc_set_layouts = {d.getLayout()};
        VkPipelineLayoutCreateInfo         pipeline_layout_create_info{
                    .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                    .pNext                  = nullptr,
                    .flags                  = 0,
                    .setLayoutCount         = static_cast<uint32_t>(rt_desc_set_layouts.size()),
                    .pSetLayouts            = rt_desc_set_layouts.data(),
                    .pushConstantRangeCount = 1,
                    .pPushConstantRanges    = &push_constant,
        };
        NVVK_CHECK(
            vkCreatePipelineLayout(m_device, &pipeline_layout_create_info, nullptr, &p.layout));
        m_dutil->DBG_NAME(p.layout);

        VkRayTracingPipelineCreateInfoKHR ray_pipeline_info{
            .sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR,
            .pNext = nullptr,
            .flags = VK_PIPELINE_CREATE_RAY_TRACING_DISPLACEMENT_MICROMAP_BIT_NV,  // #MICROMESH
            .stageCount = static_cast<uint32_t>(stages.size()),  // Stages are shaders
            .pStages    = stages.data(),
            .groupCount = static_cast<uint32_t>(shader_groups.size()),
            .pGroups    = shader_groups.data(),
            .maxPipelineRayRecursionDepth = 10,  // Ray depth
            .pLibraryInfo                 = nullptr,
            .pLibraryInterface            = nullptr,
            .pDynamicState                = nullptr,
            .layout                       = p.layout,
            .basePipelineHandle           = VK_NULL_HANDLE,
            .basePipelineIndex            = 0,
        };
        NVVK_CHECK(vkCreateRayTracingPipelinesKHR(m_device, {}, {}, 1, &ray_pipeline_info, nullptr,
                                                  (p.plines).data()));
        m_dutil->DBG_NAME(p.plines[0]);

        // Creating the SBT
        m_sbt.create(p.plines[0], ray_pipeline_info);
    }

    void writeRtDesc()
    {
        auto& d = m_rtSet;

        // Write to descriptors

        // Acceleration Structures
        VkAccelerationStructureKHR                   tlas = *m_rtTlas;
        VkWriteDescriptorSetAccelerationStructureKHR desc_as_info{
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
            .pNext = nullptr,
            .accelerationStructureCount = 1,
            .pAccelerationStructures    = &tlas,
        };

        // DescriptorImageInfo
        VkDescriptorImageInfo image_info{
            .sampler     = VK_NULL_HANDLE,
            .imageView   = m_gBuffer->getColorImageView(),
            .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        };

        // FrameInfo
        VkDescriptorBufferInfo dbi_unif{
            .buffer = m_bFrameInfo.buffer,
            .offset = 0,
            .range  = VK_WHOLE_SIZE,
        };
        VkDescriptorBufferInfo dbi_sky{
            .buffer = m_bSkyParams.buffer,
            .offset = 0,
            .range  = VK_WHOLE_SIZE,
        };

        std::array<VkWriteDescriptorSet, 4> writes{
            d.makeWrite(0, BRtTlas, &desc_as_info),
            d.makeWrite(0, BRtOutImage, &image_info),
            d.makeWrite(0, BRtFrameInfo, &dbi_unif),
            d.makeWrite(0, BRtSkyParam, &dbi_sky),
        };
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0,
                               nullptr);
    }

    void destroyResources()
    {
        vkDeviceWaitIdle(m_device);
        vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
        vkDestroyPipeline(m_device, m_graphicsPipeline, nullptr);

        for (auto& m : m_bMeshes)
        {
            m_alloc->destroy(m.vertices);
            m_alloc->destroy(m.indices);
        }
        m_alloc->destroy(m_bFrameInfo);
        m_alloc->destroy(m_bSkyParams);

        ImGui_ImplVulkan_RemoveTexture(m_heightmapImguiDesc);
        m_heightmap.destroy(*m_alloc);

        hrtxDestroyMap(m_hrtxMap);
        hrtxDestroyPipeline(m_hrtxPipeline);

        m_rtSet.deinit();
        m_gBuffer.reset();

        m_rtPipe.destroy(m_device);

        m_sbt.destroy();
        m_rtShaderRgen  = ShaderModule();
        m_rtShaderRmiss = ShaderModule();
        m_rtShaderRchit = ShaderModule();

        m_rtScratchBuffer.reset();
        m_rtBlas.clear();
        m_rtInstances.reset();
        m_rtTlas.reset();
    }
};

std::unique_ptr<nvvk::Context> createVulkanContext()
{
    // base vk app setup
    auto vkSetup = nvvk::ContextCreateInfo(false);
    vkSetup.setVersion(1, 3);

    // extensions and layers
    vkSetup.addDeviceExtension(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    // #VKRay: Activate the ray tracing extension
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel_feature{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
    };
    vkSetup.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false,
                               &accel_feature);  // To build acceleration structures
    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rt_pipeline_feature{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR,
    };
    vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME, false,
                               &rt_pipeline_feature);  // To use vkCmdTraceRaysKHR
    vkSetup.addDeviceExtension(
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // Required by ray tracing pipeline
    vkSetup.addDeviceExtension(VK_KHR_SYNCHRONIZATION_2_EXTENSION_NAME);
    VkPhysicalDeviceRayTracingPositionFetchFeaturesKHR rt_position_fetch{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_POSITION_FETCH_FEATURES_KHR,
    };
    vkSetup.addDeviceExtension(VK_KHR_RAY_TRACING_POSITION_FETCH_EXTENSION_NAME, false,
                               &rt_position_fetch);

    // #MICROMESH
    static VkPhysicalDeviceOpacityMicromapFeaturesEXT mm_opacity_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_OPACITY_MICROMAP_FEATURES_EXT,
    };
    static VkPhysicalDeviceDisplacementMicromapFeaturesNV mm_displacement_features{
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DISPLACEMENT_MICROMAP_FEATURES_NV,
    };
    vkSetup.addDeviceExtension(VK_EXT_OPACITY_MICROMAP_EXTENSION_NAME, true, &mm_opacity_features);
    vkSetup.addDeviceExtension(VK_NV_DISPLACEMENT_MICROMAP_EXTENSION_NAME, true,
                               &mm_displacement_features);
    if (!mm_opacity_features.micromap)
    {
        throw std::runtime_error("ERROR: Micro-Mesh not supported");
    }
    if (!mm_displacement_features.displacementMicromap)
    {
        throw std::runtime_error("ERROR: Micro-Mesh displacement not supported");
    }

    // Display extension
    vkSetup.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    vkSetup.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);  // 但是他没有启用校验层
    nvvkhl::addSurfaceExtensions(vkSetup.instanceExtensions);

    // Creating the Vulkan context
    auto m_context = std::make_unique<nvvk::Context>();
    m_context->init(vkSetup);
    // Disable error messages introduced by micromesh
    m_context->ignoreDebugMessage(0x901f59ec);  // Unknown extension
    m_context->ignoreDebugMessage(0xdd73dbcf);  // Unknown structure
    m_context->ignoreDebugMessage(
        0xba164058);  // Unknown flag  vkGetAccelerationStructureBuildSizesKHR:
    m_context->ignoreDebugMessage(0x22d5bbdc);  // Unknown flag  vkCreateRayTracingPipelinesKHR
    m_context->ignoreDebugMessage(0x27112e51);  // Unknown flag  vkCreateBuffer
    m_context->ignoreDebugMessage(
        0x79de34d4);  // Unknown VK_NV_displacement_micromesh, VK_NV_opacity_micromesh

    return m_context;
}

std::unique_ptr<nvvkhl::Application>
createVulkanApplication(std::unique_ptr<nvvk::Context>& context)
{
    nvvkhl::ApplicationCreateInfo spec{
        .name      = "RaytraceDisplacement Example",
        .vSync     = false,
        .dockSetup = [](ImGuiID viewportID) -> void {
            ImGuiID settingID =
                ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.2F, nullptr, &viewportID);
            ImGui::DockBuilderDockWindow("Settings", settingID);
            ImGuiID heightmapID = ImGui::DockBuilderSplitNode(viewportID, ImGuiDir_Right, 0.382F,
                                                              nullptr, &viewportID);
            ImGui::DockBuilderDockWindow("Heightmap", heightmapID);
        },
        .instance       = context->m_instance,
        .device         = context->m_device,
        .physicalDevice = context->m_physicalDevice,
        .queues =
            {
                nvvkhl::ApplicationQueue{
                    context->m_queueGCT.familyIndex,
                    context->m_queueGCT.queueIndex,
                    context->m_queueGCT.queue,
                },
                nvvkhl::ApplicationQueue{
                    context->m_queueC.familyIndex,
                    context->m_queueC.queueIndex,
                    context->m_queueC.queue,
                },
                nvvkhl::ApplicationQueue{
                    context->m_queueT.familyIndex,
                    context->m_queueT.queueIndex,
                    context->m_queueT.queue,
                },
            },
    };

    // Create the application
    return std::make_unique<nvvkhl::Application>(spec);
}

int main(int argc, char** argv)
{
    try
    {
        auto m_context = createVulkanContext();               // Create the Vulkan context
        auto app       = createVulkanApplication(m_context);  // Application Vulkan setup

        // Add all application elements
        auto test = std::make_shared<nvvkhl::ElementTesting>(argc, argv);
        app->addElement(test);  // test framework
        app->addElement<nvvkhl::ElementCamera>();
        app->addElement<nvvkhl::ElementDefaultMenu>();         // Menu / Quit
        app->addElement<nvvkhl::ElementDefaultWindowTitle>();  // Window title info
        app->addElement<RaytracingSample>();

        app->run();
        vkDeviceWaitIdle(app->getDevice());
        app.reset();

        return test->errorCode();
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
}

/**
 * FIXME - 记录编译问题
 * 有三个未解析的外部符号：
 * 1. ImGui::DockBuilderSplitNode(unsigned int, int, float, unsigned int*, unsigned int*)
 *    仅vcpkg版本包含此函数签名。在编译main.cpp.obj时，请使用nvpro版本的头文件。
 * 2. ImGui::GetForegroundDrawList(void)
 *    仅vcpkg版本包含此函数签名。在编译implot.cpp.obj时，请使用nvpro版本的头文件。
 * 3. ImGui::ArrowButtonEx(const char*, int, struct ImVec2, int)
 *    问题大概率与上述相同。
 */

/**NOTE - glsl编译成c风格的头文件cmd
 E:\VulkanSDK\1.3.280.0\Bin\glslangValidator.exe
 -g
 --target-env vulkan1.3
 --vn pathtrace_rgen
 -o .\spirv\generated_spirv\pathtrace_rgen.h
 .\pathtrace.rgen
 */