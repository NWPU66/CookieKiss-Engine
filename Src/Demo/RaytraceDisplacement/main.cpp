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
#include "nvh/cameramanipulator.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>

// cpp
#include <array>
#include <chrono>
#include <exception>
#include <functional>
#include <glm/matrix.hpp>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vulkan/vulkan_beta.h>

// 3rdparty
#define VK_ENABLE_BETA_EXTENSIONS
#define VMA_IMPLEMENTATION
#include "heightmap_rtx/include/heightmap_rtx.h"
#include "imgui/backends/imgui_impl_vulkan.h"
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include "vma/include/vk_mem_alloc.h"
#include <glm/vec4.hpp>
#include <vulkan/vulkan_core.h>

// 3rdparty - nvvk
#include "nvh/primitives.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/commands_vk.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
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
#include "Shaders/device_host.h"
#include "raytracing_vk.hpp"

// global variables
constexpr std::string_view PROJECT_NAME = "RaytraceDisplacement";
#define HEIGHTMAP_RESOLUTION 256

/**
 * @brief 计时器
 *
 */
class Timer {
public:
    Timer() : Timer("") {}
    explicit Timer(std::string message)
        : m_message(std::move(message)), m_startTime(std::chrono::system_clock::now())
    {
    }

    ~Timer()
    {
        auto endTime = std::chrono::system_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(endTime - m_startTime);
        std::cout << "Timer(): " << m_message << " Time:  " << duration.count() / 1000.0f << " s\n";
    }

private:
    std::string                                        m_message;
    std::chrono::time_point<std::chrono::system_clock> m_startTime;
};

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
    void create() {}  // TODO -

    void dispatch() {}  // TODO -

    void destroy() {}  // TODO -

private:
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
    void create(nvvkhl::AllocVma& alloc, nvvk::DebugUtil& dutil, uint32_t resolution) {}  // TODO -

    void destroy() {}  // TODO -

    void clear(VkCommandBuffer cmd) {}  // TODO -

    void animate(VkCommandBuffer cmd) {}  // TODO -

    void height() {}  // TODO -

    void velocity() {}  // TODO -

    void setMouse() {}  // TODO -

private:
    void imageLayouts() {}  // TODO -

    void imageBarrier() {}  // TODO -

    void createHeightmaps() {}  // TODO -

    void destroyHeightmaps() {}  // TODO -
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

    void onUIRender() override  // TODO -
    {
        // Setting menu
        {
        }

        // Rendering Viewport
        {
        }

        // Heightmap preview and mouse interaction
        {
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
    std::vector<PrimitiveMeshVk> m_bMeshes;
    nvvk::Buffer                 m_bFrameInfo, m_bSkyParams;

    // Data and settings
    std::vector<nvh::PrimitiveMesh> m_meshes;
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

    void createScene() {}  // TODO -

    void createGbuffers(const glm::vec2& size) {}  // TODO -

    void createVkBuffers() {}  // TODO -

    void createHrtxPipeline() {}  // TODO -

    HrtxMap createHrtxMap(const VkAccelerationStructureGeometryKHR& geometry,
                          uint32_t                                  triangleCount,
                          const PrimitiveMeshVk&                    mesh,
                          const nvvk::Texture&                      texture,
                          VkCommandBuffer                           cmd)
    {
        // TODO -
    }

    void destroyHrtxMaps() {}  // TODO -

    void createBottomLevelAS(VkCommandBuffer cmd) {}  // TODO -

    void createTopLevelAS(VkCommandBuffer cmd) {}  // TODO -

    void createRtxPipeline() {}  // TODO -

    void writeRtDesc() {}  // TODO -

    void destroyResources() {}  // TODO -
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
