// c
#include <cstddef>
#include <cstdint>
#include <cstdlib>

// cpp
#include <array>
#include <chrono>
#include <exception>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

// 3rdparty
#define VK_ENABLE_BETA_EXTENSIONS
#include "heightmap_rtx/include/heightmap_rtx.h"
#include "imgui/backends/imgui_impl_vulkan.h"
#include "imgui/imgui_camera_widget.h"
#include "imgui/imgui_helper.h"
#include <glm/vec4.hpp>
#include <vulkan/vulkan_core.h>

// 3rdparty - nvvk
#include "nvvk/buffers_vk.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/shaders_vk.hpp"

// 3rdparty - nvvkhl
#define PROJECT_NAME "RaytraceDisplacement"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_testing.hpp"

// users

// global variables

/**
 * @brief 计时器
 *
 */
class Timer {
public:
    Timer() { init(""); }
    explicit Timer(std::string message) { init(std::move(message)); }

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

    void init(std::string message)
    {
        m_message   = std::move(message);
        m_startTime = std::chrono::system_clock::now();
    }
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
private:
};

class RaytracingSample : public nvvkhl::IAppElement {
public:
    RaytracingSample()           = default;
    ~RaytracingSample() override = default;

    void onAttach(nvvkhl::Application* app) override {}  // TODO -

    void onDetach() override {}  // TODO -

    void onResize(uint32_t width, uint32_t height) override {}  // TODO -

    void onUIRender() override {}  // TODO -

    void onRender(VkCommandBuffer cmd) override {}  // TODO -

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
        nvvk::Buffer vertices;
        nvvk::Buffer indices;
    };

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
