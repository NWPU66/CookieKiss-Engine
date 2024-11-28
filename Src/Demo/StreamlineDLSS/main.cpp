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
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// 3rdparty
#define GLFW_INCLUDE_VULKAN
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include <vulkan/vulkan_core.h>

static const std::string PROJECT_NAME = "StreamlineDLSS";

// 3rdparty - nvvk
#include "nvh/primitives.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/memallocator_vma_vk.hpp"

// 3rdparty - nvvkhl
#include "nvvkhl/appbase_vk.hpp"
#include "nvvkhl/gbuffer.hpp"

// users
#define SL_MANUAL_HOOKING true
#include "streamline_wrapper.hpp"
// NOTE - streamline_wrapper负责动态加载sl的函数，main.cpp不需要再include sl的头文件

// global variables
static const int SAMPLE_WIDTH  = 1920;
static const int SAMPLE_HEIGHT = 1080;
// Streamline features to enable
static const sl::Feature SL_FEATURES[] = {
    sl::kFeatureReflex,  // Reflex is required for DLSS Frame Generation
    sl::kFeatureDLSS,    // DLSS Super Resolution
    sl::kFeatureDLSS_G,  // DLSS Frame Generation
};

class StreamlineSample : public nvvkhl::AppBaseVk {
public:
    StreamlineSample()           = default;
    ~StreamlineSample() override = default;

    void create();  // TODO -

    void destroy() override;  // TODO -

    void onResize(int width, int height) override;  // TODO -

    void prepareFrame() override;  // TODO -

    void renderFrame();  // TODO -

    void drawGui();  // TODO -

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

    nvvk::Buffer m_frameInfo;

    VkPipeline m_scenePipeline = VK_NULL_HANDLE;
    VkPipeline m_postPipeline  = VK_NULL_HANDLE;

    std::unique_ptr<nvvkhl::GBuffer> m_gBuffers;
    nvvk::Image                      m_outputImage;
    VkImageView                      m_outputImageView = VK_NULL_HANDLE;

    float     m_animationTime     = 0.0F;
    float     m_animationTimePrev = 0.0F;
    glm::mat4 m_projMatrixPrev;
    glm::mat4 m_viewMatrixPrev;

    void createScene();  // TODO -

    void createPipelines();  // TODO -

    void createImages();  // TODO -

    void destroyScene();  // TODO -

    void destroyPipelines();  // TODO -

    void destroyImages();  // TODO -
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
VkSurfaceKHR createWindowSurface(nvvk::ContextCreateInfo& context, GLFWwindow* window)
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
        VkSurfaceKHR surface = createWindowSurface(contextInfo, window);

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
        // TODO -

        // clean up
        // TODO -
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
