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
#include "backends/imgui_impl_glfw.h"
#include "glm/glm.hpp"
#include <vulkan/vulkan_core.h>

static const std::string PROJECT_NAME = "StreamlineDLSS";

// 3rdparty - nvvk
#define VMA_IMPLEMENTATION
#include "nvh/cameramanipulator.hpp"
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
#include "Shaders/common.h"

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

        createScene();
        createPipelines();
        createImages(getSize());  // FIXME - getSize()这个函数找不到
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

        m_animationTimePrev = m_animationTime;
        m_animationTime += ImGui::GetIO().DeltaTime;
    }

    void renderFrame(VkCommandBuffer cmd, sl::FrameToken* frame)
    {
        // TODO -
    }

    void drawGui()
    {
        // TODO -
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
        // TODO -
    }

    void createImages(const VkExtent2D& size)
    {
        // TODO -
    }

    void destroyScene()
    {
        // TODO -
    }

    void destroyPipelines()
    {
        // TODO -
    }

    void destroyImages()
    {
        // TODO -
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
