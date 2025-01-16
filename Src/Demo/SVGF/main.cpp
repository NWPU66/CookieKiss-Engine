// c/cpp
#include <array>
#include <assimp/material.h>
#define NOMINMAX  // NOTE - windows.h头文件中的min/max宏定义会与algorithm的std::min/max冲突
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <iterator>
#include <set>
#include <string>

// 3rdparty
#define GLM_ENABLE_EXPERIMENTAL
#include "imgui.h"
#include <fmt/core.h>
#include <glm/ext/matrix_transform.hpp>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <vulkan/vulkan_core.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

// 3rdparty - nvvk
#define PROJECT_NAME "SVGF"
#define VK_USE_PLATFORM_WIN32_KHR
#define VMA_IMPLEMENTATION
#define NVP_SUPPORTS_SHADERC true
#include "nvh/cameramanipulator.hpp"
#include "nvh/primitives.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/debug_util_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/memallocator_vma_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/shadermodulemanager_vk.hpp"
#include "nvvk/shaders_vk.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_testing.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "nvvkhl/gltf_scene_vk.hpp"

#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "tiny_gltf.h"
// NOTE - redefination problem
// define all the "tiny_gltf.h" implementation after "gltf_scene_vk.hpp"

// users
// #define LOAD_METHOD_TINYOBJLOADER
// #define TINYOBJLOADER_IMPLEMENTATION
// #include "AssetLoader.h"
#include "Shader/common.h"

// global variables
const std::string shaderFolder = "E:/Study/CodeProj/CookieKiss-Engine/Src/Demo/SVGF/Shader";
// const std::string meshFile =
//     "E:/Study/CodeProj/CookieKiss-Engine/Asset/dae_diorama_rustborn/scene.gltf";
const std::string meshFile    = "E:/Study/CodeProj/CookieKiss-Engine/Asset/cube/cube.obj";
const std::string assetFolder = "E:/Study/CodeProj/CookieKiss-Engine/Asset";

class SVGFElement : public nvvkhl::IAppElement {};

class SimpleGraphicsElement : public nvvkhl::IAppElement {
public:
    void onAttach(nvvkhl::Application* app) override
    {
        std::cout << "SimpleGraphicsElement onAttach()" << std::endl;
        m_app       = app;
        m_allocator = std::make_shared<nvvk::ResourceAllocatorVma>(
            m_app->getInstance(), m_app->getDevice(), m_app->getPhysicalDevice());
        m_debug.setup(m_app->getDevice());
        loadScene();
        createResources();  // 创建描述符集要在加载场景后
        prepareCamera();
        createGraphicsPipeline();
    }

    void onDetach() override
    {
        std::cout << "SimpleGraphicsElement onDetach()" << std::endl;
        destroyGraphicsPipeline();
        destoryScene();
        destroyResources();
        m_allocator->deinit();
        m_allocator.reset();
    }

    void onResize(uint32_t width, uint32_t height) override
    {
        vkDeviceWaitIdle(m_app->getDevice());

        // reset camera
        CameraManip.setWindowSize(width, height);

        // recreate gbuffer
        m_gbuffer->destroy();
        m_gbuffer->create(VkExtent2D{width, height}, {VK_FORMAT_R32G32B32A32_SFLOAT},
                          VK_FORMAT_D24_UNORM_S8_UINT);
    }

    void onUIRender() override
    {
        processEvent();

        ImGui::Begin("Viewport");
        ImGui::Image(m_gbuffer->getDescriptorSet(), ImGui::GetContentRegionAvail());
        ImGui::End();
    }

    void onRender(VkCommandBuffer cmd) override
    {
        // TODO - 重构

        /*FIXME - 现在出现的一些问题：
            1. 坐标转换很怪，检查一下投影变换吧：
            好像就是视场角的问题，改成45度就好很多了

            2. 顺时针是正面，设置一下绘制正面的规则：
            用Blender看了面朝向，就是这么变态，所以我只好设置成不剔除背面

            3. 鼠标和键盘操纵摄像机的逻辑有问题，操作很反人类

            4. 搞一下即时编译shader，现在手动编译太麻烦了
            修好了，we use release shaderc_shared on Windows and release shaderc_combined on Linux.

            5. normal 和 texcoord 明显有问题
            36顶点只有18法线而且有些法线明显有问题
            载入模型的问题解决了
            */
    }

private:
    nvvkhl::Application*                     m_app;
    std::shared_ptr<nvvk::ResourceAllocator> m_allocator;
    nvvk::DebugUtil                          m_debug;

    // resource
    nvvk::DescriptorSetContainer     m_descriptorSetContainer;
    std::unique_ptr<nvvkhl::GBuffer> m_gbuffer;

    // Geometry
    // TODO -

    // pipeline
    nvvk::ShaderModuleManager   m_shaderManager;
    nvvk::GraphicsPipelineState m_pipelineState;
    VkPipelineLayout            m_pipelineLayout;
    VkPipeline                  m_pipeline;

    void createResources()
    {
        // gbuffer
        m_gbuffer = std::make_unique<nvvkhl::GBuffer>(m_app->getDevice(), m_allocator.get());
        m_gbuffer->create(VkExtent2D{1920, 1080}, {VK_FORMAT_R32G32B32A32_SFLOAT},
                          VK_FORMAT_D24_UNORM_S8_UINT);
        // NOTE - m_app->getViewportSize() 拿到的是m_viewportSize
        // 这个东西只在viewport改变的时候被赋值，初值是{0, 0}

        /*NOTE - 讲一下vu来看中descriptor的结构
        最大的是desc pool，只有它持有资源，其他对象只持有应用。
        一个pool中包含多个desc set，每个set的布局相同。
        一个desc set中包含多个绑定点，每个绑定点指明一项描述符类型。
        一个绑定点中可以包含多个相同类型的desc，它们组成一个数组，在shader中通过数组索引来访问。
        */

        // TODO -
    }

    void createGraphicsPipeline()
    {
        // TODO -
    }

    void loadScene()
    {  // TODO -
    }

    void prepareCamera()
    {
        CameraManip.setWindowSize(1920, 1080);
        CameraManip.setLookat(glm::vec3(0, 0, -1), glm::vec3(0), glm::vec3(0, 1, 0));
        CameraManip.setMode(nvh::CameraManipulator::Modes::Fly);
    }

    void processEvent()
    {
        if (ImGui::IsMouseDown(ImGuiMouseButton_::ImGuiMouseButton_Left))
        {
            std::cout << "Down" << std::endl;
        }
        if (ImGui::IsMouseClicked(ImGuiMouseButton_::ImGuiMouseButton_Left))
        {
            std::cout << "Clicked" << std::endl;
        }
        if (ImGui::IsMouseDragging(ImGuiMouseButton_::ImGuiMouseButton_Left))
        {
            std::cout << "Dragging" << std::endl;
        }
        /*NOTE - IsMouseDown、IsMouseClicked、IsMouseDragging
        IsMouseClicked只在按下时触发一次
        IsMouseDown在按下时持续触发
        IsMouseDragging在按下并拖动时持续触发
        */
    }

    void destroyResources()
    {
        m_gbuffer->destroy();
        m_gbuffer.reset();
    }

    void destroyGraphicsPipeline()
    {
        // TODO -
    }

    void destoryScene()
    {
        // TODO -
    }
};

int main(int argc, char** argv)
{
    // init context
    nvvk::ContextCreateInfo contextInfo;
    contextInfo.setVersion(1, 3);
    contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    nvvkhl::addSurfaceExtensions(contextInfo.instanceExtensions);
    contextInfo.addDeviceExtension(VK_NV_RAY_TRACING_EXTENSION_NAME);
    contextInfo.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
    contextInfo.addDeviceExtension(
        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);  // required by
                                                          // VK_KHR_acceleration_structure
    nvvk::Context context;
    context.init(contextInfo);

    // init application
    nvvkhl::ApplicationCreateInfo appInfo{
        .name   = PROJECT_NAME,
        .width  = 1920,
        .height = 1080,
        // vulkan context
        .instance       = context.m_instance,
        .device         = context.m_device,
        .physicalDevice = context.m_physicalDevice,
        .queues =
            {
                nvvkhl::ApplicationQueue{
                    context.m_queueGCT.familyIndex,
                    context.m_queueGCT.queueIndex,
                    context.m_queueGCT.queue,
                },
                nvvkhl::ApplicationQueue{
                    context.m_queueC.familyIndex,
                    context.m_queueC.queueIndex,
                    context.m_queueC.queue,
                },
                nvvkhl::ApplicationQueue{
                    context.m_queueT.familyIndex,
                    context.m_queueT.queueIndex,
                    context.m_queueT.queue,
                },
            },
    };
    nvvkhl::Application app(appInfo);

    // Add all application elements
    auto test = std::make_shared<nvvkhl::ElementTesting>(argc, argv);
    app.addElement(test);  // test framework
    app.addElement<nvvkhl::ElementCamera>();
    app.addElement<nvvkhl::ElementDefaultMenu>();         // Menu / Quit
    app.addElement<nvvkhl::ElementDefaultWindowTitle>();  // Window title info
    app.addElement<SimpleGraphicsElement>();

    // run
    app.run();
    vkDeviceWaitIdle(app.getDevice());
    return test->errorCode();
}
