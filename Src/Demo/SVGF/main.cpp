// c/cpp
#include "nvh/primitives.hpp"
#include <cstddef>
#include <iterator>
#define NOMINMAX  // NOTE - windows.h头文件中的min/max宏定义会与algorithm的std::min/max冲突
#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iostream>
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

// 3rdparty - nvvk
#define PROJECT_NAME "SVGF"
#define VK_USE_PLATFORM_WIN32_KHR
#define VMA_IMPLEMENTATION
#define NVP_SUPPORTS_SHADERC true
#include "nvh/cameramanipulator.hpp"
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
#define LOAD_METHOD_TINYOBJLOADER
#define TINYOBJLOADER_IMPLEMENTATION
#include "AssetLoader.h"
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
        m_allocator = std::make_unique<nvvk::ResourceAllocatorVma>(
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
        VkRect2D renderArea{
            .offset = {0, 0},
            .extent = m_gbuffer->getSize(),
        };
        nvvk::createRenderingInfo renderingInfo(renderArea, {m_gbuffer->getColorImageView()},
                                                m_gbuffer->getDepthImageView());
        vkCmdBeginRendering(cmd, (VkRenderingInfoKHR*)&renderingInfo);

        // 手动设置viewport和scissor
        VkExtent2D size = m_gbuffer->getSize();
        VkViewport viewport{
            .x        = 0.0f,
            .y        = 0.0f,
            .width    = (float)size.width,
            .height   = (float)size.height,
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        };
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor{
            .offset = {0, 0},
            .extent = {size.width, size.height},
        };
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // binding pipeline
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

        // pre-calculate v and p
        auto p = glm::perspectiveFovRH_ZO(glm::radians(45.0f), (float)size.width,
                                          (float)size.height, 0.001f, 100.0f);
        p[1][1] *= -1;
        auto v  = CameraManip.getMatrix();
        auto vp = p * v;
        /*REVIEW - 我还是不理解这里为什么用 RH
        vulkan的NDC空间就是右手系的，p[1][1]乘上-1后会变成左手系
        */

        // rendering instances
        for (auto& inst : m_instances)
        {
            auto& prim = m_primities[inst.mesh];
            auto& VBO  = m_sceneDataGPU[inst.mesh].vertexBuffer;
            auto& EBO  = m_sceneDataGPU[inst.mesh].indexBuffer;

            // binding descriptor sets
            VkDescriptorSet descSet[] = {m_descriptorSetContainer.getSet()};
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0,
                                    std::size(descSet), descSet, 0, nullptr);

            // binding vertex buffer
            VkDeviceSize offsets[] = {0};
            vkCmdBindIndexBuffer(cmd, EBO.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdBindVertexBuffers(cmd, 0, std::size(offsets), &VBO.buffer, offsets);

            // push constant
            auto        model = inst.localMatrix();
            PushContent pc{.m = model, .mvp = vp * model};
            vkCmdPushConstants(cmd, m_pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(pc), &pc);

            // draw
            vkCmdDrawIndexed(cmd, prim.triangles.size() * 3, 1, 0, 0, 0);
        }

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
            */

        vkCmdEndRendering(cmd);
    }

private:
    struct SceneDataGPU
    {
        nvvk::Buffer vertexBuffer;
        nvvk::Buffer indexBuffer;
    };

    nvvkhl::Application*                     m_app;
    std::unique_ptr<nvvk::ResourceAllocator> m_allocator;
    nvvk::DebugUtil                          m_debug;

    // resource
    nvvk::DescriptorSetContainer     m_descriptorSetContainer;
    std::unique_ptr<nvvkhl::GBuffer> m_gbuffer;

    // Geometry
    std::vector<nvh::PrimitiveMesh> m_primities;
    std::vector<nvh::Node>          m_instances;
    std::vector<SceneDataGPU>       m_sceneDataGPU;

    // pipeline
    nvvk::ShaderModuleManager   m_shaderManager;
    nvvk::GraphicsPipelineState m_pipelineState;
    VkPipelineLayout            m_pipelineLayout;
    VkPipeline                  m_pipeline;

    void createResources()
    {
        std::cout << "createResources()" << std::endl;

        // gbuffer
        m_gbuffer = std::make_unique<nvvkhl::GBuffer>(m_app->getDevice(), m_allocator.get());
        m_gbuffer->create(VkExtent2D{1920, 1080}, {VK_FORMAT_R32G32B32A32_SFLOAT},
                          VK_FORMAT_D24_UNORM_S8_UINT);
        // NOTE - m_app->getViewportSize() 拿到的是m_viewportSize
        // 这个东西只在viewport改变的时候被赋值，初值是{0, 0}

        // descriptor and pipeline layout
        m_descriptorSetContainer.init(m_app->getDevice());
        m_descriptorSetContainer.initLayout();
        m_descriptorSetContainer.initPool(1);

        /*NOTE - 讲一下vu来看中descriptor的结构
        最大的是desc pool，只有它持有资源，其他对象只持有应用。
        一个pool中包含多个desc set，每个set的布局相同。
        一个desc set中包含多个绑定点，每个绑定点指明一项描述符类型。
        一个绑定点中可以包含多个相同类型的desc，它们组成一个数组，在shader中通过数组索引来访问。
        */

        // 管线布局
        VkPushConstantRange pushConstantRange{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset     = 0,
            .size       = sizeof(PushContent),
        };
        m_pipelineLayout = m_descriptorSetContainer.initPipeLayout(1, &pushConstantRange);
    }

    void createGraphicsPipeline()
    {
        std::cout << "createGraphicsPipeline()" << std::endl;

        // SMM
        m_shaderManager.init(m_app->getDevice());
        m_shaderManager.addDirectory(shaderFolder);
        auto vid = m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT,
                                                      "SimpleGraphicsTest.vert");
        auto fid = m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT,
                                                      "SimpleGraphicsTest.frag");

        // renderingCreate
        VkFormat                      colorAttachmentFormat = VK_FORMAT_R32G32B32A32_SFLOAT;
        VkPipelineRenderingCreateInfo renderingCreateInfo{
            .sType                   = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
            .pNext                   = nullptr,
            .colorAttachmentCount    = 1,
            .pColorAttachmentFormats = &colorAttachmentFormat,
            .depthAttachmentFormat   = VK_FORMAT_D24_UNORM_S8_UINT,
            .stencilAttachmentFormat = VK_FORMAT_D24_UNORM_S8_UINT,
        };

        // pipelineState
        m_pipelineState.addBindingDescriptions({
            VkVertexInputBindingDescription{
                .binding   = 0,
                .stride    = sizeof(nvh::PrimitiveVertex),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            },

        });
        m_pipelineState.addAttributeDescriptions({
            VkVertexInputAttributeDescription{
                .location = 0,
                .binding  = 0,
                .format   = VK_FORMAT_R32G32B32_SFLOAT,  // postion
                .offset   = offsetof(nvh::PrimitiveVertex, p),
            },
            VkVertexInputAttributeDescription{
                .location = 1,
                .binding  = 0,
                .format   = VK_FORMAT_R32G32B32_SFLOAT,  // normal
                .offset   = offsetof(nvh::PrimitiveVertex, n),
            },
            VkVertexInputAttributeDescription{
                .location = 2,
                .binding  = 0,
                .format   = VK_FORMAT_R32G32_SFLOAT,  // texcoords
                .offset   = offsetof(nvh::PrimitiveVertex, t),
            },
        });
        // m_pipelineState.rasterizationState.frontFace = VK_FRONT_FACE_CLOCKWISE;
        m_pipelineState.rasterizationState.cullMode  = VK_CULL_MODE_NONE;
        // NOTE - GraphicsPipelineState初始化一些常见的管线选项，比如动态视口大小
        // 所以在onRender函数中要手动调用vkCmdSetViewport

        // pipeline generator
        nvvk::GraphicsPipelineGenerator generator(m_app->getDevice(), m_pipelineLayout,
                                                  renderingCreateInfo, m_pipelineState);
        generator.addShader(m_shaderManager.get(vid), VK_SHADER_STAGE_VERTEX_BIT);
        generator.addShader(m_shaderManager.get(fid), VK_SHADER_STAGE_FRAGMENT_BIT);
        m_pipeline = generator.createPipeline();
    }

    void loadScene()
    {
        // create scene primitives
        m_primities = cookiekiss::loadGeometryFromFile(meshFile);

        // create scene instances
        m_instances.reserve(m_primities.size());
        for (int i = 0; i < m_primities.size(); i++)
        {
            m_instances.push_back(nvh::Node{.mesh = i});
        }

        // upload scene data to gpu
        auto cmd = m_app->createTempCmdBuffer();
        m_sceneDataGPU.reserve(m_primities.size());
        for (auto& prim : m_primities)
        {
            auto vertexBuffer = m_allocator->createBuffer(
                cmd, prim.vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            auto indexBuffer = m_allocator->createBuffer(
                cmd, prim.triangles, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            m_sceneDataGPU.emplace_back(vertexBuffer, indexBuffer);
        }
        m_app->submitAndWaitTempCmdBuffer(cmd);
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

    void destroyGraphicsPipeline() { vkDestroyPipeline(m_app->getDevice(), m_pipeline, nullptr); }

    void destoryScene()
    {
        for (auto& sceneData : m_sceneDataGPU)
        {
            m_allocator->destroy(sceneData.vertexBuffer);
            m_allocator->destroy(sceneData.indexBuffer);
        }
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
