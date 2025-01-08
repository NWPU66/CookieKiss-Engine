// c/cpp
#define NOMINMAX  // NOTE - windows.h头文件中的min/max宏定义会与algorithm的std::min/max冲突
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>

// 3rdparty
#define GLM_ENABLE_EXPERIMENTAL
#include "imgui.h"
#include <fmt/core.h>
#include <glm/ext/matrix_transform.hpp>
#include <glm/ext/vector_float3.hpp>
#include <glm/glm.hpp>
#include <glm/trigonometric.hpp>
#include <memory>
#include <vector>
#include <vulkan/vulkan_core.h>

// 3rdparty - nvvk
#define PROJECT_NAME "SVGF"
#define VK_USE_PLATFORM_WIN32_KHR
#define VMA_IMPLEMENTATION
#include "nvh/cameramanipulator.hpp"
#include "nvvk/context_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/memallocator_vma_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
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
#include "Shader/common.h"
#include "Shader/spv/simpleFrag.h"
#include "Shader/spv/simpleVert.h"

// global variables
const std::string shaderFolder = "E:/Study/CodeProj/CookieKiss-Engine/Src/Demo/SVGF/Shader/spv";
const std::string meshFile =
    "E:/Study/CodeProj/CookieKiss-Engine/Asset/dae_diorama_rustborn/scene.gltf";
const std::string assetFolder = "E:/Study/CodeProj/CookieKiss-Engine/Asset";

class SVGFElement : public nvvkhl::IAppElement {
    // TODO -

public:
    void onAttach(nvvkhl::Application* app) override {}
    void onDetach() override {}
    void onResize(uint32_t width, uint32_t height) override {}
    void onUIRender() override {}
    void onUIMenu() override {}
    void onRender(VkCommandBuffer cmd) override {}

private:
};

class SimpleGraphicsElement : public nvvkhl::IAppElement {
public:
    void onAttach(nvvkhl::Application* app) override
    {
        std::cout << "SimpleGraphicsElement onAttach()" << std::endl;
        m_app       = app;
        m_allocator = std::make_unique<nvvk::ResourceAllocatorVma>(
            m_app->getInstance(), m_app->getDevice(), m_app->getPhysicalDevice());
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
        auto p = glm::perspectiveFovLH_ZO(glm::radians(90.0f), (float)size.width,
                                          (float)size.height, 0.001f, 100.0f);
        p[1][1] *= -1;
        auto v  = CameraManip.getMatrix();
        auto vp = p * v;

        // binding vertex buffer
        const auto& nodes            = m_scene.getRenderNodes();
        const auto& VBOs             = m_VKScene->vertexBuffers();
        const auto& EBOs             = m_VKScene->indices();
        const auto& renderPrimitives = m_scene.getRenderPrimitives();
        for (size_t i = 0; i < nodes.size(); i++)
        {
            auto&       node = nodes[i];
            const auto& VBO  = VBOs[node.renderPrimID];
            const auto& EBO  = EBOs[node.renderPrimID];
            const auto& prim = renderPrimitives[node.renderPrimID];

            // binding vertex buffer
            VkBuffer     buffers[] = {VBO.position.buffer};
            VkDeviceSize offsets[] = {0};
            vkCmdBindVertexBuffers(cmd, 0, std::size(buffers), buffers, offsets);
            // NOTE - 一部分buffer为空，例如tangent、uv1和color buffer
            // FIXME - gltf_scene_vk.hpp中的normal、uv buffer没有VK_VERTEX_SHADER_STAGE_FLAG_BIT，
            // 无法作为顶点属性，把他们作为管线的外部描述符接入，用gl_VertexIndex进行索引

            // binding index buffer
            vkCmdBindIndexBuffer(cmd, EBO.buffer, 0, VK_INDEX_TYPE_UINT32);

            auto        m = node.worldMatrix;
            PushContent pc{.mvp = vp * m};
            vkCmdPushConstants(cmd, m_pipelineLayout,
                               VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                               sizeof(PushContent), &pc);

            // binding desc set
            auto pDescriptorSets = m_descriptorSetContainer.getSet(i);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipelineLayout, 0, 1,
                                    &pDescriptorSets, 0, nullptr);

            // draw
            vkCmdDrawIndexed(cmd, prim.indexCount, 1, 0, 0, 0);

            // FIXME - 坐标转换很怪，检查一下投影变换吧
        }

        vkCmdEndRendering(cmd);
    }

private:
    nvvkhl::Application*                     m_app;
    std::unique_ptr<nvvk::ResourceAllocator> m_allocator;

    // resource
    nvvk::DescriptorSetContainer     m_descriptorSetContainer;
    std::unique_ptr<nvvkhl::GBuffer> m_gbuffer;

    // Geometry
    std::unique_ptr<nvvkhl::SceneVk> m_VKScene;
    nvh::gltf::Scene                 m_scene;

    // pipeline
    //  nvvk::ShaderModuleManager   m_shaderManager;
    VkShaderModule              m_vertexShader;
    VkShaderModule              m_fragmentShader;
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
        // add pipeline descriptor
        m_descriptorSetContainer.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                            VK_SHADER_STAGE_VERTEX_BIT);  // normal buffer
        m_descriptorSetContainer.addBinding(1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                            VK_SHADER_STAGE_VERTEX_BIT);  // texcoords buffer
        m_descriptorSetContainer.initLayout();
        auto& nodes = m_scene.getRenderNodes();
        m_descriptorSetContainer.initPool(nodes.size());

        // update pipeline descriptor
        const auto& VBOs = m_VKScene->vertexBuffers();
        for (size_t i = 0; i < nodes.size(); i++)
        {
            auto&       node = nodes[i];
            const auto& VBO  = VBOs[node.renderPrimID];

            VkDescriptorBufferInfo normalBufferInfo{
                .buffer = VBO.normal.buffer,
                .offset = 0,
                .range  = VK_WHOLE_SIZE,
            };
            VkDescriptorBufferInfo texcoordsBufferInfo{
                .buffer = VBO.texCoord0.buffer,
                .offset = 0,
                .range  = VK_WHOLE_SIZE,
            };
            VkWriteDescriptorSet writeDescriptorSets[] = {
                m_descriptorSetContainer.makeWrite(i, 0, &normalBufferInfo, 1),
                m_descriptorSetContainer.makeWrite(i, 1, &texcoordsBufferInfo, 1),
            };

            vkUpdateDescriptorSets(m_app->getDevice(), std::size(writeDescriptorSets),
                                   writeDescriptorSets, 0, nullptr);
        }
        // NOTE - 描述符集的设置，为每一个要渲染的节点创建一个描述符集
        // 例如有15个渲染节点，每次渲染需要一个normal buffer和一个texcoords
        // buffer，则创建15个描述符集，每个描述符集包含两个描述符。
        // 这15个描述符集的布局、管线布局相同。

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

        // Shaders
        m_vertexShader =
            nvvk::createShaderModule(m_app->getDevice(), (char*)simpleVert, std::size(simpleVert));
        m_fragmentShader =
            nvvk::createShaderModule(m_app->getDevice(), (char*)simpleFrag, std::size(simpleFrag));

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
                .stride    = sizeof(glm::vec3),  // position
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            },
        });
        m_pipelineState.addAttributeDescriptions({
            VkVertexInputAttributeDescription{
                .location = 0,
                .binding  = 0,
                .format   = VK_FORMAT_R32G32B32_SFLOAT,
                .offset   = 0,
            },
        });
        // NOTE - GraphicsPipelineState初始化一些常见的管线选项，比如动态视口大小
        // 所以在onRender函数中要手动调用vkCmdSetViewport

        // pipeline generator
        nvvk::GraphicsPipelineGenerator generator(m_app->getDevice(), m_pipelineLayout,
                                                  renderingCreateInfo, m_pipelineState);
        generator.addShader(m_vertexShader, VK_SHADER_STAGE_VERTEX_BIT);
        generator.addShader(m_fragmentShader, VK_SHADER_STAGE_FRAGMENT_BIT);
        m_pipeline = generator.createPipeline();
        // NOTE - 缺少Shader，shaderc预编译的库有点问题，用"nvvk/shader_vk.hpp"代替
    }

    void loadScene()
    {
        // load scene
        if (!m_scene.load(meshFile)) { std::cerr << "Error loading scene" << std::endl; }

        // create vkScene
        auto cmd  = m_app->createTempCmdBuffer();
        m_VKScene = std::make_unique<nvvkhl::SceneVk>(
            m_app->getDevice(), m_app->getPhysicalDevice(), m_allocator.get());
        m_VKScene->create(cmd, m_scene);
        m_app->submitAndWaitTempCmdBuffer(cmd);
    }

    void prepareCamera()
    {
        CameraManip.setWindowSize(1920, 1080);
        CameraManip.setLookat(glm::vec3(0, 0, 1), glm::vec3(0), glm::vec3(0, 1, 0));
    }

    void processEvent()
    {
        if (ImGui::GetIO().MouseDown[0])
        {
            auto [x, y] = ImGui::GetIO().MousePos;
            CameraManip.setMousePosition(x, y);
        }

        // TODO -
    }

    void destroyResources()
    {
        m_gbuffer->destroy();
        m_gbuffer.reset();
    }

    void destroyGraphicsPipeline() { vkDestroyPipeline(m_app->getDevice(), m_pipeline, nullptr); }

    void destoryScene()
    {
        m_VKScene.reset();
        m_scene.destroy();
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
