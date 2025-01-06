// c/cpp
#define NOMINMAX  // NOTE - windows.h头文件中的min/max宏定义会与algorithm的std::min/max冲突
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>

// 3rdparty
#include <fmt/core.h>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <vulkan/vulkan_core.h>

// 3rdparty - nvvk
#define PROJECT_NAME "SVGF"
#define VK_USE_PLATFORM_WIN32_KHR
#define VMA_IMPLEMENTATION
#include "nvvk/context_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/memallocator_vma_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/renderpasses_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/element_camera.hpp"
#include "nvvkhl/element_gui.hpp"
#include "nvvkhl/element_testing.hpp"
#include "nvvkhl/gbuffer.hpp"

// users

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
        createResources();
        createGraphicsPipeline();
    }

    void onDetach() override
    {
        std::cout << "SimpleGraphicsElement onDetach()" << std::endl;
        destroyGraphicsPipeline();
        destroyResources();
        m_allocator->deinit();
        m_allocator.reset();
    }

    void onResize(uint32_t width, uint32_t height) override
    {
        // recreate gbuffer
        m_gbuffer->destroy();
        m_gbuffer->create(VkExtent2D{width, height}, {VK_FORMAT_R32G32B32A32_SFLOAT},
                          VK_FORMAT_D24_UNORM_S8_UINT);
    }

    void onUIRender() override
    {
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

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
        vkCmdBindVertexBuffers(cmd, 0, 1, &m_vertexBuffer.buffer, nullptr);
        vkCmdBindIndexBuffer(cmd, m_elementBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cmd, 6, 1, 0, 0, 0);

        vkCmdEndRendering(cmd);
    }

private:
    struct VertexInput
    {
        glm::vec2 position;
        glm::vec3 vertexColor;
    };
    const std::vector<VertexInput> m_vertices = {
        VertexInput{{1, 1}, {1, 1, 0}},
        VertexInput{{1, -1}, {1, 0, 0}},
        VertexInput{{-1, 1}, {0, 1, 0}},
        VertexInput{{-1, -1}, {0, 0, 0}},
    };
    const std::vector<uint32_t> m_vertexIndex = {
        0, 1, 2,  //
        0, 2, 3,  //
    };

    nvvkhl::Application* m_app;

    std::unique_ptr<nvvk::ResourceAllocator> m_allocator;
    nvvk::DescriptorSetContainer             m_descriptorSetContainer;
    nvvk::Buffer                             m_vertexBuffer;
    nvvk::Buffer                             m_elementBuffer;
    std::unique_ptr<nvvkhl::GBuffer>         m_gbuffer;

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
        m_pipelineLayout = m_descriptorSetContainer.initPipeLayout();

        // vertex buffer
        auto* tempBuffer = m_app->createTempCmdBuffer();
        m_vertexBuffer   = m_allocator->createBuffer(
            tempBuffer, m_vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        m_elementBuffer = m_allocator->createBuffer(
            tempBuffer, m_vertexIndex, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        m_app->submitAndWaitTempCmdBuffer(tempBuffer);
    }

    void createGraphicsPipeline()
    {
        std::cout << "createGraphicsPipeline()" << std::endl;

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
        m_pipelineState.addBindingDescription(VkVertexInputBindingDescription{
            .binding   = 0,
            .stride    = sizeof(VertexInput),  // vertex pos 2 + vertex color 3
            .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
        });
        m_pipelineState.addAttributeDescriptions({
            VkVertexInputAttributeDescription{
                .location = 0,
                .binding  = 0,
                .format   = VK_FORMAT_R32G32_SFLOAT,
                .offset   = offsetof(VertexInput, position),
            },
            VkVertexInputAttributeDescription{
                .location = 1,
                .binding  = 0,
                .format   = VK_FORMAT_R32G32B32_SFLOAT,
                .offset   = offsetof(VertexInput, vertexColor),
            },
        });

        // shader modules

        // pipeline generator
        nvvk::GraphicsPipelineGenerator generator(m_app->getDevice(), m_pipelineLayout,
                                                  renderingCreateInfo, m_pipelineState);
        m_pipeline = generator.createPipeline();
        // FIXME -
    }

    void destroyResources()
    {
        // buffer
        m_allocator->destroy(m_vertexBuffer);
        m_allocator->destroy(m_elementBuffer);
        m_gbuffer->destroy();
        m_gbuffer.reset();
    }

    void destroyGraphicsPipeline() { vkDestroyPipeline(m_app->getDevice(), m_pipeline, nullptr); }
};

int main(int argc, char** argv)
{
    // init context
    nvvk::ContextCreateInfo contextInfo;
    contextInfo.setVersion(1, 3);
    contextInfo.addDeviceExtension(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    nvvkhl::addSurfaceExtensions(contextInfo.instanceExtensions);
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
