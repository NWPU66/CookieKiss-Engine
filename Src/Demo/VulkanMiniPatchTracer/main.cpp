// c
#include <cstdint>
#include <cstdlib>

// cpp
#include <array>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// 3rdparty
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include "third_party/tinyobjloader/tiny_obj_loader.h"

// 3rdparty - nvvk
#include <nvh/fileoperations.hpp>  // For nvh::loadFile
#include <nvvk/context_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>  // For nvvk::DescriptorSetContainer
#include <nvvk/error_vk.hpp>
#include <nvvk/raytraceKHR_vk.hpp>        // For nvvk::RaytracingBuilderKHR
#include <nvvk/resourceallocator_vk.hpp>  // For NVVK memory allocators
#include <nvvk/shaders_vk.hpp>            // For nvvk::createShaderModule

// users

static const std::string asset_folder     = "E:/Study/CodeProj/CookieKiss-Engine/Asset";
static const std::string output_file      = "output.hdr";
static const uint64_t    render_width     = 800;
static const uint64_t    render_height    = 600;
static const uint32_t    workgroup_width  = 16;
static const uint32_t    workgroup_height = 8;

VkCommandBuffer AllocateAndBeginOneTimeCommandBuffer(VkDevice device, VkCommandPool cmdPool)
{
    VkCommandBufferAllocateInfo cmdAllocInfo{
        .sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool        = cmdPool,
        .level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    VkCommandBuffer cmdBuffer;
    NVVK_CHECK(vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmdBuffer));
    VkCommandBufferBeginInfo beginInfo{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    NVVK_CHECK(vkBeginCommandBuffer(cmdBuffer, &beginInfo));
    return cmdBuffer;
}

void EndSubmitWaitAndFreeCommandBuffer(VkDevice         device,
                                       VkQueue          queue,
                                       VkCommandPool    cmdPool,
                                       VkCommandBuffer& cmdBuffer)
{
    NVVK_CHECK(vkEndCommandBuffer(cmdBuffer));
    VkSubmitInfo submitInfo{
        .sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers    = &cmdBuffer,
    };
    NVVK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
    NVVK_CHECK(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(device, cmdPool, 1, &cmdBuffer);
}

VkDeviceAddress GetBufferDeviceAddress(VkDevice device, VkBuffer buffer)
{
    VkBufferDeviceAddressInfo addressInfo{
        .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
        .buffer = buffer,
    };
    return vkGetBufferDeviceAddress(device, &addressInfo);
}

class MiniPatchTracer {
public:
    MiniPatchTracer(int argc, char** argv)
        : m_exePath(argv[0], std::string(argv[0]).find_last_of("/\\") + 1)
    {
        Init();
    }
    ~MiniPatchTracer() { Cleanup(); }

    void Run()
    {
        Prepocess();
        Rendering();
        Postprocess();
    }

private:
    struct Scene
    {
        std::unique_ptr<std::vector<float>>    vertices;
        std::unique_ptr<std::vector<uint32_t>> indices;
    };

    std::string                      m_exePath;
    nvvk::Context                    m_context;
    nvvk::ResourceAllocatorDedicated m_allocator;
    VkDeviceSize                     m_bufferSizeBytes;
    nvvk::Buffer                     m_buffer, m_vertexBuffer, m_indexBuffer;
    Scene                            m_scene;
    VkCommandPool                    m_cmdPool = nullptr;
    nvvk::RaytracingBuilderKHR       m_raytracingBuilder;
    nvvk::DescriptorSetContainer     m_descriptorSetContainer;
    VkShaderModule                   m_rayTraceModule;
    VkPipeline                       m_computePipeline;

    void Init()
    {
        // create vulkan context
        std::cout << "Creating Vulkan Context" << std::endl;

        nvvk::ContextCreateInfo info;
        info.setVersion(1, 3);

        // add extensions
        VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR,
        };
        VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{
            .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR,
        };
        info.addDeviceExtension(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
        info.addDeviceExtension(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, false, &asFeatures);
        info.addDeviceExtension(VK_KHR_RAY_QUERY_EXTENSION_NAME, false, &rayQueryFeatures);

        // create context and allocator
        m_context.init(info);
        /**FIXME - &info好像不行，要改成info
        const ContextCreateInfo& info = &info
        &info被评估为“true”，相当于用true创建一个ContextCreateInfo
        正好有适合的构造函数，它的bUseValidation = true。
        ContextCreateInfo(bool bUseValidation = true);

        隐式转化这种东西还挺麻烦的。。。。。。
         */
        m_allocator.init(m_context, m_context.m_physicalDevice);
    }

    void Prepocess()
    {
        // create a buffer
        std::cout << "Creating buffer" << std::endl;
        {
            m_bufferSizeBytes = render_width * render_height * 3 * sizeof(float);
            VkBufferCreateInfo bufferCreateInfo{
                .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                .size  = m_bufferSizeBytes,
                .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            };
            m_buffer = m_allocator.createBuffer(bufferCreateInfo,                           //
                                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT         //
                                                    | VK_MEMORY_PROPERTY_HOST_CACHED_BIT    //
                                                    | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT  //
            );
        }

        // Load mesh
        std::cout << "Loading mesh" << std::endl;
        {
            tinyobj::ObjReader reader;
            reader.ParseFromFile(nvh::findFile("CornellBox-Original-Merged.obj", {asset_folder}));
            if (!reader.Valid()) { throw std::runtime_error(reader.Error()); }

            m_scene.vertices =
                std::make_unique<std::vector<float>>(reader.GetAttrib().GetVertices());
            const auto& objShapes = reader.GetShapes();
            assert(objShapes.size() == 1);  // this file only contains one shape
            const auto& objShape = objShapes[0];

            m_scene.indices = std::make_unique<std::vector<uint32_t>>(objShape.mesh.indices.size());
            for (const auto& index : objShape.mesh.indices)
            {
                m_scene.indices->push_back(index.vertex_index);
            }

            // print info
            std::cout << "Mesh has " << m_scene.vertices->size() / 3 << " vertices" << std::endl;
            std::cout << "Mesh has " << m_scene.indices->size() / 3 << " triangles" << std::endl;
        }

        // create command pool
        std::cout << "Creating command pool" << std::endl;
        {
            VkCommandPoolCreateInfo cmdPoolInfo{
                .sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                .queueFamilyIndex = m_context.m_queueGCT,
            };
            NVVK_CHECK(vkCreateCommandPool(m_context.m_device, &cmdPoolInfo, nullptr, &m_cmdPool));
        }

        // Upload the vertex and index buffers to the GPU.
        std::cout << "Uploading vertex and index buffers" << std::endl;
        {
            auto uploadCmdBuffer =
                AllocateAndBeginOneTimeCommandBuffer(m_context.m_device, m_cmdPool);
            const VkBufferUsageFlags usage =
                VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |  //
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |         //
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
            m_vertexBuffer = m_allocator.createBuffer(uploadCmdBuffer, *m_scene.vertices, usage);
            m_indexBuffer  = m_allocator.createBuffer(uploadCmdBuffer, *m_scene.indices, usage);
            EndSubmitWaitAndFreeCommandBuffer(m_context.m_device, m_context.m_queueGCT, m_cmdPool,
                                              uploadCmdBuffer);
            m_allocator.finalizeAndReleaseStaging();
        }

        // BLAS
        std::cout << "Building bottom level acceleration structure (BLAS)" << std::endl;
        {
            // Describe the bottom-level acceleration structure (BLAS)
            nvvk::RaytracingBuilderKHR::BlasInput blas;
            // get device address
            auto vertexBufferAddress =
                GetBufferDeviceAddress(m_context.m_device, m_vertexBuffer.buffer);
            auto indexBufferAddress =
                GetBufferDeviceAddress(m_context.m_device, m_indexBuffer.buffer);
            // Fill the BLAS description
            VkAccelerationStructureGeometryTrianglesDataKHR triangles{
                .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                .vertexFormat  = VK_FORMAT_R32G32B32_SFLOAT,
                .vertexData    = {.deviceAddress = vertexBufferAddress},
                .vertexStride  = 3 * sizeof(float),
                .maxVertex     = static_cast<uint32_t>(m_scene.vertices->size() / 3 - 1),
                .indexType     = VK_INDEX_TYPE_UINT32,
                .indexData     = {.deviceAddress = indexBufferAddress},
                .transformData = {.deviceAddress = 0},  // No transform
            };
            // Create a VkAccelerationStructureGeometryKHR object that says it handles opaque
            // triangles and points to the above:
            VkAccelerationStructureGeometryKHR geometry{
                .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
                .geometry     = {.triangles = triangles},
                .flags        = VK_GEOMETRY_OPAQUE_BIT_KHR,
            };
            blas.asGeometry.push_back(geometry);
            // Create offset info that allows us to say how many triangles and vertices to read
            VkAccelerationStructureBuildRangeInfoKHR offsetInfo{
                .primitiveCount  = static_cast<uint32_t>(m_scene.indices->size() / 3),
                .primitiveOffset = 0,
                .firstVertex     = 0,
                .transformOffset = 0,
            };
            blas.asBuildOffsetInfo.push_back(offsetInfo);

            // create BLAS
            m_raytracingBuilder.setup(m_context.m_device, &m_allocator, m_context.m_queueGCT);
            m_raytracingBuilder.buildBlas(
                {blas}, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
        }

        // TLAS
        std::cout << "Building top level acceleration structure (TLAS)" << std::endl;
        {
            VkTransformMatrixKHR matrix;
            matrix.matrix[0][0] = matrix.matrix[1][1] = matrix.matrix[2][2] = 1.0f;

            VkAccelerationStructureInstanceKHR instance{
                .transform                              = matrix,
                .instanceCustomIndex                    = 0,
                .mask                                   = 0xFF,
                .instanceShaderBindingTableRecordOffset = 0,
                .flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR,
                .accelerationStructureReference = m_raytracingBuilder.getBlasDeviceAddress(0),
            };

            m_raytracingBuilder.buildTlas(
                {instance}, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR);
        }

        // create descriptor set
        std::cout << "Creating descriptor set" << std::endl;
        {
            // Here's the list of bindings for the descriptor set layout, from raytrace.comp.glsl :
            // 0 - a storage buffer (the buffer `buffer`)
            // 1 - an acceleration structure (the TLAS)
            // 2 - a storage buffer (the vertex buffer)
            // 3 - a storage buffer (the index buffer)

            m_descriptorSetContainer.init(m_context.m_device);
            m_descriptorSetContainer.addBinding(0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                VK_SHADER_STAGE_COMPUTE_BIT);
            m_descriptorSetContainer.addBinding(1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1,
                                                VK_SHADER_STAGE_COMPUTE_BIT);
            m_descriptorSetContainer.addBinding(2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                VK_SHADER_STAGE_COMPUTE_BIT);
            m_descriptorSetContainer.addBinding(3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                                VK_SHADER_STAGE_COMPUTE_BIT);

            m_descriptorSetContainer.initLayout();      // 描述符布局
            m_descriptorSetContainer.initPool(1);       // 描述符集池
            m_descriptorSetContainer.initPipeLayout();  // 管线布局
        }

        // Write values into the descriptor set.
        std::cout << "Write values into the descriptor set." << std::endl;
        {
            std::array<VkWriteDescriptorSet, 4> writeDescriptorSets{};

            // 0
            VkDescriptorBufferInfo descriptorBufferInfo{
                .buffer = m_buffer.buffer,
                .range  = m_bufferSizeBytes,
            };
            writeDescriptorSets[0] = m_descriptorSetContainer.makeWrite(
                0 /*set index*/, 0 /*binding*/, &descriptorBufferInfo);
            // 1
            auto* tlasCopy = m_raytracingBuilder.getAccelerationStructure();
            VkWriteDescriptorSetAccelerationStructureKHR descriptorAS{
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR,
                .accelerationStructureCount = 1,
                .pAccelerationStructures    = &tlasCopy,
            };
            writeDescriptorSets[1] = m_descriptorSetContainer.makeWrite(0, 1, &descriptorAS);
            // 2
            VkDescriptorBufferInfo vertexDescriptorBufferInfo{
                .buffer = m_vertexBuffer.buffer,
                .range  = VK_WHOLE_SIZE,
            };
            writeDescriptorSets[2] =
                m_descriptorSetContainer.makeWrite(0, 2, &vertexDescriptorBufferInfo);
            // 3
            VkDescriptorBufferInfo indexDescriptorBufferInfo{
                .buffer = m_indexBuffer.buffer,
                .range  = VK_WHOLE_SIZE,
            };
            writeDescriptorSets[3] =
                m_descriptorSetContainer.makeWrite(0, 3, &indexDescriptorBufferInfo);

            vkUpdateDescriptorSets(m_context.m_device,
                                   static_cast<uint32_t>(writeDescriptorSets.size()),
                                   writeDescriptorSets.data(), 0, nullptr);
        }

        // shader and pipline
        std::cout << "Creating pipeline" << std::endl;
        {
            // shader
            m_rayTraceModule = nvvk::createShaderModule(
                m_context.m_device,
                nvh::loadFile("Shaders/raytrace.comp.glsl.spv", true, {asset_folder}));
            VkPipelineShaderStageCreateInfo shaderStageCreateInfo{
                .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = m_rayTraceModule,
                .pName  = "main",
            };

            // compute pipline
            VkComputePipelineCreateInfo pipelineCreateInfo{
                .sType  = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                .stage  = shaderStageCreateInfo,
                .layout = m_descriptorSetContainer.getPipeLayout(),
            };
            NVVK_CHECK(vkCreateComputePipelines(m_context.m_device, VK_NULL_HANDLE, 1,
                                                &pipelineCreateInfo, nullptr, &m_computePipeline));
        }
    }

    void Rendering()
    {
        std::cout << "Start to Rendering" << std::endl;

        auto cmdBuffer = AllocateAndBeginOneTimeCommandBuffer(m_context.m_device, m_cmdPool);

        vkCmdBindPipeline(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_computePipeline);
        auto descriptorSet = m_descriptorSetContainer.getSet(0);
        vkCmdBindDescriptorSets(cmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                                m_descriptorSetContainer.getPipeLayout(), 0, 1, &descriptorSet, 0,
                                nullptr);
        /**NOTE - binding point 和 descriptor set
        Pipeline可以绑定多个descriptor set，在shader中使用set设置要访问的set
        每个set中的描述符自己设置绑定点，在shader中通过layout(binding = x)设置要访问的描述符
         */

        // Run the compute shader with enough workgroups to cover the entire buffer:
        // 类似CUDA的线程块
        vkCmdDispatch(
            cmdBuffer,
            (static_cast<uint32_t>(render_width) + workgroup_width - 1) / workgroup_width,
            (static_cast<uint32_t>(render_height) + workgroup_height - 1) / workgroup_height, 1);
        // 内存屏障
        VkMemoryBarrier memoryBarrier{
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
        };
        vkCmdPipelineBarrier(cmdBuffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &memoryBarrier, 0, nullptr, 0,
                             nullptr);

        EndSubmitWaitAndFreeCommandBuffer(m_context.m_device, m_context.m_queueGCT, m_cmdPool,
                                          cmdBuffer);
    }

    void Postprocess()
    {
        std::cout << "Postprocess: Save Image" << std::endl;

        // Get the image data back from the GPU
        void* data = m_allocator.map(m_buffer);
        // read the data from the pointer
        for (int i = 0; i < 10; i++)
        {
            for (int j = 0; j < 3; j++)
            {
                std::cout << reinterpret_cast<float*>(data)[i * 3 + j] << " ";
            }
            std::cout << std::endl;
        }
        stbi_write_hdr(output_file.c_str(), render_width, render_height, 3,
                       reinterpret_cast<float*>(data));
        m_allocator.unmap(m_buffer);
    }

    void Cleanup()
    {
        std::cout << "Cleanup" << std::endl;

        // 销毁管线和着色器模块
        vkDestroyPipeline(m_context.m_device, m_computePipeline, nullptr);
        vkDestroyShaderModule(m_context.m_device, m_rayTraceModule, nullptr);

        // 销毁描述符集
        m_descriptorSetContainer.deinit();

        // 销毁加速结构
        m_raytracingBuilder.destroy();

        // 销毁缓冲区
        m_allocator.destroy(m_vertexBuffer);
        m_allocator.destroy(m_indexBuffer);
        m_allocator.destroy(m_buffer);

        // 销毁命令池
        vkDestroyCommandPool(m_context, m_cmdPool, nullptr);

        // 清理分配器和上下文
        m_allocator.deinit();
        m_context.deinit();
    }
};

int main(int argc, char** argv)
{
    try
    {
        std::cout << "Hello VulkanMiniPatchTracer!" << std::endl;
        auto app = std::make_unique<MiniPatchTracer>(argc, argv);
        app->Run();
        std::cout << "Done !" << std::endl;
    }
    catch (std::exception& e)
    {
        std::cerr << "Exception: " << e.what() << std::endl;
    }
    return EXIT_SUCCESS;
}
