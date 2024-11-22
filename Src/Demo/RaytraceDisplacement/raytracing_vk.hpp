#pragma once

// c

// cpp
#include <algorithm>
#include <functional>
#include <memory>
#include <optional>
#include <span>

// 3rdparty
#include <vulkan/vulkan_core.h>

// 3rdparty - nvvk
#include "nvvk/resourceallocator_vk.hpp"
#include <nvvk/memallocator_vk.hpp>

// users

namespace rt {

struct Context
{
    VkDevice                      device = VK_NULL_HANDLE;
    nvvk::ResourceAllocator*      allocator;
    const VkAllocationCallbacks*  allocationCallbacks = nullptr;
    std::function<void(VkResult)> resultCallback;
};

struct AccelerationStructureInput
{
    VkAccelerationStructureTypeKHR                        type;
    VkBuildAccelerationStructureFlagsKHR                  flags;
    std::vector<VkAccelerationStructureGeometryKHR>       geometries;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> rangeInfos;
};

class AccelerationStructureSizes {
public:
    AccelerationStructureSizes(const Context& context, const AccelerationStructureInput& input)
        : AccelerationStructureSizes(context,
                                     input.type,
                                     input.flags,
                                     input.geometries,
                                     input.rangeInfos)
    {
    }

    AccelerationStructureSizes(const Context&                                            context,
                               VkAccelerationStructureTypeKHR                            type,
                               VkBuildAccelerationStructureFlagsKHR                      flags,
                               std::span<const VkAccelerationStructureGeometryKHR>       geometries,
                               std::span<const VkAccelerationStructureBuildRangeInfoKHR> rangeInfos)
        : m_sizeInfo{
              .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR,
              .pNext = nullptr,
          }
    {
        assert(geometries.size() == rangeInfos.size());
        VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{
            .sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type          = type,
            .flags         = flags,
            .geometryCount = static_cast<uint32_t>(geometries.size()),
            .pGeometries   = geometries.data(),
        };
        std::vector<uint32_t> primitiveCounts(rangeInfos.size());
        std::transform(rangeInfos.begin(), rangeInfos.end(), primitiveCounts.begin(),
                       [](const VkAccelerationStructureBuildRangeInfoKHR& rangeInfo) {
                           return rangeInfo.primitiveCount;
                       });
        vkGetAccelerationStructureBuildSizesKHR(
            context.device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildGeometryInfo,
            primitiveCounts.data(), &m_sizeInfo);
    }

    const VkAccelerationStructureBuildSizesInfoKHR& operator*() const { return m_sizeInfo; }
    VkAccelerationStructureBuildSizesInfoKHR&       operator*() { return m_sizeInfo; }
    const VkAccelerationStructureBuildSizesInfoKHR* operator->() const { return &m_sizeInfo; }
    VkAccelerationStructureBuildSizesInfoKHR*       operator->() { return &m_sizeInfo; }

private:
    VkAccelerationStructureBuildSizesInfoKHR m_sizeInfo;
};

class Buffer {
public:
    Buffer(const Context&        context,
           VkDeviceSize          size,
           VkBufferUsageFlags    usageFlags,
           VkMemoryPropertyFlags propertyFlags)
        : m_context(&context),
          m_buffer(context.allocator->createBuffer(size, usageFlags, propertyFlags)),
          m_address(getAddress(context.device, m_buffer.buffer))
    {
        // FIXME - 源码这里是redundant的getAddress函数
    }

    template <class Range>
    Buffer(const Context&        context,
           const Range&          range,
           VkBufferUsageFlags    usageFlags,
           VkMemoryPropertyFlags propertyFlags,
           VkCommandBuffer       cmd)
        : m_context(&context),
          m_buffer(context.allocator->createBuffer(cmd,
                                                   sizeof(*range.data()) * range.size(),
                                                   range.data(),
                                                   usageFlags,
                                                   propertyFlags)),
          m_address(getAddress(context.device, m_buffer.buffer))
    {
    }

    ~Buffer()
    {
        if (m_context != nullptr) { m_context->allocator->destroy(m_buffer); }
    }

    Buffer& operator=(const Buffer& other) = delete;
    Buffer& operator=(Buffer&& other)      = delete;
    Buffer(const Buffer& other)            = delete;
    Buffer(Buffer&& other) noexcept
        : m_context(other.m_context), m_buffer(other.m_buffer), m_address(other.m_address)
    {
        other.m_context = nullptr;
        other.m_buffer  = {};
        other.m_address = {};
    }

    const VkDeviceAddress& address() const { return m_address; }
    operator const VkBuffer&() { return m_buffer.buffer; }

private:
    const Context*  m_context = nullptr;
    nvvk::Buffer    m_buffer;
    VkDeviceAddress m_address;

    static VkDeviceAddress getAddress(VkDevice device, VkBuffer buffer)
    {
        VkBufferDeviceAddressInfo bufferInfo{
            .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .pNext  = nullptr,
            .buffer = buffer,
        };
        return vkGetBufferDeviceAddress(device, &bufferInfo);
    }
};

class InstanceBuffer : public Buffer {
public:
    InstanceBuffer(const Context&                                      context,
                   std::span<const VkAccelerationStructureInstanceKHR> instances,
                   VkCommandBuffer                                     cmd)
        : Buffer(context,
                 instances,
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 cmd)
    {
        // Make sure Buffer()'s upload is complete and visible for the subsequent
        // acceleration structure build.
        VkMemoryBarrier barrier{
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier,
                             0, nullptr, 0, nullptr);
        // NOTE - 创建buffer时会把数据上传到device内存
        // 需要一个屏障barrier,来确保数据上传完成后才能开始构建加速结构
    }

private:
};

class BuiltAccelerationStructure;

class AccelerationStructure {
public:
    AccelerationStructure(const Context&                                  context,
                          VkAccelerationStructureTypeKHR                  type,
                          const VkAccelerationStructureBuildSizesInfoKHR& size,
                          VkAccelerationStructureCreateFlagsKHR           flags)
        : m_context(&context), m_type(type), m_size(size),
          m_buffer(context,
                   m_size.accelerationStructureSize,
                   VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR |
                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                   VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
          m_accelerationStructure(VK_NULL_HANDLE)
    {
        VkAccelerationStructureCreateInfoKHR createInfo{
            .sType         = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR,
            .createFlags   = flags,
            .buffer        = m_buffer,
            .offset        = 0,
            .size          = m_size.accelerationStructureSize,
            .type          = m_type,
            .deviceAddress = 0,
        };
        context.resultCallback(vkCreateAccelerationStructureKHR(
            context.device, &createInfo, context.allocationCallbacks, &m_accelerationStructure));

        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR,
            .accelerationStructure = m_accelerationStructure,
        };
        m_address = vkGetAccelerationStructureDeviceAddressKHR(context.device, &addressInfo);
    }
    ~AccelerationStructure()
    {
        if (m_context != nullptr)
        {
            vkDestroyAccelerationStructureKHR(m_context->device, m_accelerationStructure,
                                              m_context->allocationCallbacks);
        }
    }

    AccelerationStructure()                                   = delete;
    AccelerationStructure(const AccelerationStructure& other) = delete;
    AccelerationStructure(AccelerationStructure&& other) noexcept
        : m_context(other.m_context), m_buffer(std::move(other.m_buffer)), m_type(other.m_type),
          m_size(other.m_size), m_accelerationStructure(other.m_accelerationStructure),
          m_address(other.m_address)
    {
        other.m_context               = nullptr;
        other.m_type                  = {};
        other.m_size                  = {};
        other.m_accelerationStructure = VK_NULL_HANDLE;
        other.m_address               = {};
    }

    AccelerationStructure& operator=(const AccelerationStructure& other) = delete;
    AccelerationStructure& operator=(AccelerationStructure&& other)      = delete;

    const VkAccelerationStructureTypeKHR&           type() const { return m_type; }
    const VkAccelerationStructureBuildSizesInfoKHR& sizes() { return m_size; }

private:
    friend class BuiltAccelerationStructure;

    const VkAccelerationStructureKHR& object() const { return m_accelerationStructure; }
    const VkDeviceAddress&            address() const { return m_address; }

    const Context*                           m_context;
    VkAccelerationStructureTypeKHR           m_type;
    VkAccelerationStructureBuildSizesInfoKHR m_size;
    Buffer                                   m_buffer;
    VkAccelerationStructureKHR               m_accelerationStructure;
    VkDeviceAddress                          m_address;
};

/**
 * @brief 会自动调整大小的缓冲区
 *
 */
class ScratchBuffer {
public:
    explicit ScratchBuffer(const Context& context) : m_context(&context) {}
    ScratchBuffer(const Context& context, VkDeviceSize size) : m_context(&context) { resize(size); }

    const Buffer& buffer(VkDeviceSize size)
    {
        if (m_size < size) { resize(size); }
        return *m_buffer;
    }

private:
    const Context*        m_context;
    VkDeviceSize          m_size = 0;
    std::optional<Buffer> m_buffer;

    void resize(VkDeviceSize size)
    {
        m_size = size;
        m_buffer.emplace(*m_context, size,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                             VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        // std::optional会负责销毁原本的m_buffer
    }
};

class BuiltAccelerationStructure {
public:
    BuiltAccelerationStructure(const Context&                    context,
                               AccelerationStructure&&           accelerationStructure,
                               const AccelerationStructureInput& input,
                               ScratchBuffer&                    scratchBuffer,
                               VkCommandBuffer                   cmd)
        : BuiltAccelerationStructure(context,
                                     std::move(accelerationStructure),
                                     input.flags,
                                     input.geometries,
                                     input.rangeInfos,
                                     scratchBuffer,
                                     cmd)
    {
    }

    BuiltAccelerationStructure(const Context&                       context,
                               AccelerationStructure&&              accelerationStructure,
                               VkBuildAccelerationStructureFlagsKHR flags,
                               std::span<const VkAccelerationStructureGeometryKHR>       geometries,
                               std::span<const VkAccelerationStructureBuildRangeInfoKHR> rangeInfos,
                               ScratchBuffer&  scratchBuffer,
                               VkCommandBuffer cmd)
        : m_accelerationStructure(std::move(accelerationStructure))
    {
        build(context, flags, geometries, rangeInfos, false, scratchBuffer, cmd);
    }

    void rebuild(const Context&                    context,
                 const AccelerationStructureInput& input,
                 ScratchBuffer&                    scratchBuffer,
                 VkCommandBuffer                   cmd)
    {
        rebuild(context, input.flags, input.geometries, input.rangeInfos, scratchBuffer, cmd);
    }

    void rebuild(const Context&                                            context,
                 VkBuildAccelerationStructureFlagsKHR                      flags,
                 std::span<const VkAccelerationStructureGeometryKHR>       geometries,
                 std::span<const VkAccelerationStructureBuildRangeInfoKHR> rangeInfos,
                 ScratchBuffer&                                            scratchBuffer,
                 VkCommandBuffer                                           cmd)
    {
        build(context, flags, geometries, rangeInfos, false, scratchBuffer, cmd);
    }

    void update(const Context&                    context,
                const AccelerationStructureInput& input,
                ScratchBuffer&                    scratchBuffer,
                VkCommandBuffer                   cmd)
    {
        update(context, input.flags, input.geometries, input.rangeInfos, scratchBuffer, cmd);
    }

    void update(const Context&                                            context,
                VkBuildAccelerationStructureFlagsKHR                      flags,
                std::span<const VkAccelerationStructureGeometryKHR>       geometries,
                std::span<const VkAccelerationStructureBuildRangeInfoKHR> rangeInfos,
                ScratchBuffer&                                            scratchBuffer,
                VkCommandBuffer                                           cmd)
    {
        build(context, flags, geometries, rangeInfos, true, scratchBuffer, cmd);
    }

    operator const VkAccelerationStructureKHR&() const { return m_accelerationStructure.object(); }
    const VkAccelerationStructureKHR& object() const { return m_accelerationStructure.object(); }
    const VkDeviceAddress&            address() const { return m_accelerationStructure.address(); }

private:
    AccelerationStructure m_accelerationStructure;

    void build(const Context&                                            context,
               VkBuildAccelerationStructureFlagsKHR                      flags,
               std::span<const VkAccelerationStructureGeometryKHR>       geometries,
               std::span<const VkAccelerationStructureBuildRangeInfoKHR> rangeInfos,
               bool                                                      update,
               ScratchBuffer&                                            scratchBuffer,
               VkCommandBuffer                                           cmd)
    {
        assert(geometries.size() == rangeInfos.size());
        assert(!update || !!(flags & VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR));
        // NOTE - 要么不更新，要么更新的时候vk标志要置位ACCELERATION_STRUCTURE_ALLOW_UPDATE

        VkBuildAccelerationStructureModeKHR mode =
            update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR :
                     VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        auto&           sizes          = m_accelerationStructure.sizes();
        VkDeviceSize    scratchSize    = update ? sizes.updateScratchSize : sizes.buildScratchSize;
        VkDeviceAddress scratchAddress = scratchBuffer.buffer(scratchSize).address();

        VkAccelerationStructureBuildGeometryInfoKHR buildGeometryInfo{
            .sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR,
            .type  = m_accelerationStructure.type(),
            .flags = flags,
            .mode  = mode,
            .srcAccelerationStructure = update ? m_accelerationStructure.object() : VK_NULL_HANDLE,
            .dstAccelerationStructure = m_accelerationStructure.object(),
            .geometryCount            = static_cast<uint32_t>(geometries.size()),
            .pGeometries              = geometries.data(),
            .scratchData              = {.deviceAddress = scratchAddress},
        };
        auto rangeInfosPtr = rangeInfos.data();
        vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildGeometryInfo, &rangeInfosPtr);

        // NOTE - 由于scratch缓冲区在多个构建中被重复使用，我们需要一个屏障来确保一个构建
        // 完成后才能开始下一个构建。
        VkMemoryBarrier barrier{
            .sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
            .dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR,
        };
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                             VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, 0, 1, &barrier,
                             0, nullptr, 0, nullptr);

        // NOTE - scratchBuffer在构建加速结构的过程中充当临时存储空间
        // 多个构建共享同一个scratchBuffer
    };
};

inline AccelerationStructureInput createTlasInput(uint32_t        instanceCount,
                                                  VkDeviceAddress instanceBufferAddress,
                                                  VkBuildAccelerationStructureFlagsKHR flags)
{
    return AccelerationStructureInput{
        .type  = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR,
        .flags = flags,
        .geometries =
            {
                VkAccelerationStructureGeometryKHR{
                    .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
                    .pNext        = nullptr,
                    .geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR,
                    .geometry =
                        VkAccelerationStructureGeometryDataKHR{
                            .instances =
                                VkAccelerationStructureGeometryInstancesDataKHR{
                                    .sType =
                                        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR,
                                    .data = {instanceBufferAddress},
                                },
                        },
                },
            },
        .rangeInfos =
            {
                VkAccelerationStructureBuildRangeInfoKHR{
                    .primitiveCount  = instanceCount,
                    .primitiveOffset = 0,
                    .firstVertex     = 0,
                    .transformOffset = 0,
                },
            },
    };
}

struct SimpleGeometryInput
{
    uint32_t           triangleCount;
    uint32_t           maxVertex;
    VkDeviceAddress    indexAddress;
    VkDeviceAddress    vertexAddress;
    VkDeviceSize       vertexStride = sizeof(float) * 3;
    VkFormat           vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    VkIndexType        indexType    = VK_INDEX_TYPE_UINT32;
    VkGeometryFlagsKHR geometryFlags =
        VK_GEOMETRY_OPAQUE_BIT_KHR | VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;
};

inline AccelerationStructureInput
createBlasInput(std::span<const SimpleGeometryInput> simpleInputs,
                VkBuildAccelerationStructureFlagsKHR accelerationStructureFlags)
{
    AccelerationStructureInput result{
        .type  = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR,
        .flags = accelerationStructureFlags,
    };

    for (const auto& simpleInput : simpleInputs)
    {
        result.geometries.emplace_back(VkAccelerationStructureGeometryKHR{
            .sType        = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR,
            .geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR,
            .geometry =
                VkAccelerationStructureGeometryDataKHR{
                    .triangles =
                        VkAccelerationStructureGeometryTrianglesDataKHR{
                            .sType =
                                VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR,
                            .vertexFormat  = simpleInput.vertexFormat,
                            .vertexData    = {simpleInput.vertexAddress},
                            .vertexStride  = simpleInput.vertexStride,
                            .maxVertex     = simpleInput.maxVertex,
                            .indexType     = simpleInput.indexType,
                            .indexData     = {simpleInput.indexAddress},
                            .transformData = {0},
                        },
                },
            .flags = simpleInput.geometryFlags,
        });
        result.rangeInfos.emplace_back(
            VkAccelerationStructureBuildRangeInfoKHR{.primitiveCount = simpleInput.triangleCount});
    }
    return result;
}

};  // namespace rt
