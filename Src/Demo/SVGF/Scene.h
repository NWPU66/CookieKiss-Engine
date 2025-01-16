#pragma once

// c/c++
#include <cstdint>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>
#include <vulkan/vulkan_core.h>

// 3rdparty
#include "glm/ext/matrix_transform.hpp"
#include "glm/glm.hpp"
#include "glm/gtc/quaternion.hpp"
#include "glm/matrix.hpp"
#include "nvh/cameramanipulator.hpp"
#include "nvvk/buffers_vk.hpp"
#include "nvvk/descriptorsets_vk.hpp"
#include "nvvk/dynamicrendering_vk.hpp"
#include "nvvk/pipeline_vk.hpp"
#include "nvvk/resourceallocator_vk.hpp"
#include "nvvk/shadermodulemanager_vk.hpp"
#include "nvvkhl/application.hpp"
#include "nvvkhl/gbuffer.hpp"
#include "stb_image.h"
#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

// users
#include "Shader/common.h"

namespace cookiekiss {

glm::vec3 toGlmVec3(const aiVector3D& vec)
{
    return {vec.x, vec.y, vec.z};
}

glm::vec2 toGlmVec2(const aiVector3D& vec)
{
    return {vec.x, vec.y};
}

class Object {
public:
    void init(nvvkhl::Application* app, const std::shared_ptr<nvvk::ResourceAllocator>& allocator);

protected:
    nvvkhl::Application*                   m_app;
    std::weak_ptr<nvvk::ResourceAllocator> m_allocator;
};

class Scene;

struct VertexInput
{
    glm::vec3 p;  // Position
    glm::vec3 n;  // Normal
    glm::vec2 t;  // Texcoords
};

class Mesh : public Object {
public:
    ~Mesh() { destory(); }

    void bindingVBO(VkCommandBuffer cmd)
    {
        VkDeviceSize offsets[] = {0};
        vkCmdBindIndexBuffer(cmd, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindVertexBuffers(cmd, 0, std::size(offsets), &vertexBuffer.buffer, offsets);
    }

    uint32_t getIndexCount() const { return indices.size(); }

    void createMesh(aiMesh* mesh, const aiScene* sceneData, Scene& scene)
    {
        createMeshCPU(mesh, sceneData, scene);
        createMeshGPU();
    }

    void destory()
    {
        destoryMeshGPU();
        destoryMeshCPU();
    }

private:
    // cpu side
    std::vector<VertexInput> vertices;
    std::vector<uint32_t>    indices;

    // gpu side
    nvvk::Buffer vertexBuffer;
    nvvk::Buffer indexBuffer;

    void createMeshCPU(aiMesh* mesh, const aiScene* sceneData, Scene& scene)
    {
        vertices.reserve(mesh->mNumVertices);
        indices.reserve(mesh->mNumFaces * 3);

        // vertex
        for (uint32_t j = 0; j < mesh->mNumVertices; j++)
        {
            VertexInput vertex{
                .p = toGlmVec3(mesh->mVertices[j]),
                .n = toGlmVec3(mesh->mNormals[j]),
                .t = toGlmVec2(mesh->mTextureCoords[0][j]),
            };
            vertices.push_back(vertex);
        }

        // index
        for (unsigned int j = 0; j < mesh->mNumFaces; j++)
        {
            aiFace& face = mesh->mFaces[j];
            indices.push_back(face.mIndices[0]);
            indices.push_back(face.mIndices[1]);
            indices.push_back(face.mIndices[2]);
        }
    }

    void createMeshGPU()
    {
        auto cmd     = m_app->createTempCmdBuffer();
        vertexBuffer = m_allocator.lock()->createBuffer(
            cmd, vertices, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        indexBuffer = m_allocator.lock()->createBuffer(
            cmd, indices, VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        m_app->submitAndWaitTempCmdBuffer(cmd);
    }

    void destoryMeshCPU()
    {
        vertices.clear();
        indices.clear();
    }

    void destoryMeshGPU()
    {
        m_allocator.lock()->destroy(vertexBuffer);
        m_allocator.lock()->destroy(indexBuffer);
    }
};

class Texture : public Object {
public:
    ~Texture() { destroy(); }

    void createTexture(const std::string& textureFile)
    {
        createTextureCPU(textureFile);
        createTextureGPU();
    }

    nvvk::Texture getTexture() { return m_texture; }

private:
    // cpu side
    int32_t        m_width;
    int32_t        m_height;
    int32_t        m_channels;
    unsigned char* m_data;
    size_t         m_dataSize;
    std::string    m_filePath;

    // gpu side
    nvvk::Texture m_texture;

    void createTextureCPU(const std::string& textureFile)
    {
        m_filePath            = textureFile;
        unsigned char* m_data = stbi_load(textureFile.c_str(), &m_width, &m_height, &m_channels, 0);
        if (m_data == nullptr)
        {
            std::cout << "Failed to load texture: " << textureFile << std::endl;
        }
        m_dataSize = m_width * m_height * m_channels;
    }

    void createTextureGPU()
    {
        auto              cmd                = m_app->createTempCmdBuffer();
        uint32_t          queueFamilyIndex[] = {m_app->getQueue(0).familyIndex};
        VkImageCreateInfo imageInfo{
            .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
            .pNext = nullptr,
            /* .flags = , */
            .imageType = VK_IMAGE_TYPE_2D,
            .format    = VK_FORMAT_R8G8B8_UNORM,
            // REVIEW - base color 可以使用 VK_FORMAT_R8G8B8_SRGB
            .extent      = VkExtent3D{(uint32_t)m_width, (uint32_t)m_height, 1},
            .mipLevels   = 1,
            .arrayLayers = 1,
            .samples     = VK_SAMPLE_COUNT_1_BIT,
            .tiling      = VK_IMAGE_TILING_OPTIMAL,
            .usage       = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,  // TODO
            .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = std::size(queueFamilyIndex),
            .pQueueFamilyIndices   = queueFamilyIndex,
            .initialLayout         = VK_IMAGE_LAYOUT_UNDEFINED,
        };
        VkSamplerCreateInfo samplerInfo{
            .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .pNext = nullptr,
            /* .flags = , */
            .magFilter               = VK_FILTER_LINEAR,
            .minFilter               = VK_FILTER_LINEAR,
            .mipmapMode              = VK_SAMPLER_MIPMAP_MODE_LINEAR,
            .addressModeU            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW            = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .mipLodBias              = 0.0f,
            .anisotropyEnable        = VK_FALSE,
            .maxAnisotropy           = 1.0f,
            .compareEnable           = VK_FALSE,
            .compareOp               = VK_COMPARE_OP_ALWAYS,
            .minLod                  = 0.0f,
            .maxLod                  = 0.0f,
            .borderColor             = VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = VK_FALSE,
        };
        m_texture =
            m_allocator.lock()->createTexture(cmd, m_dataSize, m_data, imageInfo, samplerInfo);
        m_app->submitAndWaitTempCmdBuffer(cmd);
    }

    void destroy()
    {
        m_allocator.lock()->destroy(m_texture);  // gpu side
        stbi_image_free(m_data);                 // cpu side
    }
};

class TextureSet : public Object {
public:
    ~TextureSet() { destory(); }

    void bindingTextures(VkCommandBuffer cmd, VkPipelineLayout pipelineLayout)
    {
        VkDescriptorSet descSet[] = {m_descriptorSetContainer.getSet()};
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0,
                                std::size(descSet), descSet, 0, nullptr);
    }

    void createTextureSet(aiMesh* mesh, const aiScene* sceneData, Scene& scene)
    {
        // load textures, cpu side
        aiMaterial* material       = sceneData->mMaterials[mesh->mMaterialIndex];
        m_texture_baseColor        = loadTexture(material, aiTextureType_BASE_COLOR, scene);
        m_texture_normal           = loadTexture(material, aiTextureType_NORMALS, scene);
        m_texture_metallic         = loadTexture(material, aiTextureType_METALNESS, scene);
        m_texture_roughness        = loadTexture(material, aiTextureType_DIFFUSE_ROUGHNESS, scene);
        m_texture_ambientOcclusion = loadTexture(material, aiTextureType_AMBIENT_OCCLUSION, scene);

        // create descriptor set
        m_descriptorSetContainer.init(m_app->getDevice());
        m_descriptorSetContainer.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                            VK_SHADER_STAGE_FRAGMENT_BIT);
        m_descriptorSetContainer.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                            VK_SHADER_STAGE_FRAGMENT_BIT);
        m_descriptorSetContainer.addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                            VK_SHADER_STAGE_FRAGMENT_BIT);
        m_descriptorSetContainer.addBinding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                            VK_SHADER_STAGE_FRAGMENT_BIT);
        m_descriptorSetContainer.addBinding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                            VK_SHADER_STAGE_FRAGMENT_BIT);
        m_descriptorSetContainer.initLayout();
        m_descriptorSetContainer.initPool(1);

        // modify descriptor set, gpu side
        std::vector<VkWriteDescriptorSet> writeDescSets;
        for (const auto& tex : {
                 m_texture_baseColor,
                 m_texture_normal,
                 m_texture_metallic,
                 m_texture_roughness,
                 m_texture_ambientOcclusion,
             })
        {
            auto descImageInfo = tex.lock()->getTexture().descriptor;
            auto writeDescSet  = m_descriptorSetContainer.makeWrite(0, 0, &descImageInfo);
            writeDescSets.push_back(writeDescSet);
        }
        vkUpdateDescriptorSets(m_app->getDevice(), writeDescSets.size(), writeDescSets.data(), 0,
                               nullptr);
    }

    static nvvk::DescriptorSetContainer& getDescriptorSetExample(VkDevice device)
    {
        static bool                         init = false;
        static nvvk::DescriptorSetContainer descriptorSetContainer;
        if (!init)
        {
            descriptorSetContainer.init(device);
            descriptorSetContainer.addBinding(0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                              VK_SHADER_STAGE_FRAGMENT_BIT);
            descriptorSetContainer.addBinding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                              VK_SHADER_STAGE_FRAGMENT_BIT);
            descriptorSetContainer.addBinding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                              VK_SHADER_STAGE_FRAGMENT_BIT);
            descriptorSetContainer.addBinding(3, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                              VK_SHADER_STAGE_FRAGMENT_BIT);
            descriptorSetContainer.addBinding(4, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1,
                                              VK_SHADER_STAGE_FRAGMENT_BIT);
            descriptorSetContainer.initLayout();
        }
        return descriptorSetContainer;
    }

private:
    nvvk::DescriptorSetContainer m_descriptorSetContainer;

    enum TextureType {
        eNone,
        eBaseColor,
        eNormal,
        eEmissive,
        eMetallic,
        eRoughness,
        eAmbientOcclusion,
        eCount,
    };

    // PBR textures
    std::weak_ptr<Texture> m_texture_baseColor;
    std::weak_ptr<Texture> m_texture_normal;
    std::weak_ptr<Texture> m_texture_metallic;
    std::weak_ptr<Texture> m_texture_roughness;
    std::weak_ptr<Texture> m_texture_ambientOcclusion;

    std::shared_ptr<Texture> loadTexture(aiMaterial* material, aiTextureType type, Scene& scene);

    void destory() { m_descriptorSetContainer.deinit(); }
};

class Pipeline : public Object {
public:
    virtual ~Pipeline()                               = 0;
    virtual void bindingPipeline(VkCommandBuffer cmd) = 0;
};

class GraphicsPipeline : public Pipeline {
public:
    ~GraphicsPipeline() override
    {
        vkDestroyPipeline(m_app->getDevice(), m_pipeline, nullptr);
        vkDestroyPipelineLayout(m_app->getDevice(), m_pipelineLayout, nullptr);
        m_shaderManager.deinit();
    }

    void createPipeline(const std::string& vertShaderName,
                        const std::string& fragShaderName,
                        const std::string& shaderFolder)
    {
        // SMM
        m_shaderManager.init(m_app->getDevice());
        m_shaderManager.addDirectory(shaderFolder);
        auto vid = m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, vertShaderName);
        auto fid = m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, fragShaderName);

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
                .stride    = sizeof(VertexInput),
                .inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
            },

        });
        m_pipelineState.addAttributeDescriptions({
            VkVertexInputAttributeDescription{
                .location = 0,
                .binding  = 0,
                .format   = VK_FORMAT_R32G32B32_SFLOAT,  // postion
                .offset   = offsetof(VertexInput, p),
            },
            VkVertexInputAttributeDescription{
                .location = 1,
                .binding  = 0,
                .format   = VK_FORMAT_R32G32B32_SFLOAT,  // normal
                .offset   = offsetof(VertexInput, n),
            },
            VkVertexInputAttributeDescription{
                .location = 2,
                .binding  = 0,
                .format   = VK_FORMAT_R32G32_SFLOAT,  // texcoords
                .offset   = offsetof(VertexInput, t),
            },
        });

        // Other Settings
        m_pipelineState.rasterizationState.cullMode = VK_CULL_MODE_NONE;

        // create pipelayout
        VkPushConstantRange pushConstantRange{
            .stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset     = 0,
            .size       = sizeof(PushContent),
        };
        m_pipelineLayout = TextureSet::getDescriptorSetExample(m_app->getDevice())
                               .initPipeLayout(1, &pushConstantRange);

        nvvk::GraphicsPipelineGenerator generator(m_app->getDevice(), m_pipelineLayout,
                                                  renderingCreateInfo, m_pipelineState);
        generator.addShader(m_shaderManager.get(vid), VK_SHADER_STAGE_VERTEX_BIT);
        generator.addShader(m_shaderManager.get(fid), VK_SHADER_STAGE_FRAGMENT_BIT);
        m_pipeline = generator.createPipeline();
    }

    void bindingPipeline(VkCommandBuffer cmd) override
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    }

    VkPipelineLayout getPipelineLayout() const { return m_pipelineLayout; }

private:
    nvvk::ShaderModuleManager   m_shaderManager;
    nvvk::GraphicsPipelineState m_pipelineState;
    VkPipelineLayout            m_pipelineLayout;
    VkPipeline                  m_pipeline;
};

class Instance : public Object {
public:
    void createInstance(aiMesh* mesh, const aiScene* sceneData, Scene& scene);

    void draw(VkCommandBuffer cmd, glm::mat4 vpMatrix, glm::mat4 parentMatrix = glm::mat4(1))
    {
        // binding pipeline
        m_pipeline.lock()->bindingPipeline(cmd);

        // binding VBO
        m_mesh.lock()->bindingVBO(cmd);

        // binding texture
        auto pipelineLayout =
            dynamic_cast<GraphicsPipeline*>(m_pipeline.lock().get())->getPipelineLayout();
        m_textureSet.lock()->bindingTextures(cmd, pipelineLayout);

        // pc
        auto        mMatrix = localMatrix();
        PushContent pc{.m = mMatrix, .mvp = vpMatrix * mMatrix};
        vkCmdPushConstants(cmd, pipelineLayout,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(pc),
                           &pc);

        // draw
        vkCmdDrawIndexed(cmd, m_mesh.lock()->getIndexCount(), 1, 0, 0, 0);
    }

private:
    std::weak_ptr<Mesh>       m_mesh;
    std::weak_ptr<TextureSet> m_textureSet;
    std::weak_ptr<Pipeline>   m_pipeline;

    glm::vec3 translation;
    glm::quat rotation;
    glm::vec3 scale;

    glm::mat4 localMatrix(glm::mat4 parentMatrix = glm::mat4(1)) const
    {
        glm::mat4 translationMatrix = glm::translate(glm::mat4(1), translation);
        glm::mat4 rotationMatrix    = glm::mat4_cast(rotation);
        glm::mat4 scaleMatrix       = glm::scale(glm::mat4(1), scale);
        glm::mat4 combinedMatrix = translationMatrix * rotationMatrix * scaleMatrix * parentMatrix;
        return combinedMatrix;
    }
};

class Node : public Object {
public:
    void createNode(aiNode* node, const aiScene* sceneData, Scene& scene);

    void draw(VkCommandBuffer cmd, glm::mat4 vpMatrix, glm::mat4 parentMatrix = glm::mat4(1))
    {
        parentMatrix = parentMatrix * m_transform;
        for (auto& instance : m_instances)
        {
            instance.lock()->draw(cmd, vpMatrix, parentMatrix);
        }
        for (auto& child : m_children)
        {
            child.lock()->draw(cmd, vpMatrix, parentMatrix);
        }
    }

    std::shared_ptr<Node> getParentNode() { return m_parent.lock(); }
    void                  setParentNode(std::shared_ptr<Node> node) { m_parent = node; }
    void                  addChildNode(std::shared_ptr<Node> node) { m_children.push_back(node); }
    void                  removeChildNode(std::shared_ptr<Node> node)
    {
        for (auto it = m_children.begin(); it != m_children.end(); it++)
        {
            if (it->lock() == node)
            {
                m_children.erase(it);
                break;
            }
        }
    }

private:
    std::weak_ptr<Node>              m_parent;
    std::vector<std::weak_ptr<Node>> m_children;

    std::vector<std::weak_ptr<Instance>> m_instances;
    glm::mat4                            m_transform = glm::mat4(1);
};

class Scene : public Object {
public:
    ~Scene() { destroy(); }

    void createScene(const std::string& filePath)
    {
        Assimp::Importer importer;
        const aiScene*   sceneData =
            importer.ReadFile(filePath, aiProcessPreset_TargetRealtime_MaxQuality);
        if (!sceneData || sceneData->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !sceneData->mRootNode)
        {
            std::cout << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        }

        processNode(sceneData->mRootNode, sceneData);
    }

    void destroy()
    {
        m_meshes.clear();
        m_instances.clear();
        m_pipelines.clear();
        m_textures.clear();
        m_nodes.clear();
    }

    void draw(VkCommandBuffer cmd, nvvkhl::GBuffer& gbuffer)
    {
        // preparation for drawing
        VkRect2D renderArea{
            .offset = {0, 0},
            .extent = gbuffer.getSize(),
        };
        nvvk::createRenderingInfo renderingInfo(renderArea, {gbuffer.getColorImageView()},
                                                gbuffer.getDepthImageView());
        vkCmdBeginRendering(cmd, (VkRenderingInfoKHR*)&renderingInfo);

        // 手动设置viewport和scissor
        VkExtent2D size = gbuffer.getSize();
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

        // pre-calculate v and p
        auto pMatrix = glm::perspectiveFovRH_ZO(glm::radians(45.0f), (float)size.width,
                                                (float)size.height, 0.001f, 100.0f);
        pMatrix[1][1] *= -1;
        auto vMatrix  = CameraManip.getMatrix();
        auto vpMatrix = pMatrix * vMatrix;
        /*REVIEW - 我还是不理解这里为什么用 RH
        vulkan的NDC空间就是右手系的，p[1][1]乘上-1后会变成左手系
        */

        // rendering
        m_rootNode.lock()->draw(cmd, vpMatrix, glm::mat4(1));
        vkCmdEndRendering(cmd);
    }

private:
    friend class Node;
    friend class Instance;
    friend class TextureSet;

    std::vector<std::shared_ptr<Mesh>>     m_meshes;
    std::vector<std::shared_ptr<Texture>>  m_textures;
    std::vector<std::shared_ptr<Pipeline>> m_pipelines;
    std::vector<std::shared_ptr<Instance>> m_instances;
    std::vector<std::shared_ptr<Node>>     m_nodes;

    std::weak_ptr<Node> m_rootNode;

    std::shared_ptr<Node>
    processNode(aiNode* node, const aiScene* sceneData, std::shared_ptr<Node> parentNode = {})
    {
        // process self
        auto pNode = std::make_shared<Node>();
        pNode->init(m_app, m_allocator.lock());
        pNode->createNode(node, sceneData, *this);
        if (parentNode == nullptr)  // set parent
        {
            m_rootNode = pNode;  // root node
        }
        else { pNode->setParentNode(parentNode); }
        m_nodes.push_back(pNode);

        // process children
        for (unsigned int i = 0; i < node->mNumChildren; i++)
        {
            auto pChildNode = processNode(node->mChildren[i], sceneData);
            pNode->addChildNode(pChildNode);  // set child node
        }

        return pNode;
    }
};

// Implimentation =================================================================

void Node::createNode(aiNode* node, const aiScene* sceneData, Scene& scene)
{
    for (unsigned int i = 0; i < node->mNumMeshes; i++)
    {
        aiMesh* mesh      = sceneData->mMeshes[node->mMeshes[i]];
        auto    pInstance = std::make_shared<Instance>();
        pInstance->init(m_app, m_allocator.lock());
        pInstance->createInstance(mesh, sceneData, scene);

        // add to scene
        scene.m_instances.push_back(pInstance);

        // add to node
        m_instances.push_back(pInstance);
    }
}

void Instance::createInstance(aiMesh* mesh, const aiScene* sceneData, Scene& scene)
{
    // load mesh
    auto pMesh = std::make_shared<Mesh>();
    pMesh->init(m_app, m_allocator.lock());
    pMesh->createMesh(mesh, sceneData, scene);
    m_mesh = pMesh;
    scene.m_meshes.push_back(pMesh);

    // load texture set
    auto pTextureSet = std::make_shared<TextureSet>();
    pTextureSet->init(m_app, m_allocator.lock());
    pTextureSet->createTextureSet(mesh, sceneData, scene);
    m_textureSet = pTextureSet;

    // TODO - load pipeline
}

std::shared_ptr<Texture>
TextureSet::loadTexture(aiMaterial* material, aiTextureType type, Scene& scene)
{
    aiString str;
    material->GetTexture(type, 0, &str);  // always load the first texture

    // load texture
    std::string textureFile = str.C_Str();
    auto        pTexture    = std::make_shared<Texture>();
    pTexture->init(m_app, m_allocator.lock());
    pTexture->createTexture(textureFile);

    // add to scene
    scene.m_textures.push_back(pTexture);

    return pTexture;
}

};  // namespace cookiekiss
