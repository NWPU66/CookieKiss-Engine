#pragma once

// c/c++
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

// third party
#include "nvh/primitives.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// #define LOAD_METHOD_ASSIMP
// #define LOAD_METHOD_TINYOBJLOADER

#ifdef LOAD_METHOD_ASSIMP
#    include <assimp/Importer.hpp>
#    include <assimp/postprocess.h>
#    include <assimp/scene.h>
#    ifdef LOAD_METHOD_TINYOBJLOADER
#        error                                                                                     \
            "Both LOAD_METHOD_ASSIMP and LOAD_METHOD_TINYOBJLOADER are defined. Only one can be defined."
#    endif
#endif
#ifdef LOAD_METHOD_TINYOBJLOADER
#    include <tinyobjloader/tiny_obj_loader.h>
#endif

/* NOTE - nvh Primitives
struct PrimitiveVertex
{
  glm::vec3 p;  // Position
  glm::vec3 n;  // Normal
  glm::vec2 t;  // Texture Coordinates
};

struct PrimitiveTriangle
{
  glm::uvec3 v;  // vertex indices
};

struct PrimitiveMesh
{
  std::vector<PrimitiveVertex>   vertices;   // Array of all vertex
  std::vector<PrimitiveTriangle> triangles;  // Indices forming triangles
};
*/

namespace cookiekiss {

std::tuple<std::vector<nvh::PrimitiveMesh>, std::vector<nvh::Node>>
loadGeometryFromFile(const std::string& filePath, bool loadInstances = false);

#ifdef LOAD_METHOD_ASSIMP

glm::vec3 toGlmVec3(const aiVector3D& vec)
{
    return {vec.x, vec.y, vec.z};
}

glm::vec2 toGlmVec2(const aiVector3D& vec)
{
    return {vec.x, vec.y};
}

void processNode(aiNode* node, const aiScene* scene)
{
    // 处理节点所有的网格（如果有的话）
    for (unsigned int i = 0; i < node->mNumMeshes; i++)
    {
        // TODO -
    }

    // 接下来对它的子节点重复这一过程
    for (unsigned int i = 0; i < node->mNumChildren; i++)
    {
        processNode(node->mChildren[i], scene);
    }
}

std::tuple<std::vector<nvh::PrimitiveMesh>, std::vector<nvh::Node>>
loadGeometryFromFile(const std::string& filePath, bool loadInstances)
{
    Assimp::Importer importer;
    const aiScene*   scene = importer.ReadFile(filePath, aiProcessPreset_TargetRealtime_MaxQuality);
    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode)
    {
        std::cout << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        throw std::runtime_error(std::string("ERROR::ASSIMP::") + importer.GetErrorString());
        return {};
    }

    std::vector<nvh::PrimitiveMesh> primitives(scene->mNumMeshes, {});
    std::vector<nvh::Node>          instances;

    // create primitives using meshes
    for (uint32_t i = 0; i < scene->mNumMeshes; i++)
    {
        aiMesh* mesh = scene->mMeshes[i];
        auto&   prim = primitives[i];
        prim.vertices.reserve(mesh->mNumVertices);
        prim.triangles.reserve(mesh->mNumFaces);

        // vertex
        for (uint32_t j = 0; j < mesh->mNumVertices; j++)
        {
            nvh::PrimitiveVertex vertex{
                .p = toGlmVec3(mesh->mVertices[j]),
                .n = toGlmVec3(mesh->mNormals[j]),
                .t = toGlmVec2(mesh->mTextureCoords[0][j]),
            };
            prim.vertices.push_back(vertex);
        }

        // index
        for (unsigned int j = 0; j < mesh->mNumFaces; j++)
        {
            aiFace&                face = mesh->mFaces[j];
            nvh::PrimitiveTriangle index{
                glm::uvec3{
                    face.mIndices[0],
                    face.mIndices[1],
                    face.mIndices[2],
                },
            };
            prim.triangles.push_back(index);
        }

        // TODO - textures
        if (mesh->mMaterialIndex >= 0)
        {
            aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];
        }
    }

    if (loadInstances)
    {
        // TODO - create instance for nodes
        processNode(scene->mRootNode, scene);
    }

    return {primitives, instances};
}

#endif

// ======================================================================

#ifdef LOAD_METHOD_TINYOBJLOADER

std::tuple<std::vector<nvh::PrimitiveMesh>, std::vector<nvh::Node>>
loadGeometryFromFile(const std::string& filePath, bool loadInstancesv)
{
    // 判断文件的后缀是不是.obj
    if (filePath.substr(filePath.find_last_of(".") + 1) != "obj")
    {
        std::cout << "Error: File is not a .obj file." << std::endl;
        throw std::runtime_error("File is not a .obj file.");
        return {};
    }

    tinyobj::attrib_t                attrib;
    std::vector<tinyobj::shape_t>    shapes;
    std::vector<tinyobj::material_t> materials;
    std::string                      warn;
    std::string                      err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filePath.c_str());
    if (!warn.empty()) { std::cout << warn << std::endl; }
    if (!err.empty()) { std::cerr << err << std::endl; }
    if (!ret) { exit(1); }
    // FIXME - 你在想什么，tinyobjloader肯定要导入obj呀

    // pre allocate memory
    std::vector<nvh::PrimitiveMesh> primitives(shapes.size(), {});

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++)
    {
        // pre allocate memory
        auto& p = primitives[s];
        p.vertices.reserve(shapes[s].mesh.indices.size());
        p.triangles.reserve(shapes[s].mesh.indices.size() / 3);

        // Loop over faces(polygon)
        size_t index_offset   = 0;
        size_t triangle_index = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
        {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            if (fv != 3)
            {
                index_offset += fv;
                continue;
            }

            // Loop over vertices in the face.
            for (size_t v = 0; v < fv; v++)
            {
                // access to vertex
                tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                nvh::PrimitiveVertex vertex{};

                vertex.p = glm::vec3{
                    attrib.vertices[3 * size_t(idx.vertex_index) + 0],
                    attrib.vertices[3 * size_t(idx.vertex_index) + 1],
                    attrib.vertices[3 * size_t(idx.vertex_index) + 2],
                };
                vertex.n = (idx.normal_index >= 0) ?
                               glm::vec3{
                                   attrib.normals[3 * size_t(idx.normal_index) + 0],
                                   attrib.normals[3 * size_t(idx.normal_index) + 1],
                                   attrib.normals[3 * size_t(idx.normal_index) + 2],
                               } :
                               glm::vec3{0, 0, 0};
                // vertex.t = (idx.texcoord_index >= 0) ?
                //                          {attrib.texcoords[3 * size_t(idx.texcoord_index) + 0],
                //                           attrib.texcoords[3 * size_t(idx.texcoord_index) + 1],
                //                           attrib.texcoords[3 * size_t(idx.texcoord_index) + 2]} :
                //                          {0, 0, 0};
                vertex.t = (idx.texcoord_index >= 0) ?
                               glm::vec2{
                                   attrib.texcoords[2 * size_t(idx.texcoord_index) + 0],
                                   attrib.texcoords[2 * size_t(idx.texcoord_index) + 1],
                               } :
                               glm::vec2{0, 0};
                p.vertices.push_back(vertex);
            }

            // indices
            p.triangles.push_back(
                nvh::PrimitiveTriangle{{triangle_index, triangle_index + 1, triangle_index + 2}});
            triangle_index += 3;

            index_offset += fv;

            // material
            // shapes[s].mesh.material_ids[f];
        }
    }

    // no instances to create
    return {primitives, {}};
    // NOTE - std::vector<nvh::PrimitiveMesh> primitives = std::move(loadGeometryFromFile(......));

    /* NOTE -
    在返回局部变量时，编译器会自动应用返回值优化(RVO)
    显式使用 std::move 反而可能阻止RVO的发生
    C++17标准明确规定了这种情况下必须发生复制消除

    编译器会自动将返回值直接构造在调用者的栈空间中
    这种优化比手动使用 std::move 更高效
    只有在返回右值引用时才需要使用 std::move
    */
}

#endif

}  // namespace cookiekiss
