#pragma once

// c/c++
#include <cstddef>
#include <string>
#include <vector>

// third party
#include "nvh/primitives.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// #define LOAD_METHOD_ASSIMP
// #define LOAD_METHOD_TINYOBJLOADER

#ifdef LOAD_METHOD_ASSIMP
#    ifdef LOAD_METHOD_TINYOBJLOADER
#        error                                                                                     \
            "Both LOAD_METHOD_ASSIMP and LOAD_METHOD_TINYOBJLOADER are defined. Only one can be defined."
#    endif
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

std::vector<nvh::PrimitiveMesh>& loadGeometryFromFile(const std::string& filePath);

#ifdef LOAD_METHOD_ASSIMP

std::vector<nvh::PrimitiveMesh>& loadGeometryFromFile(const std::string& filePath)
{
    // TODO -
}

#endif

// ================================================================

#ifdef LOAD_METHOD_TINYOBJLOADER

#    include <tinyobjloader/tiny_obj_loader.h>

std::vector<nvh::PrimitiveMesh>& loadGeometryFromFile(const std::string& filePath)
{
    tinyobj::attrib_t             attrib;
    std::vector<tinyobj::shape_t> shapes;
    // std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filePath.c_str());
    if (!warn.empty()) { std::cout << warn << std::endl; }
    if (!err.empty()) { std::cerr << err << std::endl; }
    if (!ret) { exit(1); }

    // pre allocate memory
    std::vector<nvh::PrimitiveMesh> primitives(shapes.size(), {});

    // Loop over shapes
    for (size_t s = 0; s < shapes.size(); s++)
    {
        // pre allocate memory
        auto& p = primitives[s];
        p.vertices.resize(shapes[s].mesh.indices.size());
        p.triangles.resize(shapes[s].mesh.indices.size());

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

                glm::vec3 position = {attrib.vertices[3 * size_t(idx.vertex_index) + 0],
                                      attrib.vertices[3 * size_t(idx.vertex_index) + 1],
                                      attrib.vertices[3 * size_t(idx.vertex_index) + 2]};
                glm::vec3 normal   = (idx.normal_index >= 0) ?
                                         {attrib.normals[3 * size_t(idx.normal_index) + 0],
                                          attrib.normals[3 * size_t(idx.normal_index) + 1],
                                          attrib.normals[3 * size_t(idx.normal_index) + 2]} :
                                         {0, 0, 0};
                // glm::vec3 texcoord = (idx.texcoord_index >= 0) ?
                //                          {attrib.texcoords[3 * size_t(idx.texcoord_index) + 0],
                //                           attrib.texcoords[3 * size_t(idx.texcoord_index) + 1],
                //                           attrib.texcoords[3 * size_t(idx.texcoord_index) + 2]} :
                //                          {0, 0, 0};
                glm::vec2 texcoord = (idx.texcoord_index >= 0) ?
                                         {attrib.texcoords[3 * size_t(idx.texcoord_index) + 0],
                                          attrib.texcoords[3 * size_t(idx.texcoord_index) + 1]} :
                                         {0, 0};
                p.vertices.emplace_back({position, normal, texcoord});
            }

            // indices
            p.triangles.push_back(glm::uvec3(vertex_index, vertex_index + 1, vertex_index + 2));
            triangle_index += 3;

            index_offset += fv;

            // material
            // shapes[s].mesh.material_ids[f];
        }
    }

    return primitives;
    // NOTE - std::vector<nvh::PrimitiveMesh> primitives = std::move(loadGeometryFromFile(......));
}

#endif

}  // namespace cookiekiss
