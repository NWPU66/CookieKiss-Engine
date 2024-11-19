# VulkanMiniPatchTracer
set(DEMO_NAME VulkanMiniPatchTracer)

# dependency 
# find_package(glm CONFIG REQUIRED)
# find_package(glog CONFIG REQUIRED)
# find_package(assimp CONFIG REQUIRED)
# find_package(Stb REQUIRED)
find_package(Vulkan REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

# src
file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
add_executable(${DEMO_NAME} ${SRC})

# includes
target_include_directories(${DEMO_NAME} PUBLIC
    ${VCPKG_INCLUDE_DIR}
    ${3rdparty}/nvpro_core
    ${3rdparty}/nvpro_core/nvp
    )

# linking
target_link_libraries(${DEMO_NAME} PRIVATE
    ${3rdparty}/nvpro_core_lib/nvpro_core_gl_vk_shaderc.lib
    Vulkan::Vulkan
    imgui::imgui
    glfw
    fmt::fmt-header-only
    )

#install
# install(TARGETS ${DEMO_NAME} 
#     RUNTIME DESTINATION ./demo
#     LIBRARY DESTINATION ./demo
#     ARCHIVE DESTINATION ./demo)
