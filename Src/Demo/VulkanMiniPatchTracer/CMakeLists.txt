# VulkanMiniPatchTracer
set(DEMO_NAME VulkanMiniPatchTracer)

# dependency 
find_package(Vulkan REQUIRED)
# find_package(imgui CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)

# src
file(GLOB SRC 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
add_executable(${DEMO_NAME} ${SRC})
set_target_properties(${DEMO_NAME} PROPERTIES 
    FOLDER ${FOLDER_NAME})

# includes
target_include_directories(${DEMO_NAME} PUBLIC
    ${3rdparty}/nvpro_core
    ${3rdparty}/nvpro_core/third_party/imgui
    ${3rdparty}/nvpro_core/nvp
    ${VCPKG_INCLUDE_DIR}
    )

# linking
target_link_libraries(${DEMO_NAME} PRIVATE
    ${3rdparty}/Library/Debug/nvpro_core_gl_vk_shaderc.lib
    # ${3rdparty}/Library/Debug/fmt.lib
    # ${3rdparty}/Library/Debug/glfw3.lib
    ${3rdparty}/Library/Debug/imgui.lib
    ${3rdparty}/Library/Debug/implot.lib
    Vulkan::Vulkan
    # imgui::imgui
    glfw
    fmt::fmt-header-only
    )

#install
# install(TARGETS ${DEMO_NAME} 
#     RUNTIME DESTINATION ./demo
#     LIBRARY DESTINATION ./demo
#     ARCHIVE DESTINATION ./demo)
