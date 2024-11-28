# StreamlineDLSS
set(DEMO_NAME StreamlineDLSS)

# dependency 
find_package(Vulkan REQUIRED)
find_package(imgui CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(implot CONFIG REQUIRED)

# src
file(GLOB SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp ${CMAKE_CURRENT_SOURCE_DIR}/*.h)
file(GLOB IMGUI_SRC ${3rdparty}/nvpro_core/third_party/imgui/*.cpp)
file(GLOB IMPLOT_SRC ${3rdparty}/nvpro_core/third_party/implot/*.cpp)
add_executable(${DEMO_NAME} 
    ${SRC}
    # ${IMGUI_SRC}
    # ${IMPLOT_SRC}
    # ${3rdparty}/nvpro_core/third_party/imgui/backends/imgui_impl_vulkan.cpp
    )

# includes
target_include_directories(${DEMO_NAME} PRIVATE
    ${VCPKG_INCLUDE_DIR}
    ${3rdparty}/nvpro_core
    ${3rdparty}/nvpro_core/nvp
    ${3rdparty}/nvpro_core/third_party
    ${3rdparty}/nvpro_core/third_party/vma/include
    ${3rdparty}
    )

# linking
target_link_libraries(${DEMO_NAME} PRIVATE
    ${3rdparty}/nvpro_core_lib/nvpro_core_gl_vk_shaderc.lib
    Vulkan::Vulkan
    # imgui::imgui
    # implot::implot
    glfw
    # fmt::fmt-header-only
    )

#install
# install(TARGETS ${DEMO_NAME} 
#     RUNTIME DESTINATION ./demo
#     LIBRARY DESTINATION ./demo
#     ARCHIVE DESTINATION ./demo)
