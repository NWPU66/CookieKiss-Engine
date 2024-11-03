set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# add sub directories
# subdirs(
#     ${CMAKE_CURRENT_SOURCE_DIR}/RT_in_One_Weekend
# )

# HelloCookieKissEngine programe
# find_package(glm CONFIG REQUIRED)
# find_package(imgui CONFIG REQUIRED)
add_executable(HelloCookieKissEngine HelloCookieKissEngine.cpp)
target_include_directories(HelloCookieKissEngine PRIVATE
    ${VCPKG_INCLUDE_DIR}
    ${SRC_DIR})
target_link_libraries(HelloCookieKissEngine PRIVATE 
    # imgui::imgui
    #glm::glm
    )
install(TARGETS HelloCookieKissEngine RUNTIME DESTINATION ./)
