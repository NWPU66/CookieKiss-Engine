cmake_minimum_required(VERSION 3.25)

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
	message("current platform: Linux ")
elseif(CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
	message("current platform: Windows ")
    set(CMAKE_TOOLCHAIN_FILE "E:/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(VCPKG_INCLUDE_DIR "E:/vcpkg/installed/x64-windows/include")
else()
	message("current platform: unkonw ")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS true)
project(CookieKiss-Engine
    VERSION 0.1.0
    DESCRIPTION "CookieKiss Engine"
    HOMEPAGE_URL "https://github.com/NWPU66/CookieKiss-Engine"
    LANGUAGES CXX)

# add subdirectory
set(3rdparty ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
# set(BASE_DIRECTORY ${3rdparty})
add_subdirectory(3rdparty)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/Src)
# NOTE - subdir 和 add_subdirectory 其实都一样，但文档中更推荐使用 add_subdirectory

# install path
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/Install)
