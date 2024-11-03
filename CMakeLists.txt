cmake_minimum_required(VERSION 3.25)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_HOST_SYSTEM_NAME MATCHES "Linux")
	message("current platform: Linux ")
elseif(CMAKE_HOST_SYSTEM_NAME MATCHES "Windows")
	message("current platform: Windows ")
    set(CMAKE_TOOLCHAIN_FILE "E:/vcpkg/scripts/buildsystems/vcpkg.cmake")
    set(VCPKG_INCLUDE_DIR "E:/vcpkg/installed/x64-windows/include")
else()
	message("current platform: unkonw ")
endif()

set(CMAKE_CXX_STANDARD 17)
project(CookieKiss-Engine
    VERSION 0.1.0
    DESCRIPTION "CookieKiss Engine"
    HOMEPAGE_URL "https://github.com/NWPU66/CookieKiss-Engine"
    LANGUAGES CXX CUDA)

# add subdirectory
# subdirs(${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)
subdirs(${CMAKE_CURRENT_SOURCE_DIR}/Src)

# install path
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/Install)
