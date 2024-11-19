# nvpro_core
# include(${CMAKE_CURRENT_SOURCE_DIR}/nvpro_core/cmake/setup.cmake)
# subdirs(${CMAKE_CURRENT_SOURCE_DIR}/nvpro_core)

# nvpro_core pre_compiled lib
file(GLOB NVPRO_CORE_LIBS ${CMAKE_CURRENT_SOURCE_DIR}/nvpro_core_lib/*.lib)
message(STATUS "NVPRO_CORE_LIBS: ${NVPRO_CORE_LIBS}")
