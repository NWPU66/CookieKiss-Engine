# subdirs(${CMAKE_CURRENT_SOURCE_DIR}/util)

# libraries
find_package(glm CONFIG REQUIRED)
find_package(Stb REQUIRED)

# cuda compiler
enable_language(CUDA)
# set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
# file(GLOB_RECURSE CUDA_LIBS "E:/cuda118_x64_lib/*.lib")
# message(${CUDA_LIBS})
# link_directories(${CUDA_ROOT}/lib/x64)

# set(CUDA_NVCC_FLAGS -g -G -dc)
# NOTE - nvidia nsight vscode 调试要点
# 首先，编译：使用nvcc编译时要加上“-g -G”，-g：将c++代码编译成可调式的版本，-G则针对cuda代码
# 其次，在vscode上配置launch.json文件，申明debuggerPath：cuda-gdb和program
# 最后按下小按钮即可
# 另外，没有使用-G编译的cuda代码，调试的时候会跳过cuda的部分

file(GLOB UTIL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.cuh)

file(GLOB_RECURSE CH_FILES ${CMAKE_CURRENT_SOURCE_DIR}/Ch*.cu)
foreach(CH_FILE ${CH_FILES})
    #获取文件名
    get_filename_component(CH_FILE_NAME ${CH_FILE} NAME_WE)
    #生成可执行文件
    add_executable(${CH_FILE_NAME} 
        ${CH_FILE})
    #添加头文件目录
    target_include_directories(${CH_FILE_NAME} PRIVATE
        ${VCPKG_INCLUDE_DIR}
        ${SRC_DIR}
        # ${CUDA_ROOT}/include
        ${CMAKE_CURRENT_SOURCE_DIR}
        )
    #添加链接库
    target_link_libraries(${CH_FILE_NAME} PRIVATE 
        # glm::glm
        # ${CUDA_LIBS}
        )
    #安装
    install(TARGETS ${CH_FILE_NAME} RUNTIME DESTINATION ./)
endforeach()