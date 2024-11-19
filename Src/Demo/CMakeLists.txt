#搜索每一个前缀是"demo_"的文件夹，包含他们
file(GLOB demo_folders  "*")
# 包含所有找到的子项目
foreach(folder ${demo_folders})
    if(IS_DIRECTORY ${folder})
        add_subdirectory(${folder})
    endif()
endforeach()