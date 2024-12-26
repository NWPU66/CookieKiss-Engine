#搜索每一个前缀是"demo_"的文件夹，包含他们
file(GLOB demo_folders  "*")
foreach(folder ${demo_folders})
    if(IS_DIRECTORY ${folder})
        get_filename_component(FOLDER_NAME ${folder} NAME)
        add_subdirectory(${folder})
    endif()
endforeach()
