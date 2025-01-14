// Copyright 2016 The Shaderc Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <shaderc/shaderc.hpp>

namespace cookiekiss {

std::vector<uint32_t>
compileShaderFromFile(const std::string& filePath, shaderc_shader_kind kind, bool optimize = false);

// Returns GLSL shader source text after preprocessing.
std::string preprocess_shader(const std::string&  source_name,
                              shaderc_shader_kind kind,
                              const std::string&  source);

// Compiles a shader to SPIR-V assembly. Returns the assembly text
// as a string.
std::string compile_file_to_assembly(const std::string&  source_name,
                                     shaderc_shader_kind kind,
                                     const std::string&  source,
                                     bool                optimize = false);

// Compiles a shader to a SPIR-V binary. Returns the binary as
// a vector of 32-bit words.
std::vector<uint32_t> compile_file(const std::string&  source_name,
                                   shaderc_shader_kind kind,
                                   const std::string&  source,
                                   bool                optimize = false);

#ifdef ONLINESHADERCOMPILER_IMPLIMENTATION

std::vector<uint32_t>
compileShaderFromFile(const std::string& filePath, shaderc_shader_kind kind, bool optimize)
{
    auto          file     = std::filesystem::path(filePath);
    auto          fileName = file.filename().string();
    std::ifstream fs(file);
    std::string   content;
    fs >> content;

    // Preprocessing
    auto preprocessed = preprocess_shader(fileName, kind, content);

    // Compiling
    auto assembly = compile_file_to_assembly(fileName, kind, preprocessed, optimize);
    auto spirv    = compile_file(fileName, kind, preprocessed, optimize);

    fs.close();
    return spirv;
}

std::string preprocess_shader(const std::string&  source_name,
                              shaderc_shader_kind kind,
                              const std::string&  source)
{
    shaderc::Compiler       compiler;
    shaderc::CompileOptions options;

    // Like -DMY_DEFINE=1
    options.AddMacroDefinition("MY_DEFINE", "1");

    shaderc::PreprocessedSourceCompilationResult result =
        compiler.PreprocessGlsl(source, kind, source_name.c_str(), options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        std::cerr << result.GetErrorMessage();
        return "";
    }

    return {result.cbegin(), result.cend()};
}

std::string compile_file_to_assembly(const std::string&  source_name,
                                     shaderc_shader_kind kind,
                                     const std::string&  source,
                                     bool                optimize)
{
    shaderc::Compiler       compiler;
    shaderc::CompileOptions options;

    // Like -DMY_DEFINE=1
    options.AddMacroDefinition("MY_DEFINE", "1");
    if (optimize) options.SetOptimizationLevel(shaderc_optimization_level_size);

    shaderc::AssemblyCompilationResult result =
        compiler.CompileGlslToSpvAssembly(source, kind, source_name.c_str(), options);

    if (result.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        std::cerr << result.GetErrorMessage();
        return "";
    }

    return {result.cbegin(), result.cend()};
}

std::vector<uint32_t> compile_file(const std::string&  source_name,
                                   shaderc_shader_kind kind,
                                   const std::string&  source,
                                   bool                optimize)
{
    shaderc::Compiler       compiler;
    shaderc::CompileOptions options;

    // Like -DMY_DEFINE=1
    options.AddMacroDefinition("MY_DEFINE", "1");
    if (optimize) options.SetOptimizationLevel(shaderc_optimization_level_size);

    shaderc::SpvCompilationResult module =
        compiler.CompileGlslToSpv(source, kind, source_name.c_str(), options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success)
    {
        std::cerr << module.GetErrorMessage();
        return std::vector<uint32_t>();
    }

    return {module.cbegin(), module.cend()};
}

#endif

}  // namespace cookiekiss
