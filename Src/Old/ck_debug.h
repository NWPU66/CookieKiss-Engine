#pragma once

#include <cstdint>

#include <glad/glad.h>

#ifdef NDEBUG
#    define GL_CHECK()
#else
#    define GL_CHECK() ck::glCheckError_(__FILE__, __LINE__)
#endif

namespace ck {

GLenum glCheckError_(const char* file, int32_t line);

void APIENTRY glDebugOutput(GLenum        source,
                            GLenum        type,
                            GLuint        id,
                            GLenum        severity,
                            GLsizei       length,
                            const GLchar* message,
                            const void*   userParam);

};  // namespace ck
