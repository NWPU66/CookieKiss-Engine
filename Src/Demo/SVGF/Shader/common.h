#ifdef __cplusplus
#    include <glm/glm.hpp>
using mat4 = glm::mat4;
using vec2 = glm::vec2;
using vec3 = glm::vec3;
#endif  // __cplusplus

struct PushContent
{
    mat4 m;
    mat4 mvp;
};
