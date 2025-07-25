#ifdef VERTEX_SHADER
#define VS2FS out
#else
#define VS2FS in
#endif

uniform mat4 model_to_world;
uniform mat4 model_to_ndc;

#ifdef VERTEX_SHADER

layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
void main() {
	gl_Position = model_to_ndc * vec4(position, 1);
}

#endif


#ifdef FRAGMENT_SHADER

out vec4 fragColor;
void main() {
    fragColor = vec4(1,1,1,0.05);
}

#endif