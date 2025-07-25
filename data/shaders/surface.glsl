#ifdef VERTEX_SHADER
#define VS2FS out
#else
#define VS2FS in
#endif

uniform mat4 model_to_world;
uniform mat4 model_to_ndc;
uniform vec3 camera_position;

VS2FS vec3 v_normal;
VS2FS vec3 v_world_position;
VS2FS vec3 v_view;


#ifdef VERTEX_SHADER

layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
void main() {
	v_normal = (model_to_world * vec4(normal, 0)).xyz;
	v_world_position = (model_to_world * vec4(position, 1)).xyz;
	v_view = camera_position - v_world_position;
	gl_Position = model_to_ndc * vec4(position, 1);
}

#endif


#ifdef FRAGMENT_SHADER

out vec4 fragColor;
void main() {
	vec3 L = normalize(vec3(1,3,2));
	vec3 N = v_normal;
	vec3 V = normalize(v_view);
	vec3 H = normalize(V + L);

	float NL = max(dot(N, L),0.0);
	float NH = max(dot(N, H),0.0) * sign(NL);

	vec3 diff;
	
	diff = vec3(clamp(NL,0.,1.));
	//diff = vec3(NL * 0.5 + 0.5) * 0.8;

	vec3 spec;
	spec = vec3(min(pow(NH*1.0,20)*0.2, 1.0));

	vec3 col = diff*0.5 + vec3(spec) + vec3(.0,.5,.9)*0.25;
    fragColor = vec4(col,1);
    //fragColor = vec4(N,1);
}

#endif