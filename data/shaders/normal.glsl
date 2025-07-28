#ifdef VERTEX_SHADER
#define VS2FS out
#else
#define VS2FS in
#endif

uniform mat4 model_to_world;
uniform mat4 model_to_ndc;
uniform float normals_size;
layout(location=0) uniform sampler3D sdf_texture;
uniform bool accurate_normals;

VS2FS vec3 v_normal;


vec3 sampleNormal(vec3 fp) {
	fp -= 0.5;
	vec3 fpFloor = floor(fp);
	vec3 t = fp - fpFloor;
	ivec3 p = ivec3(floor(fp));

	float d[3][3][3];

	for (int x = 0; x < 3; ++x)
	for (int y = 0; y < 3; ++y)
	for (int z = 0; z < 3; ++z)
		d[x][y][z] = texelFetch(sdf_texture, p + ivec3(x,y,z), 0).x;

	vec3 n[2][2][2];

	for (int x = 0; x < 2; ++x)
	for (int y = 0; y < 2; ++y)
	for (int z = 0; z < 2; ++z)
		n[x][y][z] = normalize(vec3(
			d[x+0][y+0][z+0] - d[x+1][y+0][z+0] +
			d[x+0][y+0][z+1] - d[x+1][y+0][z+1] +
			d[x+0][y+1][z+0] - d[x+1][y+1][z+0] +
			d[x+0][y+1][z+1] - d[x+1][y+1][z+1],
			d[x+0][y+0][z+0] - d[x+0][y+1][z+0] +
			d[x+0][y+0][z+1] - d[x+0][y+1][z+1] +
			d[x+1][y+0][z+0] - d[x+1][y+1][z+0] +
			d[x+1][y+0][z+1] - d[x+1][y+1][z+1],
			d[x+0][y+0][z+0] - d[x+0][y+0][z+1] +
			d[x+0][y+1][z+0] - d[x+0][y+1][z+1] +
			d[x+1][y+0][z+0] - d[x+1][y+0][z+1] +
			d[x+1][y+1][z+0] - d[x+1][y+1][z+1]
		));

	return mix(mix(
		mix(n[0][0][0], n[0][0][1], t. z),
		mix(n[0][1][0], n[0][1][1], t. z), t.y), mix(
		mix(n[1][0][0], n[1][0][1], t. z),
		mix(n[1][1][0], n[1][1][1], t. z), t.y), t.x);
}


#ifdef VERTEX_SHADER

struct BufferVertex {
	float[3] position;
	float[3] normal;
};

struct Vertex {
	vec3 position;
	vec3 normal;
};

layout(std430, binding=0) restrict readonly buffer Vertices { BufferVertex vertices[]; };

vec2 to_vec2(float[2] f) { return vec2(f[0], f[1]); }
vec3 to_vec3(float[3] f) { return vec3(f[0], f[1], f[2]); }
Vertex unfuck(BufferVertex b) {
	Vertex v;
	v.position = to_vec3(b.position);
	v.normal = to_vec3(b.normal);
	return v;
}

void main() {
	BufferVertex bv = vertices[gl_VertexID >> 1];
	Vertex v = unfuck(bv);
	vec3 position = v.position;
	
	vec3 v_world_position = (model_to_world * vec4(position, 1)).xyz;

	if (accurate_normals) {
		v_normal = sampleNormal(v_world_position.zyx).zyx;
	} else {
		v_normal = v.normal;
	}

	v_normal = v_normal * 0.5 + 0.5;

	if (bool(gl_VertexID & 1)) {
		position += v_normal * normals_size;
	}
	gl_Position = model_to_ndc * vec4(position, 1);
}

#endif


#ifdef FRAGMENT_SHADER

out vec4 fragColor;
void main() {
    fragColor = vec4(v_normal,1);
}

#endif