#ifdef VERTEX_SHADER
#define VS2FS out
#else
#define VS2FS in
#endif

uniform mat4 model_to_world;
uniform mat4 model_to_ndc;
uniform vec3 camera_position;
layout(location=0) uniform sampler3D sdf_texture;
uniform bool accurate_normals;

VS2FS vec3 v_normal;
VS2FS vec3 v_world_position;
VS2FS vec3 v_view;


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
	vec3 N;
	if (accurate_normals)
		N = sampleNormal(v_world_position.zyx).zyx; // 2650 fps at 80% usage
	else 
		N = v_normal; // 3400 fps at 40% usage
	N = normalize(N);

	//N = texture(normals_texture, v_world_position.zyx / 128.0).xyz * 2 - 1;
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