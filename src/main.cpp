// couldn't make gather faster than shuffles

#define TL_IMPL
#include "common.h"
#include <tl/main.h>
#include <tl/math_random.h>
#include <tl/file.h>
#include <tl/precise_time.h>
#include <tl/opengl.h>
#include <tl/win32.h>
#include <tl/includer.h>
#include <tl/linear_set.h>
#include <tl/mesh.h>

#pragma push_macro("assert")
#include <imgui.h>
#include <backends/imgui_impl_win32.h>
#include <backends/imgui_impl_opengl3.h>
#pragma pop_macro("assert")

gl::Functions gl_functions;
gl::Functions *tl_opengl_functions() { return &gl_functions; };

void sdf_to_triangles_starting(v3u size, Span<f32> sdf, GList<Vertex> &vertices, GList<u32> &indices, GList<u32> &index_grid);
void sdf_to_triangles_sse(v3u size, Span<f32> sdf, Span<u8> negative_bitfield, Span<u8> surface_bitfield, GList<Vertex> &vertices, GList<u32> &indices, GList<u32> &index_grid);
void sdf_to_triangles_avx(v3u size, Span<f32> sdf, Span<u8> negative_bitfield, Span<u8> surface_bitfield, GList<Vertex> &vertices, GList<u32> &indices, GList<u32> &index_grid);

#define sdf_to_triangles sdf_to_triangles_avx

forceinline s16 gradient_noise_s16(v3s coordinate, s32 step) {
	v3s floored = floor(coordinate, step);
	v3s tile = floored / step;
	v3f local = (v3f)(coordinate - floored) * reciprocal((f32)step);

	v3f t0 = local;
	v3f t1 = t0 - 1;

	static constexpr v3f directions[] = {
		normalize(v3f{ 1, 1, 1}),
		normalize(v3f{-1, 1, 1}),
		normalize(v3f{ 1,-1, 1}),
		normalize(v3f{-1,-1, 1}),
		normalize(v3f{ 1, 1,-1}),
		normalize(v3f{-1, 1,-1}),
		normalize(v3f{ 1,-1,-1}),
		normalize(v3f{-1,-1,-1})
	};
	static_assert(is_power_of_2(count_of(directions)));

	v3f t = smoothstep3(local);

	auto get_direction = [&](v3s offset) { return directions[(DefaultRandomizer{}.template random<u32>(tile + offset) >> 13)]; };
	v3f g000 = get_direction({0, 0, 0});
	v3f g100 = get_direction({1, 0, 0});
	v3f g010 = get_direction({0, 1, 0});
	v3f g110 = get_direction({1, 1, 0});
	v3f g001 = get_direction({0, 0, 1});
	v3f g101 = get_direction({1, 0, 1});
	v3f g011 = get_direction({0, 1, 1});
	v3f g111 = get_direction({1, 1, 1});
	f32 v000 = dot(g000, v3f{ t0.x, t0.y, t0.z});
	f32 v100 = dot(g100, v3f{ t1.x, t0.y, t0.z});
	f32 v010 = dot(g010, v3f{ t0.x, t1.y, t0.z});
	f32 v110 = dot(g110, v3f{ t1.x, t1.y, t0.z});
	f32 v001 = dot(g001, v3f{ t0.x, t0.y, t1.z});
	f32 v101 = dot(g101, v3f{ t1.x, t0.y, t1.z});
	f32 v011 = dot(g011, v3f{ t0.x, t1.y, t1.z});
	f32 v111 = dot(g111, v3f{ t1.x, t1.y, t1.z});

	return
		lerp(
			lerp(lerp(v000, v100, t.x), lerp(v010, v110, t.x), t.y),
			lerp(lerp(v001, v101, t.x), lerp(v011, v111, t.x), t.y),
			t.z
		) / sqrt3 + 0.5f;
}

constexpr int N = 128;

struct Sdf {
	f32 sdf[N][N][N];
	
	// bit for each value in sdf: 1 if negative, 0 if positive
	u8 negative_bitfield[N*N*N/8];

	// bit for each 2x2x2 cube in sdf: 1 if there is a surface, 0 if not.
	// NOTE: the dimension of this should be N-1, but i want to keep indices the same as in other arrays,
	//       so there's a bit of wasted bits: ~2% when N is 128. (127*127 + 127)*3 + 1.
	u8 surface_bitfield[N*N*N/8];
} sdf;

void verify_sdf_aux_bit_fields(v3u size, Span<f32> sdf, Span<u8> negative_bitfield, Span<u8> surface_bitfield) {
	auto at = [&](auto &arr, u32 x, u32 y, u32 z) -> decltype(auto) {
		return arr[x * (size.y*size.z) + y * (size.z) + z];
	};

	for (u32 x = 0; x < size.x; ++x) {
	for (u32 y = 0; y < size.y; ++y) {
	for (u32 z = 0; z < size.z; ++z) {
		assert(((negative_bitfield.data[(x*size.y*size.z + y*size.z + z)/8] >> (z%8)) & 1) == (*(u32 *)&at(sdf.data,x,y,z)>>31));
	}
	}
	}
	for (u32 x = 0; x < size.x-1; ++x) {
	for (u32 y = 0; y < size.y-1; ++y) {
	for (u32 z = 0; z < size.z-1; ++z) {
		bool n0 = (negative_bitfield.data[((x+0)*size.y*size.z + (y+0)*size.z + (z+0))/8] >> ((z+0)%8)) & 1;
		bool n1 = (negative_bitfield.data[((x+0)*size.y*size.z + (y+0)*size.z + (z+1))/8] >> ((z+1)%8)) & 1;
		bool n2 = (negative_bitfield.data[((x+0)*size.y*size.z + (y+1)*size.z + (z+0))/8] >> ((z+0)%8)) & 1;
		bool n3 = (negative_bitfield.data[((x+0)*size.y*size.z + (y+1)*size.z + (z+1))/8] >> ((z+1)%8)) & 1;
		bool n4 = (negative_bitfield.data[((x+1)*size.y*size.z + (y+0)*size.z + (z+0))/8] >> ((z+0)%8)) & 1;
		bool n5 = (negative_bitfield.data[((x+1)*size.y*size.z + (y+0)*size.z + (z+1))/8] >> ((z+1)%8)) & 1;
		bool n6 = (negative_bitfield.data[((x+1)*size.y*size.z + (y+1)*size.z + (z+0))/8] >> ((z+0)%8)) & 1;
		bool n7 = (negative_bitfield.data[((x+1)*size.y*size.z + (y+1)*size.z + (z+1))/8] >> ((z+1)%8)) & 1;
		bool surface = (surface_bitfield[(x*size.y*size.z + y*size.z + z)/8] >> (z%8)) & 1;
		int sum = n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7;
		if (surface) {
			assert(sum > 0 && sum < 8);
		} else {
			assert(sum == 0 || sum == 8);
		}
	}
	}
	}
}

void update_sdf_auxiliary_fields(aabb<v3s> range = {{}, {N,N,N}}) {
	timed_block_always("aux update");

	// range = {{}, {N,N,N}};

	for (int x = range.min.x; x < range.max.x; ++x) {
	for (int y = range.min.y; y < range.max.y; ++y) {
	for (int z = range.min.z; z < range.max.z; ++z) {
		u8 &byte = sdf.negative_bitfield[(x*N*N + y*N + z)/8];
		bool is_negative = *(u32 *)&sdf.sdf[x][y][z] >> 31;
		byte &= ~(1 << (z % 8));
		byte |= is_negative << (z % 8);
	}
	}
	}
	for (int x = max(0,range.min.x-1); x < min(range.max.x, N-1); ++x) {
	for (int y = max(0,range.min.y-1); y < min(range.max.y, N-1); ++y) {
	for (int z = max(0,range.min.z-1); z < min(range.max.z, N-1); ++z) {

		u8 r = 
			(((*(u16*)&sdf.negative_bitfield[((x+0)*N*N + (y+0)*N + z)/8] >> (z%8)) & 0x3) << 0) |
			(((*(u16*)&sdf.negative_bitfield[((x+0)*N*N + (y+1)*N + z)/8] >> (z%8)) & 0x3) << 2) |
			(((*(u16*)&sdf.negative_bitfield[((x+1)*N*N + (y+0)*N + z)/8] >> (z%8)) & 0x3) << 4) |
			(((*(u16*)&sdf.negative_bitfield[((x+1)*N*N + (y+1)*N + z)/8] >> (z%8)) & 0x3) << 6);
		
		bool surface = (u8)(r + 1) >= 2;
		
		u8 &byte = sdf.surface_bitfield[(x*N*N + y*N + z)/8];
		byte &= ~(1 << (z % 8));
		byte |= surface << (z % 8);
	}
	}
	}

	//verify_sdf_aux_bit_fields({N,N,N}, flatten(sdf.sdf), sdf.negative_bitfield, sdf.surface_bitfield);
}

void write_sdf_to_file() {
	for (int x = 0; x < N; ++x) {
	for (int y = 0; y < N; ++y) {
	for (int z = 0; z < N; ++z) {
		//sdf[x][y][z] = gradient_noise_s16({x,y,z}, 8) - 0.5 - (sinf(y * tau / 32.0f)) * 0.25f;
		//sdf[x][y][z] = value_noise_smooth<f32>(V3f(x,y,z) / 8) - 0.5 - (y - N/2) / 32.0f;

		v3f p = V3f(x,y,z) - N/2;

		f32 d = (gradient_noise({x,y,z}, 8) - 0.5f) * 32 - (length(p) - N / 3); // MAIN
		//d = clamp(d, -1.0f, 1.0f);
		//d = roundf(map(d, -1.f, 1.f, -1.f, 254.f));

		sdf.sdf[x][y][z] = d;
	}
	}
	}
	update_sdf_auxiliary_fields();

	write_entire_file(u8"sdf.bin"s, value_as_bytes(sdf));
}

void load_sdf_from_file() {
	auto buffer = read_entire_file(u8"sdf.bin"s).value();
	memcpy(&sdf, buffer.data, sizeof(sdf));
	free(buffer);
}


GList<Vertex> vertices;
GList<u32> indices;
GList<u32> index_grid;

void write_mesh_to_obj() {
	StringBuilder builder;
		
	for (auto v : vertices) {
		append_format(builder, "v {} {} {}\n", v.position.x, v.position.y, v.position.z);
	}
	for (auto v : vertices) {
		append_format(builder, "vn {} {} {}\n", v.normal.x, v.normal.y, v.normal.z);
	}

	for (umm i = 0; i < indices.count; i += 3) {
		int a = indices[i+0]+1;
		int b = indices[i+1]+1;
		int c = indices[i+2]+1;
		append_format(builder, "f {}//{} {}//{} {}//{}\n", a,a,b,b,c,c);
	}

	write_entire_file(u8"chunk.obj"s, builder);
}

HWND hwnd;
v2s screen_size;


LRESULT wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
	extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
	if (ImGui_ImplWin32_WndProcHandler(hwnd, msg, wp, lp))
		return true;

	switch (msg) {
		case WM_CLOSE: {
			PostQuitMessage(0);
			return 0;
		}
		case WM_SIZE: {
			v2s new_size = {
				LOWORD(lp),
				HIWORD(lp),
			};

			if (!new_size.x || !new_size.y || (wp == SIZE_MINIMIZED))
				return 0;

			screen_size = new_size;
			return 0;
		}
	}
	return DefWindowProcW(hwnd, msg, wp, lp);
}

bool init_window() {
	hwnd = create_class_and_window(u8"testing"s, wnd_proc, u8"Testing"s);
	
	init_rawinput(RawInput_mouse);

	if (!gl::init_opengl((NativeWindowHandle)hwnd, gl::Init_debug)) {
		current_logger.error("gl::init_opengl failed. {}", win32_error());
		return false;
	}

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDepthFunc(GL_LESS);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO &io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;
	io.ConfigDragClickToInputText = true;
	
	auto &style = ImGui::GetStyle();
	style.HoverDelayShort = 2.0f;
	
	ImGui::StyleColorsDark();

	ImGui_ImplWin32_InitForOpenGL(hwnd);
	ImGui_ImplOpenGL3_Init();

	return true;
}

struct GpuMesh {
	GLuint vb, ib, va;
	u32 vertex_count, index_count;
};

GpuMesh create_gpu_mesh(Span<Vertex> vertices, Span<u32> indices) {
	GpuMesh mesh = {};
	
	glCreateBuffers(1, &mesh.vb);
	glNamedBufferData(mesh.vb, sizeof(vertices[0]) * vertices.count, vertices.data, GL_STATIC_DRAW);

	glCreateBuffers(1, &mesh.ib);
	glNamedBufferData(mesh.ib, sizeof(indices[0]) * indices.count, indices.data, GL_STATIC_DRAW);

	glCreateVertexArrays(1, &mesh.va);
	glVertexArrayVertexBuffer(mesh.va, 0, mesh.vb, 0, sizeof(Vertex));
	glVertexArrayElementBuffer(mesh.va, mesh.ib);
	glEnableVertexArrayAttrib(mesh.va, 0); glVertexArrayAttribBinding(mesh.va, 0, 0); glVertexArrayAttribFormat(mesh.va, 0, 3, GL_FLOAT, false, offsetof(Vertex, position));
	glEnableVertexArrayAttrib(mesh.va, 1); glVertexArrayAttribBinding(mesh.va, 1, 0); glVertexArrayAttribFormat(mesh.va, 1, 3, GL_FLOAT, false, offsetof(Vertex, normal));

	mesh.vertex_count = vertices.count;
	mesh.index_count = indices.count;

	return mesh;
}

void update_gpu_mesh(GpuMesh &mesh, Span<Vertex> vertices, Span<u32> indices) {
	glNamedBufferData(mesh.vb, sizeof(vertices[0]) * vertices.count, vertices.data, GL_STATIC_DRAW);
	glNamedBufferData(mesh.ib, sizeof(indices[0]) * indices.count, indices.data, GL_STATIC_DRAW);

	mesh.vertex_count = vertices.count;
	mesh.index_count = indices.count;
}

template <umm size_x, umm size_y, umm size_z>
GLuint create_texture(v3u8 const (&pixels)[size_z][size_y][size_x]) {
	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_3D, texture);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB8, size_x, size_y, size_z, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
	return texture;
}
template <umm size_x, umm size_y, umm size_z>
void update_texture(GLuint &texture, v3u8 const (&pixels)[size_z][size_y][size_x]) {
	glBindTexture(GL_TEXTURE_3D, texture);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB8, size_x, size_y, size_z, 0, GL_RGB, GL_UNSIGNED_BYTE, pixels);
}

template <umm size_x, umm size_y, umm size_z>
GLuint create_texture(f32 const (&pixels)[size_z][size_y][size_x]) {
	GLuint texture;
	glGenTextures(1, &texture);
	glBindTexture(GL_TEXTURE_3D, texture);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_REPEAT);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, size_x, size_y, size_z, 0, GL_RED, GL_FLOAT, pixels);
	return texture;
}
template <umm size_x, umm size_y, umm size_z>
void update_texture(GLuint &texture, f32 const (&pixels)[size_z][size_y][size_x]) {
	glBindTexture(GL_TEXTURE_3D, texture);
	glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, size_x, size_y, size_z, 0, GL_RED, GL_FLOAT, pixels);
}


void bind_texture(GLuint program, u32 slot, char const *name, GLuint texture, GLuint sampler = 0, GLenum target = GL_TEXTURE_2D) {
	glActiveTexture(GL_TEXTURE0 + slot); 
	glBindTexture(target, texture); 
	gl::set_uniform(program, name, (int)slot); 
	glBindSampler(slot, sampler);
}

f64 current_time;
f32 frame_time = 1.0f / 60;

struct Program {
	struct SourceFile : includer::SourceFileBase {
		FileTime last_write_time = 0;
			
		void init() {
			last_write_time = get_file_write_time(path).value_or(0);
		}
		bool was_modified() {
			return last_write_time < get_file_write_time(path).value_or(0);
		}
	};

	String path;
	includer::Includer<SourceFile> includer;
	List<utf8> text;

	GLenum vs = 0;
	GLenum fs = 0;
	GLuint program = 0;

	bool failed_to_load = false;
	f64 retry_after = 0;

	void reload() {
		println("Recompiling {}", path);

		bool ok = includer.load(path, &text, includer::LoadOptions{
			.append_location_info = +[](StringBuilder &builder, String path, u32 line) {
				append_format(builder, "\n#line {} \"{}\"\n", line, path);
			}
		});

		if (!ok) {
			failed_to_load = true;
			retry_after = current_time + 0.1;
			return;
		}
		failed_to_load = false;

		GLuint vs = gl::create_shader(GL_VERTEX_SHADER, 430, true, as_chars(text));
		GLuint fs = gl::create_shader(GL_FRAGMENT_SHADER, 430, true, as_chars(text));
		GLuint program = gl::create_program({.vertex = vs, .fragment = fs});

		if (vs && fs && program) {
			glDeleteProgram(this->program);
			glDeleteShader(this->fs);
			glDeleteShader(this->vs);
			this->vs = vs;
			this->fs = fs;
			this->program = program;
		}
	}
	void free() {
		tl::free(text);
	}
	bool needs_reload() {
		if (failed_to_load && current_time >= retry_after)
			return true;

		for (auto &source_file : includer.source_files) {
			if (source_file.was_modified()) {
				return true;
			}
		}
		return false;
	}
};

String program_path;
String program_directory;

String resource_path(String relative_path) {
	return tformat(u8"{}/../data/{}"s, program_directory, relative_path);
}

extern "C" void bench_h();
extern "C" void bench_v();

s32 tl_main(Span<String> args) {
	set_console_encoding(Encoding::utf8);
	
	program_path = args[0];
	replace_inplace(program_path, u8'\\', u8'/');
	program_directory = parse_path(program_path).directory;

	if (1) {
		write_sdf_to_file();
	} else {
		load_sdf_from_file();
	}
	

	init_window();
	
	sdf_to_triangles({N,N,N}, flatten(sdf.sdf), sdf.negative_bitfield, sdf.surface_bitfield, vertices, indices, index_grid);
	GpuMesh mesh = create_gpu_mesh(vertices, indices);
	GLuint sdf_texture = create_texture(sdf.sdf);
	
	GLuint linear_sampler;
	glGenSamplers(1, &linear_sampler);
	glSamplerParameteri(linear_sampler, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glSamplerParameteri(linear_sampler, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glSamplerParameteri(linear_sampler, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glSamplerParameteri(linear_sampler, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glSamplerParameteri(linear_sampler, GL_TEXTURE_WRAP_R, GL_REPEAT);


	Program surface_program = {.path = to_list(resource_path(u8"shaders/surface.glsl"s))};
	Program wireframe_program = {.path = to_list(resource_path(u8"shaders/wireframe.glsl"s))};
	
	Program *all_programs[] = { &surface_program, &wireframe_program };

	for (auto program : all_programs) {
		program->reload();
	}
	
	PreciseTimer frame_timer = create_precise_timer();
	v2s old_screen_size = {-1,-1};
	
	v3f camera_position = V3f(N)*0.9;
	v3f camera_angles = V3f(pi/5, -pi/4, 0);

	f64 smoothed_fps = 0;
	//f32 smoothed_fps_lerp_t = 0.1;

	MSG msg = {};
	while (1) {
		while (PeekMessageW(&msg, 0, 0, 0, PM_REMOVE)) {
			switch (msg.message) {
				case WM_QUIT: {
					return 0;
				}
			}
			TranslateMessage(&msg);
			DispatchMessageW(&msg);
		}


		for (auto program : all_programs) {
			if (program->needs_reload()) {
				program->reload();
			}
		}

		// 
		// Begin GUI frame
		// 
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplWin32_NewFrame();
		ImGui::NewFrame();
		
		// 
		// GUI
		//
	
		auto bench = []<int id>(char const *name, auto exec){
			static f64 time_sum = 0;
			static f64 time_sum_div = 0;
			static f64 time_min = infinity<f64>;
			if (ImGui::Button(name)) {
				time_sum = 0;
				time_sum_div = 0;
				for (umm i = 0; i < 256*4; ++i) {
					auto timer = create_precise_timer();
					exec();
					auto time = elapsed_time(timer);
					time_sum += time;
					time_sum_div++;
					time_min = min(time_min, time);
				}
			}
			ImGui::SameLine();

			auto time = time_sum / time_sum_div;
			ImGui::Text("%s", tformat("avg: {}ms ({}/s) | best: {}ms ({}/s)\0"s, time * 1000, format_bytes(sizeof(sdf) / time), time_min * 1000, format_bytes(sizeof(sdf) / time_min)).data);
		};

		static bool enable_wireframe = false;
		static bool vsync = true;
		static bool accurate_normals = false;

		if (ImGui::Begin("Window")) {
			ImGui::Text("%.1f fps", smoothed_fps);
			//ImGui::SliderFloat("Smoothed FPS lerp T", &smoothed_fps_lerp_t, 0.001, 1, "%.3f", ImGuiSliderFlags_Logarithmic);
			if (ImGui::Checkbox("VSync", &vsync)) {
				wglSwapIntervalEXT(vsync);
			}
			ImGui::Checkbox("Wireframe", &enable_wireframe);
			ImGui::Checkbox("Accurate normals", &accurate_normals);
			if (ImGui::Button("Remesh Starting")) {
				update_sdf_auxiliary_fields();
				sdf_to_triangles_starting({N,N,N}, flatten(sdf.sdf), vertices, indices, index_grid);
				update_gpu_mesh(mesh, vertices, indices);
				update_texture(sdf_texture, sdf.sdf);
			}
			if (ImGui::Button("Remesh SSE")) {
				update_sdf_auxiliary_fields();
				sdf_to_triangles({N,N,N}, flatten(sdf.sdf), sdf.negative_bitfield, sdf.surface_bitfield, vertices, indices, index_grid);
				update_gpu_mesh(mesh, vertices, indices);
				update_texture(sdf_texture, sdf.sdf);
			}
			if (ImGui::Button("Calculate normals from triangles")) {
				for (auto &vertex : vertices) {
					vertex.normal = {};
				}

				for (u32 i = 0; i < indices.count; i += 3) {
					auto &a = vertices[indices[i + 0]];
					auto &b = vertices[indices[i + 1]];
					auto &c = vertices[indices[i + 2]];
					v3f normal = normalize(cross(b.position - a.position, c.position - a.position));
					a.normal += normal;
					b.normal += normal;
					c.normal += normal;
				}

				for (auto &vertex : vertices) {
					vertex.normal = normalize(vertex.normal);
				}
				update_gpu_mesh(mesh, vertices, indices);
			}
			bench.operator()<0>("Starting bench", [&]{sdf_to_triangles_starting({N,N,N}, flatten(sdf.sdf), vertices, indices, index_grid);});
			bench.operator()<1>("SSE bench"   , [&]{sdf_to_triangles_sse({N,N,N}, flatten(sdf.sdf), sdf.negative_bitfield, sdf.surface_bitfield, vertices, indices, index_grid);});
			bench.operator()<1>("AVX bench"   , [&]{sdf_to_triangles_avx({N,N,N}, flatten(sdf.sdf), sdf.negative_bitfield, sdf.surface_bitfield, vertices, indices, index_grid);});
			bench.operator()<2>("H bench"     , [&]{bench_h();});
			bench.operator()<3>("V bench"     , [&]{bench_v();});
		}
		ImGui::End();
		
		enum class Brush {
			smooth,
			sphere,
		};
		static Brush brush;
		static float brush_radius = 5;
		
		if (ImGui::Begin("Tools")) {
			
			if (ImGui::Button("Smooth")) brush = Brush::smooth;
			ImGui::SameLine();
			if (ImGui::Button("Sphere")) brush = Brush::sphere;

			ImGui::SliderFloat("Radius", &brush_radius, 1, 32, "%.3f", ImGuiSliderFlags_Logarithmic);

			ImGui::TextUnformatted("Keys:");
			ImGui::TextUnformatted(" Shift - faster");
			ImGui::TextUnformatted(" Alt   - slower");
			ImGui::TextUnformatted(" Ctrl  - invert");
		}
		ImGui::End();

		ImGui::Render();
		
		// 
		// Update state
		//
		
		m4 world_to_ndc = m4::perspective_right_handed((f32)screen_size.x / screen_size.y, radians(75), 0.01f, 1000.0f) * m4::rotation_r_yxz(camera_angles) * m4::translation(-camera_position);
		m4 ndc_to_world = inverse(world_to_ndc);

		static ImVec2 prev_mouse_position;
		ImVec2 mouse_position = ImGui::GetMousePos();
		ImVec2 mouse_delta = {mouse_position.x - prev_mouse_position.x, mouse_position.y - prev_mouse_position.y};
		prev_mouse_position = mouse_position;

		auto raycast_world = [&] {
			v2f cursor_pos = std::bit_cast<v2f>(ImGui::GetMousePos());
			v2f cursor_ndc = map<v2f,v2f>(cursor_pos, {}, (v2f)screen_size, {-1,1}, {1,-1});
			v4f cursor_wpos4 = ndc_to_world * V4f(cursor_ndc,0,1);
			v3f cursor_dir = normalize(cursor_wpos4.xyz / cursor_wpos4.w - camera_position);
			ray<v3f> cursor_ray = {
				camera_position,
				cursor_dir,
			};

			Optional<RaycastHit<v3f>> hit = {};

			for (umm i = 0; i < indices.count; i += 3) {
				auto a = vertices[indices[i+0]].position;
				auto b = vertices[indices[i+1]].position;
				auto c = vertices[indices[i+2]].position;
				hit = min(hit, raycast(cursor_ray, triangle<v3f>{a,b,c}));
			}

			return hit;
		};

		if (!ImGui::GetIO().WantCaptureMouse) {
			if (ImGui::IsKeyDown(ImGuiKey_MouseRight)) {
				camera_angles.x += mouse_delta.y * 0.003f;
				camera_angles.y += mouse_delta.x * 0.003f;
			}
		
			f32 force = 15;
			if (ImGui::IsKeyDown(ImGuiKey_LeftShift)) force *= 10;
			if (ImGui::IsKeyDown(ImGuiKey_LeftAlt)) force /= 10;
			if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl)) force *= -1;
			
			int R = ceilf(brush_radius);

			switch (brush) {
				case Brush::smooth: {
					if (ImGui::IsKeyDown(ImGuiKey_MouseLeft)) {
						auto hit = raycast_world();
						if (hit) {
							REDECLARE_VAL(hit, hit.value());

							auto hit_position_int = (v3s)floor(hit.position);

							aabb<v3s> range = {max(V3s(0),hit_position_int - R), min(hit_position_int + R + 1, V3s(N))};

							for (s32 x = range.min.x; x < range.max.x; ++x) {
							for (s32 y = range.min.y; y < range.max.y; ++y) {
							for (s32 z = range.min.z; z < range.max.z; ++z) {
								v3s p = {x,y,z};
								sdf.sdf[p.x][p.y][p.z] -= frame_time * force * map_clamped<f32,f32>(distance((v3f)p, hit.position), 0, brush_radius, 1, 0);
							}
							}
							}

							update_sdf_auxiliary_fields(range);
							sdf_to_triangles({N,N,N}, flatten(sdf.sdf), sdf.negative_bitfield, sdf.surface_bitfield, vertices, indices, index_grid);
							update_gpu_mesh(mesh, vertices, indices);
							update_texture(sdf_texture, sdf.sdf);
						}
					}
					break;
				}
				case Brush::sphere: {
					if (ImGui::IsKeyPressed(ImGuiKey_MouseLeft, false)) {
						auto hit = raycast_world();
						if (hit) {
							REDECLARE_VAL(hit, hit.value());
							
							auto hit_position_int = (v3s)floor(hit.position);

							aabb<v3s> range = {max(V3s(0),hit_position_int - R), min(hit_position_int + R + 1, V3s(N))};

							if (force > 0) {
								for (s32 x = range.min.x; x < range.max.x; ++x) {
								for (s32 y = range.min.y; y < range.max.y; ++y) {
								for (s32 z = range.min.z; z < range.max.z; ++z) {
									v3s p = {x,y,z};
									sdf.sdf[p.x][p.y][p.z] = min(sdf.sdf[p.x][p.y][p.z], map_clamped<f32,f32>(distance((v3f)p, hit.position), brush_radius - 1, brush_radius, -1, 1));
								}
								}
								}
							} else {
								for (s32 x = range.min.x; x < range.max.x; ++x) {
								for (s32 y = range.min.y; y < range.max.y; ++y) {
								for (s32 z = range.min.z; z < range.max.z; ++z) {
									v3s p = {x,y,z};
									sdf.sdf[p.x][p.y][p.z] = max(sdf.sdf[p.x][p.y][p.z], map_clamped<f32,f32>(distance((v3f)p, hit.position), brush_radius - 1, brush_radius, 1, -1));
								}
								}
								}
							}

							update_sdf_auxiliary_fields(range);
							sdf_to_triangles({N,N,N}, flatten(sdf.sdf), sdf.negative_bitfield, sdf.surface_bitfield, vertices, indices, index_grid);
							update_gpu_mesh(mesh, vertices, indices);
							update_texture(sdf_texture, sdf.sdf);
						}
					}
					break;
				}
			}
		}

		f32 speed = 15;
		if (ImGui::IsKeyDown(ImGuiKey_LeftShift)) speed *= 10;
		if (ImGui::IsKeyDown(ImGuiKey_LeftAlt)) speed /= 10;

		camera_position += m3::rotation_r_zxy(-camera_angles) * (frame_time * speed * v3f {
			(f32)(ImGui::IsKeyDown(ImGuiKey_D) - ImGui::IsKeyDown(ImGuiKey_A)),
			(f32)(ImGui::IsKeyDown(ImGuiKey_E) - ImGui::IsKeyDown(ImGuiKey_Q)),
			(f32)(ImGui::IsKeyDown(ImGuiKey_S) - ImGui::IsKeyDown(ImGuiKey_W)),
		});
	
		// 
		// Main render
		//

		//glClearColor(.3, .6, .9, 1);
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glViewport(0, 0, screen_size.x, screen_size.y);

		glEnable(GL_DEPTH_TEST);
		glDepthFunc(GL_LESS);

		glEnable(GL_CULL_FACE);
		glCullFace(GL_BACK);

		glDisable(GL_BLEND);
		
		m4 model_to_world = m4::translation({0,0,0}) * m4::rotation_r_zxy({0,0,0});
		m4 model_to_ndc = world_to_ndc * model_to_world;

		glBindVertexArray(mesh.va);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, mesh.vb);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, mesh.ib);

		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		glUseProgram(surface_program.program);
		gl::set_uniform(surface_program.program, "model_to_world", model_to_world);
		gl::set_uniform(surface_program.program, "model_to_ndc", model_to_ndc);
		gl::set_uniform(surface_program.program, "camera_position", camera_position);
		bind_texture(surface_program.program, 0, "sdf_texture", sdf_texture, 0, GL_TEXTURE_3D);
		gl::set_uniform(surface_program.program, "accurate_normals", accurate_normals);
		glDrawElements(GL_TRIANGLES, mesh.index_count, GL_UNSIGNED_INT, 0);

		if (enable_wireframe) {
			//glEnable(GL_LINE_SMOOTH);
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			glEnable(GL_POLYGON_OFFSET_LINE);
			glPolygonOffset(0, -16);
			glDepthFunc(GL_LEQUAL);
			glEnable(GL_BLEND);
			glDisable(GL_CULL_FACE);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
			glUseProgram(wireframe_program.program);
			gl::set_uniform(wireframe_program.program, "model_to_world", model_to_world);
			gl::set_uniform(wireframe_program.program, "model_to_ndc", model_to_ndc);
			glDrawElements(GL_TRIANGLES, mesh.index_count, GL_UNSIGNED_INT, 0);
		}

		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		gl::present();

		frame_time = reset(frame_timer);
		current_time += frame_time;

		//smoothed_fps = lerp(smoothed_fps, 1.0 / frame_time, smoothed_fps_lerp_t);
		smoothed_fps = lerp(smoothed_fps, 1.0 / frame_time, min(frame_time * 5, 1.0f));

		current_temporary_allocator.clear();
	}

	return 0;
}
