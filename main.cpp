// couldn't make gather faster than shuffles

#define TL_IMPL
#include <tl/main.h>
#include <tl/math.h>
#include <tl/math_random.h>
#include <tl/file.h>
#include <tl/precise_time.h>
#include <tl/opengl.h>
#include <tl/win32.h>
#include <tl/includer.h>
#include <tl/linear_set.h>

#pragma push_macro("assert")
#include <imgui.h>
#include <backends/imgui_impl_win32.h>
#include <backends/imgui_impl_opengl3.h>
#pragma pop_macro("assert")

using namespace tl;
using String = Span<utf8>;

gl::Functions gl_functions;
gl::Functions *tl_opengl_functions() { return &gl_functions; };

#define ASSERTION_FAILURE(...) 0
#define assert(...)

template <class T>
using GList = List<T, DefaultAllocator>;

#define timed_block(name) \
	auto CONCAT(_timer, __LINE__) = create_precise_timer(); \
	defer { println("{}: {}ms", name, 1000 * elapsed_time(CONCAT(_timer, __LINE__)));  }

#define timed_block(...)

struct Vertex {
	v3f position;
	v3f normal;
};

#define EXPANDABLE 0

void sdf_to_triangles_starting(v3u size, Span<f32> sdf, GList<Vertex> &vertices, GList<u32> &indices, GList<u32> &index_grid) {
	assert(size.x * size.y * size.z == sdf.count);
	defer {
		for (auto index : indices) {
			assert(index < vertices.count);
		}
	};
	
	#if EXPANDABLE
	// Reserve two layers. Expand in loop if needed
	vertices.reserve(2 * size.y * size.z);
	indices .reserve(2 * size.y * size.z * 18);
	#else
	vertices.reserve(size.x * size.y * size.z);
	indices .reserve(size.x * size.y * size.z * 18);
	#endif

	index_grid.reserve(size.x * size.y * size.z);

	vertices.count = 0;
	indices.count = 0;

	struct Edge {
		u8 a, b;
	};

	u8 edges[][2] {
		{0b000, 0b001},
		{0b010, 0b011},
		{0b100, 0b101},
		{0b110, 0b111},
		{0b000, 0b010},
		{0b001, 0b011},
		{0b100, 0b110},
		{0b101, 0b111},
		{0b000, 0b100},
		{0b001, 0b101},
		{0b010, 0b110},
		{0b011, 0b111},
	};
	
	f32 const e = 0;
	v3f corners[8] {
		{e,   e,   e  },
		{e,   e,   1-e},
		{e,   1-e, e  },
		{e,   1-e, 1-e},
		{1-e, e,   e  },
		{1-e, e,   1-e},
		{1-e, 1-e, e  },
		{1-e, 1-e, 1-e},
	};

	u32 current_index = 0;

	auto at = [&](auto &arr, u32 x, u32 y, u32 z) -> decltype(auto) {
		return arr[x * (size.y*size.z) + y * (size.z) + z];
	};

	{
		for (u32 lx = 0; lx < size.x-1; ++lx) {
			#if EXPANDABLE
			// Expand if don't have full layer ahead.
			vertices.reserve_exponential(vertices.count + size.y * size.z);
			#endif
		for (u32 ly = 0; ly < size.y-1; ++ly) {
		for (u32 lz = 0; lz < size.z-1; ++lz) {
			f32 d[8];

			d[0b000] = at(sdf.data, lx+0, ly+0, lz+0);
			d[0b001] = at(sdf.data, lx+0, ly+0, lz+1);
			d[0b010] = at(sdf.data, lx+0, ly+1, lz+0);
			d[0b011] = at(sdf.data, lx+0, ly+1, lz+1);
			d[0b100] = at(sdf.data, lx+1, ly+0, lz+0);
			d[0b101] = at(sdf.data, lx+1, ly+0, lz+1);
			d[0b110] = at(sdf.data, lx+1, ly+1, lz+0);
			d[0b111] = at(sdf.data, lx+1, ly+1, lz+1);

			u8 e =
				(u8)(d[0b000] < 0) +
				(u8)(d[0b001] < 0) +
				(u8)(d[0b010] < 0) +
				(u8)(d[0b011] < 0) +
				(u8)(d[0b100] < 0) +
				(u8)(d[0b101] < 0) +
				(u8)(d[0b110] < 0) +
				(u8)(d[0b111] < 0);

			if (e != 0 && e != 8) {
				v3f point = {};
				f32 divisor = 0;

				for (auto &edge : edges) {
					auto a = edge[0];
					auto b = edge[1];
					if ((d[a] < 0) != (d[b] < 0)) {
						point += lerp(corners[a], corners[b], V3f((f32)d[a] / (d[a] - d[b])));
						divisor += 1;
					}
				}
				point /= divisor;

				Vertex vertex;
				vertex.position = point + V3f(lx,ly,lz);

				vertex.normal = normalize(v3f{
					d[0b000] - d[0b100] +
					d[0b001] - d[0b101] +
					d[0b010] - d[0b110] +
					d[0b011] - d[0b111],
					d[0b000] - d[0b010] +
					d[0b001] - d[0b011] +
					d[0b100] - d[0b110] +
					d[0b101] - d[0b111],
					d[0b000] - d[0b001] +
					d[0b010] - d[0b011] +
					d[0b100] - d[0b101] +
					d[0b110] - d[0b111],
				});

				at(index_grid.data, lx, ly, lz) = current_index++;

				vertices.data[vertices.count++] = vertex;
			}
		}
		}
		}
	}

	{
		for (s32 lx = 1; lx < size.x-1; ++lx) {
			#if EXPANDABLE
			// Expand if don't have full layer ahead.
			indices.reserve_exponential(indices.count + size.y * size.z  * 18);
			#endif
		for (s32 ly = 1; ly < size.y-1; ++ly) {
		for (s32 lz = 1; lz < size.z-1; ++lz) {
			auto quad = [&](v3s _0, v3s _1, v3s _2, v3s _3) {
				auto i0 = at(index_grid.data, _0.x, _0.y, _0.z);
				auto i1 = at(index_grid.data, _1.x, _1.y, _1.z);
				auto i2 = at(index_grid.data, _2.x, _2.y, _2.z);
				auto i3 = at(index_grid.data, _3.x, _3.y, _3.z);

				indices.data[indices.count++] = i0;
				indices.data[indices.count++] = i1;
				indices.data[indices.count++] = i2;
				indices.data[indices.count++] = i1;
				indices.data[indices.count++] = i3;
				indices.data[indices.count++] = i2;
			};

			auto s0 = at(sdf.data, lx+0,ly+0,lz+0);
			auto s1 = at(sdf.data, lx+1,ly+0,lz+0);
			auto s2 = at(sdf.data, lx+0,ly+1,lz+0);
			auto s3 = at(sdf.data, lx+0,ly+0,lz+1);
			if ((s0 < 0) != (s1 < 0)) {
				v3s _0 = {lx, ly-1, lz-1};
				v3s _1 = {lx, ly+0, lz-1};
				v3s _2 = {lx, ly-1, lz+0};
				v3s _3 = {lx, ly+0, lz+0};

				if (s0 < 0) {
					std::swap(_0, _3);
				}

				quad(_0, _1, _2, _3);
			}
			if ((s0 < 0) != (s2 < 0)) {
				v3s _0 = {lx+0, ly, lz+0};
				v3s _1 = {lx+0, ly, lz-1};
				v3s _2 = {lx-1, ly, lz+0};
				v3s _3 = {lx-1, ly, lz-1};

				if (s0 < 0) {
					std::swap(_0, _3);
				}

				quad(_0, _1, _2, _3);
			}
			if ((s0 < 0) != (s3 < 0)) {
				v3s _0 = {lx-1, ly-1, lz};
				v3s _1 = {lx+0, ly-1, lz};
				v3s _2 = {lx-1, ly+0, lz};
				v3s _3 = {lx+0, ly+0, lz};

				if (s0 < 0) {
					std::swap(_0, _3);
				}

				quad(_0, _1, _2, _3);
			}
		}
		}
		}
	}
}

//
// 3 2 1 0
//
// d c b a  m0
// h g f e  m1
// l k j i  m2
// ? ? ? ?  m3
//
//  =>
//
// 3 2 1 0
//
// ? i e a  m0
// ? j f b  m1
// ? k g c  m2
// ? l h d  m3
//
void transpose(__m128 &m3, __m128 &m2, __m128 &m1, __m128 &m0) {
	__m128 t0 = _mm_unpacklo_ps(m0, m1);
	__m128 t1 = _mm_unpackhi_ps(m0, m1);
	__m128 t2 = _mm_unpacklo_ps(m2, m3);
	__m128 t3 = _mm_unpackhi_ps(m2, m3);
	m0 = _mm_movelh_ps(t0, t2);
	m1 = _mm_movehl_ps(t2, t0);
	m2 = _mm_movelh_ps(t1, t3);
	m3 = _mm_movehl_ps(t3, t1);
}

#define pshuf_ps(a, b) _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(a), b))

void sdf_to_triangles_sse(v3u size, Span<f32> sdf, Span<u8> negative_bitfield, Span<u8> surface_bitfield, GList<Vertex> &vertices, GList<u32> &indices, GList<u32> &index_grid) {
	auto at = [&](auto &arr, u32 x, u32 y, u32 z) -> decltype(auto) {
		return arr[x * (size.y*size.z) + y * (size.z) + z];
	};
	auto bit_at = [&](u32 x, u32 y, u32 z) -> bool {
		umm bit_index = x * (size.y*size.z) + y * (size.z) + z;
		return (negative_bitfield.data[bit_index / 8] >> (bit_index % 8)) & 1;
	};

	// from 1.35 to 1.81 gb/s ??? when assert expands to nothing?
	#if 0
	assert(size.x * size.y * size.z <= sdf.count);
	assert(size.x * size.y * size.z / 8 <= negative_bitfield.count);
	assert(size.x * size.y * size.z / 8 <= surface_bitfield.count);
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
	defer {
		for (auto index : indices) {
			assert(index < vertices.count);
		}
	};
	#endif
	
	#if EXPANDABLE
	// Reserve two layers. Expand in loop if needed
	vertices.reserve(2 * size.y * size.z);
	indices .reserve(2 * size.y * size.z * 18);
	#else
	vertices.reserve(size.x * size.y * size.z);
	indices .reserve(size.x * size.y * size.z * 18);
	#endif
	
	// Reduced size of index grid from N*N*N to 2*N*N
	// 3.0 gb/s
	index_grid.reserve(2 * size.y * size.z);

	vertices.count = 0;
	indices.count = 0;
	
	umm sizeyz = size.y*size.z;
	umm indices_count = 0;

	for (u32 lx = 0; lx < size.x-1; ++lx) {
		#if EXPANDABLE
		// Expand if don't have full layer ahead.
		vertices.reserve_exponential(vertices.count + size.y * size.z);
		#endif
	for (u32 ly = 0; ly < size.y-1; ++ly) {
	for (u32 lz = 0; lz < size.z-1; ++lz) {
		
		// Keep both skips by surface and by negative bitfields, they both skip a lot


		// 2.1 gb/s
		u64 s = *(s64 *)&surface_bitfield[(lx*size.y*size.z + ly*size.z + lz) / 8] >> (lz%8);
		if (s + 1 < 2) {
			lz += 55;
			continue;
		}

		u64 r = 
			(((*(u64*)&negative_bitfield.data[((lx+0)*size.y*size.z + (ly+0)*size.z + lz)/8] >> (lz%8)) & 0xffff) << 0) |
			(((*(u64*)&negative_bitfield.data[((lx+0)*size.y*size.z + (ly+1)*size.z + lz)/8] >> (lz%8)) & 0xffff) << 16) |
			(((*(u64*)&negative_bitfield.data[((lx+1)*size.y*size.z + (ly+0)*size.z + lz)/8] >> (lz%8)) & 0xffff) << 32) |
			(((*(u64*)&negative_bitfield.data[((lx+1)*size.y*size.z + (ly+1)*size.z + lz)/8] >> (lz%8)) & 0xffff) << 48);
			
		// . . . . . . . . 7 6 5 4 3 2[1 0]
		// . . . . . . . . f e d c b a[9 8]
		// . . . . . . . . n m l k j i[h g]
		// . . . . . . . . v u t s r q[p o]

		// . . . . . . . . 7 6 5 4 3[2 1]0
		// . . . . . . . . f e d c b[a 9]8
		// . . . . . . . . n m l k j[i h]g
		// . . . . . . . . v u t s r[q p]o

		// ...

		if (r + 1 < 2) {
			// all are zeros or all are ones
			lz += 14;
			continue;
		}

		if ((u32)(count_bits(r & 0x0003000300030003) - 1) < 7) {
			__m128 r11 = _mm_castpd_ps(_mm_load_sd((double *)&at(sdf.data, lx+0, ly+0, lz+0)));
			__m128 r12 = _mm_castpd_ps(_mm_load_sd((double *)&at(sdf.data, lx+0, ly+1, lz+0)));
			__m128 r21 = _mm_castpd_ps(_mm_load_sd((double *)&at(sdf.data, lx+1, ly+0, lz+0)));
			__m128 r22 = _mm_castpd_ps(_mm_load_sd((double *)&at(sdf.data, lx+1, ly+1, lz+0)));
			// ? ? 1 0
			// ? ? 3 2
			// ? ? 5 4
			// ? ? 7 6

			__m128 d3210 = _mm_shuffle_ps(r11, r12, _MM_SHUFFLE(1,0,1,0));
			__m128 d7654 = _mm_shuffle_ps(r21, r22, _MM_SHUFFLE(1,0,1,0));

			at(index_grid.data, lx & 1, ly, lz) = vertices.count;
			Vertex *vertex = &vertices.data[vertices.count++];

			#if 1

			////////////////
			// HORIZONTAL //
			////////////////

			__m128 d6420 = _mm_shuffle_ps(d3210, d7654, _MM_SHUFFLE(2,0,2,0));
			__m128 d5410 = _mm_shuffle_ps(d3210, d7654, _MM_SHUFFLE(1,0,1,0));
			__m128 d7531 = _mm_shuffle_ps(d3210, d7654, _MM_SHUFFLE(3,1,3,1));
			__m128 d7632 = _mm_shuffle_ps(d3210, d7654, _MM_SHUFFLE(3,2,3,2));
				
			__m128 c0 = _mm_xor_ps(d6420, d7531);
			__m128 c1 = _mm_xor_ps(d5410, d7632);
			__m128 c2 = _mm_xor_ps(d3210, d7654);
			c0 = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(c0), 31));
			c1 = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(c1), 31));
			c2 = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(c2), 31));

			int c = _mm_movemask_ps(c0) | (_mm_movemask_ps(c1) << 4) | (_mm_movemask_ps(c2) << 8);

			__m128 t0 = _mm_div_ps(d6420, _mm_sub_ps(d6420, d7531));
			__m128 t1 = _mm_div_ps(d5410, _mm_sub_ps(d5410, d7632));
			__m128 t2 = _mm_div_ps(d3210, _mm_sub_ps(d3210, d7654));
				
			__m128 f1010 = _mm_set_ps(1,0,1,0);
			__m128 f1100 = _mm_set_ps(1,1,0,0);
			__m128 f1111 = _mm_set1_ps(1);

			__m128 position = _mm_or_ps(_mm_or_ps(
				_mm_dp_ps(_mm_add_ps(_mm_add_ps(
					_mm_and_ps(c0, f1100),
					_mm_and_ps(c1, f1100)),
					_mm_and_ps(c2, t2)), 
					f1111, 0b1111'0001),
				_mm_dp_ps(_mm_add_ps(_mm_add_ps(
					_mm_and_ps(c0, f1010),
					_mm_and_ps(c1, t1)),
					_mm_and_ps(c2, f1100)),
					f1111, 0b1111'0010)),
				_mm_dp_ps(_mm_add_ps(_mm_add_ps(
					_mm_and_ps(c0, t0),
					_mm_and_ps(c1, f1010)),
					_mm_and_ps(c2, f1010)),
					f1111, 0b1111'0100)
			);
				
			position = _mm_div_ps(position, _mm_set1_ps(count_bits(c)));
			position = _mm_add_ps(position, _mm_set_ps(0, lz, ly, lx));
			_mm_storeu_ps((float *)&vertex->position, position);

			__m128 normal = _mm_or_ps(_mm_or_ps(
				_mm_dp_ps(_mm_sub_ps(d3210, d7654), f1111, 0b1111'0001),
				_mm_dp_ps(_mm_sub_ps(d5410, d7632), f1111, 0b1111'0010)),
				_mm_dp_ps(_mm_sub_ps(d6420, d7531), f1111, 0b1111'0100));
				
			normal = _mm_mul_ps(normal, _mm_permute_ps(_mm_rsqrt_ss(_mm_dp_ps(normal, normal, 0b0111'0001)), _MM_SHUFFLE(0,0,0,0)));
			_mm_storeu_ps((float *)&vertex->normal, normal);
			#else

			//////////////
			// VERTICAL //
			//////////////

			__m128 d6420 = _mm_shuffle_ps(d3210, d7654, _MM_SHUFFLE(2,0,2,0));
			__m128 d5410 = _mm_shuffle_ps(d3210, d7654, _MM_SHUFFLE(1,0,1,0));
			__m128 d7531 = _mm_shuffle_ps(d3210, d7654, _MM_SHUFFLE(3,1,3,1));
			__m128 d7632 = _mm_shuffle_ps(d3210, d7654, _MM_SHUFFLE(3,2,3,2));
				
			__m128 c0 = _mm_xor_ps(d3210, d7654);
			__m128 c1 = _mm_xor_ps(d5410, d7632);
			__m128 c2 = _mm_xor_ps(d6420, d7531);
			c0 = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(c0), 31));
			c1 = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(c1), 31));
			c2 = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(c2), 31));

			int cmask = _mm_movemask_ps(c0) | (_mm_movemask_ps(c1) << 4) | (_mm_movemask_ps(c2) << 8);

			__m128 t0 = _mm_div_ps(d3210, _mm_sub_ps(d3210, d7654));
			__m128 t1 = _mm_div_ps(d5410, _mm_sub_ps(d5410, d7632));
			__m128 t2 = _mm_div_ps(d6420, _mm_sub_ps(d6420, d7531));
				
			__m128 f1111 = _mm_set1_ps(1);

			__m128 c3;
			__m128 t3;
			transpose(t3, t2, t1, t0);
			transpose(c3, c2, c1, c0);
				
			constexpr int w = 0; // Doesn't matter

			__m128 c215 = _mm_shuffle_ps(c2, c1, _MM_SHUFFLE(1, 2, 2, w)); c215 = _mm_shuffle_ps(c215, c215, _MM_SHUFFLE(0,3,2,1));
			__m128 c6a9 = _mm_shuffle_ps(c2, c1, _MM_SHUFFLE(w, 0, 0, 1));
			__m128 c337 = _mm_shuffle_ps(c3, c3, _MM_SHUFFLE(w, 1, 2, 2));
			__m128 c7b3 = _mm_shuffle_ps(c3, c3, _MM_SHUFFLE(w, 0, 0, 1));
				
			__m128 position = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_and_ps(c0, t0),
			                                                   _mm_and_ps(c1, t1)),
			                                        _mm_add_ps(_mm_and_ps(c2, t2),
			                                                   _mm_and_ps(c3, t3))),
			                             _mm_add_ps(_mm_add_ps(_mm_and_ps(c215, f1111),
			                                                   _mm_and_ps(c6a9, f1111)),
			                                        _mm_add_ps(_mm_and_ps(c337, f1111),
			                                                   _mm_and_ps(c7b3, f1111))));

			position = _mm_div_ps(position, _mm_set1_ps(count_bits(cmask)));
			position = _mm_add_ps(position, _mm_set_ps(0, lz, ly, lx));
			_mm_storeu_ps((float *)&vertex->position, position);

			__m128 d000x = _mm_shuffle_ps(d3210, d3210, _MM_SHUFFLE(0,0,0,0));
			__m128 d124x = _mm_shuffle_ps(d7654, d3210, _MM_SHUFFLE(1,2,0,0));
			__m128 d211x = _mm_shuffle_ps(d3210, d3210, _MM_SHUFFLE(2,1,1,0));
			__m128 d225x = _mm_shuffle_ps(d7654, d3210, _MM_SHUFFLE(2,2,1,0));
			__m128 d442x = _mm_shuffle_ps(d3210, d7654, _MM_SHUFFLE(0,0,2,0));
			__m128 d566x = _mm_shuffle_ps(d7654, d7654, _MM_SHUFFLE(1,2,2,0));
			__m128 d653x = _mm_shuffle_ps(d3210, d7654, _MM_SHUFFLE(2,1,3,0));
			__m128 d777x = _mm_shuffle_ps(d7654, d7654, _MM_SHUFFLE(3,3,3,0));

			// TODO: normal looks different from reference
			__m128 normal = _mm_add_ps(_mm_add_ps(_mm_sub_ps(d000x, d124x),
			                                      _mm_sub_ps(d211x, d225x)),
			                           _mm_add_ps(_mm_sub_ps(d442x, d566x), 
			                                      _mm_sub_ps(d653x, d777x)));
			normal = _mm_shuffle_ps(normal, normal, _MM_SHUFFLE(0,3,2,1));
				

			__m128 invlen = _mm_rsqrt_ss(_mm_dp_ps(normal, normal, 0b0111'0001));
			normal = _mm_mul_ps(normal, _mm_shuffle_ps(invlen, invlen, _MM_SHUFFLE(0,0,0,0)));
			_mm_storeu_ps((float *)&vertex->normal, normal);
			#endif

			// Merged vertex and triangle creation loops
			// 2.5 gb/s
			if ((lx > 0) & (ly > 0) & (lz > 0)) {
				bool s0 = (r >> 0) & 1;
				bool s1 = (r >> 32) & 1;
				bool s3 = (r >> 1) & 1;
				bool s2 = (r >> 16) & 1;

				//     v3s -> umm
				// 3.0gb/s -> 3.1gb/s
				auto quad = [&](umm _0, umm _1, umm _2, umm _3) {
					auto i0 = index_grid.data[_0];
					auto i1 = index_grid.data[_1];
					auto i2 = index_grid.data[_2];
					auto i3 = index_grid.data[_3];

					indices.data[indices_count++] = i0;
					indices.data[indices_count++] = i1;
					indices.data[indices_count++] = i2;
					indices.data[indices_count++] = i1;
					indices.data[indices_count++] = i3;
					indices.data[indices_count++] = i2;
				};
				
				#if 1
				// Branching, bit faster
				#define CSWAP if (s0) std::swap(_0, _3)
				#else
				// Branchless, bit slower
				#define CSWAP                     \
					umm d = (_0 ^ _3) & (smm)-s0; \
					_0 ^= d;                      \
					_3 ^= d
				#endif


				if (s0 != s1) {
					umm _0 = ((lx-0) & 1)*sizeyz + (ly-1)*size.z + (lz-1);
					umm _1 = ((lx-0) & 1)*sizeyz + (ly-0)*size.z + (lz-1);
					umm _2 = ((lx-0) & 1)*sizeyz + (ly-1)*size.z + (lz-0);
					umm _3 = ((lx-0) & 1)*sizeyz + (ly-0)*size.z + (lz-0);

					CSWAP;

					quad(_0, _1, _2, _3);
				}
				if (s0 != s2) {
					umm _0 = ((lx-0) & 1)*sizeyz + (ly-0)*size.z + (lz-0);
					umm _1 = ((lx-0) & 1)*sizeyz + (ly-0)*size.z + (lz-1);
					umm _2 = ((lx-1) & 1)*sizeyz + (ly-0)*size.z + (lz-0);
					umm _3 = ((lx-1) & 1)*sizeyz + (ly-0)*size.z + (lz-1);
					
					CSWAP;

					quad(_0, _1, _2, _3);
				}
				if (s0 != s3) {
					umm _0 = ((lx-1) & 1)*sizeyz + (ly-1)*size.z + (lz-0);
					umm _1 = ((lx-0) & 1)*sizeyz + (ly-1)*size.z + (lz-0);
					umm _2 = ((lx-1) & 1)*sizeyz + (ly-0)*size.z + (lz-0);
					umm _3 = ((lx-0) & 1)*sizeyz + (ly-0)*size.z + (lz-0);

					CSWAP;

					quad(_0, _1, _2, _3);
				}
				#undef CSWAP
			}
		}
	}
	}
	}

	indices.count = indices_count;
}

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

struct {
	f32 sdf[N][N][N];
	
	// bit for each value in sdf: 1 if negative, 0 if positive
	u8 negative_bitfield[N*N*N/8];

	// bit for each 2x2x2 cube in sdf: 1 if there is a surface, 0 if not.
	// NOTE: the dimension of this should be N-1, but i want to keep indices the same as in other arrays,
	//       so there's a bit of wasted bits: ~2% when N is 128. (127*127 + 127)*3 + 1.
	u8 surface_bitfield[N*N*N/8];
} sdf;

void update_sdf_auxiliary_fields() {
	memset(sdf.negative_bitfield, 0, sizeof(sdf.negative_bitfield));
	memset(sdf.surface_bitfield, 0, sizeof(sdf.surface_bitfield));
	for (int x = 0; x < N; ++x) {
	for (int y = 0; y < N; ++y) {
	for (int z = 0; z < N; ++z) {
		sdf.negative_bitfield[(x*N*N + y*N + z)/8] |= (*(u32 *)&sdf.sdf[x][y][z] >> 31) << (z % 8);
	}
	}
	}
	for (int x = 0; x < N-1; ++x) {
	for (int y = 0; y < N-1; ++y) {
	for (int z = 0; z < N-1; ++z) {

		u8 r = 
			(((*(u16*)&sdf.negative_bitfield[((x+0)*N*N + (y+0)*N + z)/8] >> (z%8)) & 0x3) << 0) |
			(((*(u16*)&sdf.negative_bitfield[((x+0)*N*N + (y+1)*N + z)/8] >> (z%8)) & 0x3) << 2) |
			(((*(u16*)&sdf.negative_bitfield[((x+1)*N*N + (y+0)*N + z)/8] >> (z%8)) & 0x3) << 4) |
			(((*(u16*)&sdf.negative_bitfield[((x+1)*N*N + (y+1)*N + z)/8] >> (z%8)) & 0x3) << 6);
		
		bool surface = (u8)(r + 1) >= 2;
		
		sdf.surface_bitfield[(x*N*N + y*N + z)/8] |= surface << (z % 8);
	}
	}
	}
}

void write_sdf_to_file() {
	for (int x = 0; x < N; ++x) {
	for (int y = 0; y < N; ++y) {
	for (int z = 0; z < N; ++z) {
		//sdf[x][y][z] = gradient_noise_s16({x,y,z}, 8) - 0.5 - (sinf(y * tau / 32.0f)) * 0.25f;
		//sdf[x][y][z] = value_noise_smooth<f32>(V3f(x,y,z) / 8) - 0.5 - (y - N/2) / 32.0f;

		v3f p = V3f(x,y,z) - N/2;

		sdf.sdf[x][y][z] = clamp((gradient_noise({x,y,z}, 8) - 0.5f) * 32 - (length(p) - N / 3), -1.0f, 1.0f);
		//sdf.sdf[x][y][z] = roundf(map(sdf.sdf[x][y][z], -1.f, 1.f, -1.f, 254.f));
	}
	}
	}
	update_sdf_auxiliary_fields();

	write_entire_file(u8"sdf.bin"s, value_as_bytes(sdf));
}

void load_sdf_from_file() {
	auto buffer = read_entire_file(u8"sdf.bin"s);
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

struct Program {
	struct SourceFile : includer::SourceFileBase {
		FileTime last_write_time = 0;
			
		void init() {
			last_write_time = get_file_write_time(path).value_or(0);;
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

	void reload() {
		println("Recompiling {}", path);

		includer.load(path, &text, includer::LoadOptions{
			.append_location_info = +[](StringBuilder &builder, String path, u32 line) {
				append_format(builder, "\n#line {} \"{}\"\n", line, path);
			}
		});

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
	
	sdf_to_triangles_sse({N,N,N}, flatten(sdf.sdf), sdf.negative_bitfield, sdf.surface_bitfield, vertices, indices, index_grid);
	GpuMesh mesh = create_gpu_mesh(vertices, indices);

	Program surface_program = {.path = to_list(resource_path(u8"shaders/surface.glsl"s))};
	Program wireframe_program = {.path = to_list(resource_path(u8"shaders/wireframe.glsl"s))};
	
	Program *all_programs[] = { &surface_program, &wireframe_program };

	for (auto program : all_programs) {
		program->reload();
	}
	
	f32 frame_time = 1.0f / 60;
	f32 time = 0;
	PreciseTimer frame_timer = create_precise_timer();
	v2s old_screen_size = {-1,-1};
	
	v3f camera_position = V3f(N*1.1f, N*1.1f, N*1.1f);
	v3f camera_angles = V3f(pi/4, -pi/4, 0);

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

		if (ImGui::Begin("Window")) {
			ImGui::Checkbox("Wireframe", &enable_wireframe);
			if (ImGui::Button("Remesh Starting")) {
				update_sdf_auxiliary_fields();
				sdf_to_triangles_starting({N,N,N}, flatten(sdf.sdf), vertices, indices, index_grid);
				update_gpu_mesh(mesh, vertices, indices);
			}
			if (ImGui::Button("Remesh SSE")) {
				update_sdf_auxiliary_fields();
				sdf_to_triangles_sse({N,N,N}, flatten(sdf.sdf), sdf.negative_bitfield, sdf.surface_bitfield, vertices, indices, index_grid);
				update_gpu_mesh(mesh, vertices, indices);
			}

			bench.operator()<0>("Starting bench", [&]{sdf_to_triangles_starting({N,N,N}, flatten(sdf.sdf), vertices, indices, index_grid);});
			bench.operator()<1>("SSE bench"   , [&]{sdf_to_triangles_sse({N,N,N}, flatten(sdf.sdf), sdf.negative_bitfield, sdf.surface_bitfield, vertices, indices, index_grid);});
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
		
		m4 world_to_ndc = m4::perspective_right_handed((f32)screen_size.x / screen_size.y, pi/2, 0.01f, 1000.0f) * m4::rotation_r_yxz(camera_angles) * m4::translation(-camera_position);
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

			println("dir {}", cursor_ray.direction);

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

							for (s32 x = max(0,hit_position_int.x-R); x <= min(hit_position_int.x+R,N-1); ++x) {
							for (s32 y = max(0,hit_position_int.y-R); y <= min(hit_position_int.y+R,N-1); ++y) {
							for (s32 z = max(0,hit_position_int.z-R); z <= min(hit_position_int.z+R,N-1); ++z) {
								v3s p = {x,y,z};
								sdf.sdf[p.x][p.y][p.z] -= frame_time * force * map_clamped<f32,f32>(distance((v3f)p, hit.position), 0, brush_radius, 1, 0);
							}
							}
							}

							update_sdf_auxiliary_fields();
							sdf_to_triangles_sse({N,N,N}, flatten(sdf.sdf), sdf.negative_bitfield, sdf.surface_bitfield, vertices, indices, index_grid);
							update_gpu_mesh(mesh, vertices, indices);
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

							if (force > 0) {
								for (s32 x = max(0,hit_position_int.x-R); x <= min(hit_position_int.x+R,N-1); ++x) {
								for (s32 y = max(0,hit_position_int.y-R); y <= min(hit_position_int.y+R,N-1); ++y) {
								for (s32 z = max(0,hit_position_int.z-R); z <= min(hit_position_int.z+R,N-1); ++z) {
									v3s p = {x,y,z};
									sdf.sdf[p.x][p.y][p.z] = min(sdf.sdf[p.x][p.y][p.z], map_clamped<f32,f32>(distance((v3f)p, hit.position), brush_radius - 1, brush_radius, -1, 1));
								}
								}
								}
							} else {
								for (s32 x = max(0,hit_position_int.x-R); x <= min(hit_position_int.x+R,N-1); ++x) {
								for (s32 y = max(0,hit_position_int.y-R); y <= min(hit_position_int.y+R,N-1); ++y) {
								for (s32 z = max(0,hit_position_int.z-R); z <= min(hit_position_int.z+R,N-1); ++z) {
									v3s p = {x,y,z};
									sdf.sdf[p.x][p.y][p.z] = max(sdf.sdf[p.x][p.y][p.z], map_clamped<f32,f32>(distance((v3f)p, hit.position), brush_radius - 1, brush_radius, 1, -1));
								}
								}
								}
							}

							update_sdf_auxiliary_fields();
							sdf_to_triangles_sse({N,N,N}, flatten(sdf.sdf), sdf.negative_bitfield, sdf.surface_bitfield, vertices, indices, index_grid);
							update_gpu_mesh(mesh, vertices, indices);
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
		time += frame_time;

		current_temporary_allocator.clear();
	}

	return 0;
}
