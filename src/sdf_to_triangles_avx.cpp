#include "common.h"

// 
// x7 x6 x5 x4 x3 x2 x1 x0
// y7 y6 y5 y4 y3 y2 y1 y0
// z7 z6 z5 z4 z3 z2 z1 z0
// w7 w6 w5 w4 w3 w2 w1 w0
//
//  =>
// w4 z4 y4 x4 w0 z0 y0 x0
// w5 z5 y5 x5 w1 z1 y1 x1
// w6 z6 y6 x6 w2 z2 y2 x2
// w7 z7 y7 x7 w3 z3 y3 x3
//
forceinline static void unpack(__m256 &m3, __m256 &m2, __m256 &m1, __m256 &m0) {
	#if 1
	// Modifed from https://gist.github.com/nihui/37d98b705a6a28911d77c502282b4748
    __m256 t0 = _mm256_unpacklo_ps(m0, m1); // x0 y0 x1 y1 x4 y4 x5 y5
    __m256 t1 = _mm256_unpackhi_ps(m0, m1); // x2 y2 x3 y3 x6 y6 x7 y7
    __m256 t2 = _mm256_unpacklo_ps(m2, m3); // z0 w0 z1 w1 z4 w4 z5 w5
    __m256 t3 = _mm256_unpackhi_ps(m2, m3); // z2 w2 z3 w3 z6 w6 z7 w7

    m0 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0)); // x0 y0 z0 w0 x4 y4 z4 w4
    m1 = _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2)); // x1 y1 x1 w1 x5 y5 z5 w5
    m2 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0)); // x2 y2 z2 w2 x6 y6 z6 w6
    m3 = _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2)); // x3 y3 z3 w3 x7 y7 z7 w7
	#elif 1

	#else
	__m256 r0;
	__m256 r1;
	__m256 r2;
	__m256 r3;
	
	((float *)&r0)[0] = ((float *)&m0)[0];
	((float *)&r0)[1] = ((float *)&m1)[0];
	((float *)&r0)[2] = ((float *)&m2)[0];
	((float *)&r0)[3] = ((float *)&m3)[0];
	((float *)&r0)[4] = ((float *)&m0)[4];
	((float *)&r0)[5] = ((float *)&m1)[4];
	((float *)&r0)[6] = ((float *)&m2)[4];
	((float *)&r0)[7] = ((float *)&m3)[4];
	((float *)&r1)[0] = ((float *)&m0)[1];
	((float *)&r1)[1] = ((float *)&m1)[1];
	((float *)&r1)[2] = ((float *)&m2)[1];
	((float *)&r1)[3] = ((float *)&m3)[1];
	((float *)&r1)[4] = ((float *)&m0)[5];
	((float *)&r1)[5] = ((float *)&m1)[5];
	((float *)&r1)[6] = ((float *)&m2)[5];
	((float *)&r1)[7] = ((float *)&m3)[5];
	((float *)&r2)[0] = ((float *)&m0)[2];
	((float *)&r2)[1] = ((float *)&m1)[2];
	((float *)&r2)[2] = ((float *)&m2)[2];
	((float *)&r2)[3] = ((float *)&m3)[2];
	((float *)&r2)[4] = ((float *)&m0)[6];
	((float *)&r2)[5] = ((float *)&m1)[6];
	((float *)&r2)[6] = ((float *)&m2)[6];
	((float *)&r2)[7] = ((float *)&m3)[6];
	((float *)&r3)[0] = ((float *)&m0)[3];
	((float *)&r3)[1] = ((float *)&m1)[3];
	((float *)&r3)[2] = ((float *)&m2)[3];
	((float *)&r3)[3] = ((float *)&m3)[3];
	((float *)&r3)[4] = ((float *)&m0)[7];
	((float *)&r3)[5] = ((float *)&m1)[7];
	((float *)&r3)[6] = ((float *)&m2)[7];
	((float *)&r3)[7] = ((float *)&m3)[7];

	m0 = r0;
	m1 = r1;
	m2 = r2;
	m3 = r3;
	#endif
}

void sdf_to_triangles_avx(v3u size, Span<f32> sdf, Span<u8> negative_bitfield, Span<u8> surface_bitfield, GList<Vertex> &vertices, GList<u32> &indices, GList<u32> &index_grid) {
	auto at = [&](auto &arr, u32 x, u32 y, u32 z) -> decltype(auto) {
		return arr[x * (size.y*size.z) + y * (size.z) + z];
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
	
	#if !RESERVE_ALL
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

	umm prev_layer_vertices_begin = 0;
	umm prev_layer_vertices_end   = 0;

	#if SIMD_METHOD == DEFERRED
	struct Deferred {
		__m256 lx;
		__m256 ly;
		__m256 lz;
		__m256 d0;
		__m256 d1;
		__m256 d2;
		__m256 d3;
		__m256 d4;
		__m256 d5;
		__m256 d6;
		__m256 d7;
	} deferred;
	u8 deferred_index = 0;

	#pragma push_macro("forceinline")
	#undef forceinline
	auto write_deferred_vertices = [&] ()
		[[msvc::forceinline]]
	{
	#pragma pop_macro("forceinline")
		__m256 d0 = deferred.d0;
		__m256 d1 = deferred.d1;
		__m256 d2 = deferred.d2;
		__m256 d3 = deferred.d3;
		__m256 d4 = deferred.d4;
		__m256 d5 = deferred.d5;
		__m256 d6 = deferred.d6;
		__m256 d7 = deferred.d7;
		
		__m256 s0 = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(d0), 31));
		__m256 s1 = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(d1), 31));
		__m256 s2 = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(d2), 31));
		__m256 s3 = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(d3), 31));
		__m256 s4 = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(d4), 31));
		__m256 s5 = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(d5), 31));
		__m256 s6 = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(d6), 31));
		__m256 s7 = _mm256_castsi256_ps(_mm256_srai_epi32(_mm256_castps_si256(d7), 31));

		__m256 c0 = _mm256_xor_ps(s0, s1);
		__m256 c1 = _mm256_xor_ps(s2, s3);
		__m256 c2 = _mm256_xor_ps(s4, s5);
		__m256 c3 = _mm256_xor_ps(s6, s7);
		__m256 c4 = _mm256_xor_ps(s0, s2);
		__m256 c5 = _mm256_xor_ps(s1, s3);
		__m256 c6 = _mm256_xor_ps(s4, s6);
		__m256 c7 = _mm256_xor_ps(s5, s7);
		__m256 c8 = _mm256_xor_ps(s0, s4);
		__m256 c9 = _mm256_xor_ps(s1, s5);
		__m256 ca = _mm256_xor_ps(s2, s6);
		__m256 cb = _mm256_xor_ps(s3, s7);

		__m256 t0 = _mm256_div_ps(d0, _mm256_sub_ps(d0, d1));
		__m256 t1 = _mm256_div_ps(d2, _mm256_sub_ps(d2, d3));
		__m256 t2 = _mm256_div_ps(d4, _mm256_sub_ps(d4, d5));
		__m256 t3 = _mm256_div_ps(d6, _mm256_sub_ps(d6, d7));
		__m256 t4 = _mm256_div_ps(d0, _mm256_sub_ps(d0, d2));
		__m256 t5 = _mm256_div_ps(d1, _mm256_sub_ps(d1, d3));
		__m256 t6 = _mm256_div_ps(d4, _mm256_sub_ps(d4, d6));
		__m256 t7 = _mm256_div_ps(d5, _mm256_sub_ps(d5, d7));
		__m256 t8 = _mm256_div_ps(d0, _mm256_sub_ps(d0, d4));
		__m256 t9 = _mm256_div_ps(d1, _mm256_sub_ps(d1, d5));
		__m256 ta = _mm256_div_ps(d2, _mm256_sub_ps(d2, d6));
		__m256 tb = _mm256_div_ps(d3, _mm256_sub_ps(d3, d7));

		__m256 f1111 = _mm256_set1_ps(1);
		
		__m256 _0, _1, _2, _3;

		_0 = _mm256_and_ps(c8, t8);
		_1 = _mm256_and_ps(c9, t9);
		_2 = _mm256_add_ps(_mm256_add_ps(_mm256_and_ps(c2, f1111), _mm256_and_ps(c6, f1111)), _mm256_and_ps(ca, ta));
		_3 = _mm256_add_ps(_mm256_add_ps(_mm256_and_ps(c3, f1111), _mm256_and_ps(c7, f1111)), _mm256_and_ps(cb, tb));
		__m256 px = _mm256_add_ps(_mm256_add_ps(_0, _1), _mm256_add_ps(_2, _3));
		
		_0 = _mm256_and_ps(c4, t4);
		_1 = _mm256_add_ps(_mm256_and_ps(c1, f1111), _mm256_and_ps(c5, t5));
		_2 = _mm256_add_ps(_mm256_and_ps(c6, t6), _mm256_and_ps(ca, f1111));
		_3 = _mm256_add_ps(_mm256_add_ps(_mm256_and_ps(c3, f1111), _mm256_and_ps(c7, t7)), _mm256_and_ps(cb, f1111));
		__m256 py = _mm256_add_ps(_mm256_add_ps(_0, _1), _mm256_add_ps(_2, _3));
		
		_0 = _mm256_and_ps(c0, t0);
		_1 = _mm256_add_ps(_mm256_add_ps(_mm256_and_ps(c1, t1), _mm256_and_ps(c5, f1111)), _mm256_and_ps(c9, f1111));
		_2 = _mm256_and_ps(c2, t2);
		_3 = _mm256_add_ps(_mm256_add_ps(_mm256_and_ps(c3, t3), _mm256_and_ps(c7, f1111)), _mm256_and_ps(cb, f1111));
		__m256 pz = _mm256_add_ps(_mm256_add_ps(_0, _1), _mm256_add_ps(_2, _3));
		
		__m256 c = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_and_ps(f1111, c0),
		                                                                   _mm256_and_ps(f1111, c1)),
		                                                     _mm256_add_ps(_mm256_and_ps(f1111, c2),
		                                                                   _mm256_and_ps(f1111, c3))),
		                                       _mm256_add_ps(_mm256_add_ps(_mm256_and_ps(f1111, c4),
		                                                                   _mm256_and_ps(f1111, c5)),
		                                                     _mm256_add_ps(_mm256_and_ps(f1111, c6),
		                                                                   _mm256_and_ps(f1111, c7)))),
			                                   _mm256_add_ps(_mm256_add_ps(_mm256_and_ps(f1111, c8),
		                                                                   _mm256_and_ps(f1111, c9)),
		                                                     _mm256_add_ps(_mm256_and_ps(f1111, ca),
		                                                                   _mm256_and_ps(f1111, cb))));
		
		px = _mm256_add_ps(_mm256_div_ps(px, c), deferred.lx);
		py = _mm256_add_ps(_mm256_div_ps(py, c), deferred.ly);
		pz = _mm256_add_ps(_mm256_div_ps(pz, c), deferred.lz);

		#if 0
		_mm256_i32scatter_ps(&vertices.data[vertices.count].position.x, _mm256_setr_epi32(0,6,12,18,24,30,36,42), px, 4);
		_mm256_i32scatter_ps(&vertices.data[vertices.count].position.y, _mm256_setr_epi32(0,6,12,18,24,30,36,42), py, 4);
		_mm256_i32scatter_ps(&vertices.data[vertices.count].position.z, _mm256_setr_epi32(0,6,12,18,24,30,36,42), pz, 4);
		#elif 1
		__m256 pw;
		unpack(pw, pz, py, px);
		Vertex *vertex = vertices.data + vertices.count;
		_mm_storeu_ps(vertex++->position.s, _mm256_extractf128_ps(px, 0));
		_mm_storeu_ps(vertex++->position.s, _mm256_extractf128_ps(py, 0));
		_mm_storeu_ps(vertex++->position.s, _mm256_extractf128_ps(pz, 0));
		_mm_storeu_ps(vertex++->position.s, _mm256_extractf128_ps(pw, 0));
		_mm_storeu_ps(vertex++->position.s, _mm256_extractf128_ps(px, 1));
		_mm_storeu_ps(vertex++->position.s, _mm256_extractf128_ps(py, 1));
		_mm_storeu_ps(vertex++->position.s, _mm256_extractf128_ps(pz, 1));
		_mm_storeu_ps(vertex++->position.s, _mm256_extractf128_ps(pw, 1));
		#elif 1
		__m256 pw;
		unpack(pw, pz, py, px);
		
		memcpy(vertices.data[vertices.count+0].position.s, (float *)&px + 0, 12);
		memcpy(vertices.data[vertices.count+1].position.s, (float *)&py + 0, 12);
		memcpy(vertices.data[vertices.count+2].position.s, (float *)&pz + 0, 12);
		memcpy(vertices.data[vertices.count+3].position.s, (float *)&pw + 0, 12);
		memcpy(vertices.data[vertices.count+4].position.s, (float *)&px + 4, 12);
		memcpy(vertices.data[vertices.count+5].position.s, (float *)&py + 4, 12);
		memcpy(vertices.data[vertices.count+6].position.s, (float *)&pz + 4, 12);
		memcpy(vertices.data[vertices.count+7].position.s, (float *)&pw + 4, 12);
		#else
		vertices.data[vertices.count+0].position = {((float *)&px)[0], ((float *)&py)[0], ((float *)&pz)[0]};
		vertices.data[vertices.count+1].position = {((float *)&px)[1], ((float *)&py)[1], ((float *)&pz)[1]};
		vertices.data[vertices.count+2].position = {((float *)&px)[2], ((float *)&py)[2], ((float *)&pz)[2]};
		vertices.data[vertices.count+3].position = {((float *)&px)[3], ((float *)&py)[3], ((float *)&pz)[3]};
		vertices.data[vertices.count+4].position = {((float *)&px)[4], ((float *)&py)[4], ((float *)&pz)[4]};
		vertices.data[vertices.count+5].position = {((float *)&px)[5], ((float *)&py)[5], ((float *)&pz)[5]};
		vertices.data[vertices.count+6].position = {((float *)&px)[6], ((float *)&py)[6], ((float *)&pz)[6]};
		vertices.data[vertices.count+7].position = {((float *)&px)[7], ((float *)&py)[7], ((float *)&pz)[7]};
		#endif

		__m256 nx = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(d0, d4), _mm256_sub_ps(d1, d5)), _mm256_add_ps(_mm256_sub_ps(d2, d6), _mm256_sub_ps(d3, d7)));
		__m256 ny = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(d0, d2), _mm256_sub_ps(d1, d3)), _mm256_add_ps(_mm256_sub_ps(d4, d6), _mm256_sub_ps(d5, d7)));
		__m256 nz = _mm256_add_ps(_mm256_add_ps(_mm256_sub_ps(d0, d1), _mm256_sub_ps(d2, d3)), _mm256_add_ps(_mm256_sub_ps(d4, d5), _mm256_sub_ps(d6, d7)));
		
		#if 0
		_mm256_i32scatter_ps(&vertices.data[vertices.count].normal.x, _mm256_setr_epi32(0,6,12,18,24,30,36,42), nx, 4);
		_mm256_i32scatter_ps(&vertices.data[vertices.count].normal.y, _mm256_setr_epi32(0,6,12,18,24,30,36,42), ny, 4);
		_mm256_i32scatter_ps(&vertices.data[vertices.count].normal.z, _mm256_setr_epi32(0,6,12,18,24,30,36,42), nz, 4);
		#elif 1
		__m256 nw;
		unpack(nw, nz, ny, nx);
		vertex = vertices.data + vertices.count;
		_mm_maskstore_ps(vertex++->normal.s, _mm_set_epi32(0,-1,-1,-1), _mm256_extractf128_ps(nx, 0));
		_mm_maskstore_ps(vertex++->normal.s, _mm_set_epi32(0,-1,-1,-1), _mm256_extractf128_ps(ny, 0));
		_mm_maskstore_ps(vertex++->normal.s, _mm_set_epi32(0,-1,-1,-1), _mm256_extractf128_ps(nz, 0));
		_mm_maskstore_ps(vertex++->normal.s, _mm_set_epi32(0,-1,-1,-1), _mm256_extractf128_ps(nw, 0));
		_mm_maskstore_ps(vertex++->normal.s, _mm_set_epi32(0,-1,-1,-1), _mm256_extractf128_ps(nx, 1));
		_mm_maskstore_ps(vertex++->normal.s, _mm_set_epi32(0,-1,-1,-1), _mm256_extractf128_ps(ny, 1));
		_mm_maskstore_ps(vertex++->normal.s, _mm_set_epi32(0,-1,-1,-1), _mm256_extractf128_ps(nz, 1));
		_mm_maskstore_ps(vertex++->normal.s, _mm_set_epi32(0,-1,-1,-1), _mm256_extractf128_ps(nw, 1));
		#elif 1
		__m256 nw;
		unpack(nw, nz, ny, nx);
		memcpy(vertices.data[vertices.count+0].normal.s, (float *)&nx + 0, 12);
		memcpy(vertices.data[vertices.count+1].normal.s, (float *)&ny + 0, 12);
		memcpy(vertices.data[vertices.count+2].normal.s, (float *)&nz + 0, 12);
		memcpy(vertices.data[vertices.count+3].normal.s, (float *)&nw + 0, 12);
		memcpy(vertices.data[vertices.count+4].normal.s, (float *)&nx + 4, 12);
		memcpy(vertices.data[vertices.count+5].normal.s, (float *)&ny + 4, 12);
		memcpy(vertices.data[vertices.count+6].normal.s, (float *)&nz + 4, 12);
		memcpy(vertices.data[vertices.count+7].normal.s, (float *)&nw + 4, 12);
		#else
		vertices.data[vertices.count+0].normal = {((float *)&nx)[0], ((float *)&ny)[0], ((float *)&nz)[0]};
		vertices.data[vertices.count+1].normal = {((float *)&nx)[1], ((float *)&ny)[1], ((float *)&nz)[1]};
		vertices.data[vertices.count+2].normal = {((float *)&nx)[2], ((float *)&ny)[2], ((float *)&nz)[2]};
		vertices.data[vertices.count+3].normal = {((float *)&nx)[3], ((float *)&ny)[3], ((float *)&nz)[3]};
		vertices.data[vertices.count+4].normal = {((float *)&nx)[4], ((float *)&ny)[4], ((float *)&nz)[4]};
		vertices.data[vertices.count+5].normal = {((float *)&nx)[5], ((float *)&ny)[5], ((float *)&nz)[5]};
		vertices.data[vertices.count+6].normal = {((float *)&nx)[6], ((float *)&ny)[6], ((float *)&nz)[6]};
		vertices.data[vertices.count+7].normal = {((float *)&nx)[7], ((float *)&ny)[7], ((float *)&nz)[7]};
		#endif
	};
	#endif

	f32 fx = 0; u32 lx = 0; for (; lx < size.x-1; ++lx, ++fx) {
		#if !RESERVE_ALL
		// Expand if don't have full layer ahead.
		vertices.reserve_exponential(vertices.count + size.y * size.z);
		#endif
	f32 fy = 0; u32 ly = 0; for (; ly < size.y-1; ++ly, ++fy) {
	f32 fz = 0; u32 lz = 0; for (; lz < size.z-1; ++lz, ++fz) {
		
		u64 s = *(s64 *)&surface_bitfield[(lx*size.y*size.z + ly*size.z + lz) / 8] >> (lz%8);
		int z = min(count_trailing_zeros(s), 56u);
		if (z) {
			lz += z - 1;
			fz += z - 1;
			continue;
		}

		assert(s & 1);
		
		u32 r = 
			((u8)(*(u16*)&negative_bitfield.data[((lx+0)*size.y*size.z + (ly+0)*size.z + lz)/8] >> (lz%8)) << 0) |
			((u8)(*(u16*)&negative_bitfield.data[((lx+0)*size.y*size.z + (ly+1)*size.z + lz)/8] >> (lz%8)) << 8) |
			((u8)(*(u16*)&negative_bitfield.data[((lx+1)*size.y*size.z + (ly+0)*size.z + lz)/8] >> (lz%8)) << 16) |
			((u8)(*(u16*)&negative_bitfield.data[((lx+1)*size.y*size.z + (ly+1)*size.z + lz)/8] >> (lz%8)) << 24);
		
		#if SIMD_METHOD == DEFERRED
		////////////////////
		////////////////////
		//// 8 VERTICES ////
		////////////////////
		////////////////////
			
		((f32 *)&deferred.lx)[deferred_index] = fx;
		((f32 *)&deferred.ly)[deferred_index] = fy;
		((f32 *)&deferred.lz)[deferred_index] = fz;
		((f32 *)&deferred.d0)[deferred_index] = at(sdf.data, lx+0, ly+0, lz+0);
		((f32 *)&deferred.d1)[deferred_index] = at(sdf.data, lx+0, ly+0, lz+1);
		((f32 *)&deferred.d2)[deferred_index] = at(sdf.data, lx+0, ly+1, lz+0);
		((f32 *)&deferred.d3)[deferred_index] = at(sdf.data, lx+0, ly+1, lz+1);
		((f32 *)&deferred.d4)[deferred_index] = at(sdf.data, lx+1, ly+0, lz+0);
		((f32 *)&deferred.d5)[deferred_index] = at(sdf.data, lx+1, ly+0, lz+1);
		((f32 *)&deferred.d6)[deferred_index] = at(sdf.data, lx+1, ly+1, lz+0);
		((f32 *)&deferred.d7)[deferred_index] = at(sdf.data, lx+1, ly+1, lz+1);

		at(index_grid.data, lx & 1, ly, lz) = vertices.count + deferred_index;

		++deferred_index;

		if (deferred_index == 8) {
			write_deferred_vertices();

			vertices.count += deferred_index;
			deferred_index = 0;
		}

		#else
		not_implemented();
		#endif

		// Merged vertex and triangle creation loops
		// 2.5 gb/s
		if ((lx > 0) & (ly > 0) & (lz > 0)) {
			bool s0 = (r >> 0) & 1;
			bool sz = (r >> 1) & 1;
			bool sy = (r >> 8) & 1;
			bool sx = (r >> 16) & 1;

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

				#if CALCULATE_NORMALS == FROM_TRIANGLES
				auto &v0 = vertices.data[i0];
				auto &v1 = vertices.data[i1];
				auto &v2 = vertices.data[i2];
				auto &v3 = vertices.data[i3];
					
				v3f n0 = normalize(cross(v1.position - v0.position, v2.position - v0.position));
				v0.normal += n0;
				v1.normal += n0;
				v2.normal += n0;

				v3f n1 = normalize(cross(v3.position - v1.position, v2.position - v1.position));
				v1.normal += n1;
				v3.normal += n1;
				v2.normal += n1;
				#endif
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


			if (s0 != sx) {
				umm _0 = ((lx-0) & 1)*sizeyz + (ly-1)*size.z + (lz-1);
				umm _1 = ((lx-0) & 1)*sizeyz + (ly-0)*size.z + (lz-1);
				umm _2 = ((lx-0) & 1)*sizeyz + (ly-1)*size.z + (lz-0);
				umm _3 = ((lx-0) & 1)*sizeyz + (ly-0)*size.z + (lz-0);

				CSWAP;

				quad(_0, _1, _2, _3);
			}
			if (s0 != sy) {
				umm _0 = ((lx-0) & 1)*sizeyz + (ly-0)*size.z + (lz-0);
				umm _1 = ((lx-0) & 1)*sizeyz + (ly-0)*size.z + (lz-1);
				umm _2 = ((lx-1) & 1)*sizeyz + (ly-0)*size.z + (lz-0);
				umm _3 = ((lx-1) & 1)*sizeyz + (ly-0)*size.z + (lz-1);
					
				CSWAP;

				quad(_0, _1, _2, _3);
			}
			if (s0 != sz) {
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
		#if CALCULATE_NORMALS == FROM_TRIANGLES
		for (umm i = prev_layer_vertices_begin; i < prev_layer_vertices_end; ++i) {
			vertices.data[i].normal = normalize(vertices.data[i].normal);
		}
		prev_layer_vertices_begin = prev_layer_vertices_end;
		prev_layer_vertices_end = vertices.count;
		#endif
	}
	
	#if SIMD_METHOD == DEFERRED
	write_deferred_vertices();
	vertices.count += deferred_index;
	#endif
	
	indices.count = indices_count;
}
