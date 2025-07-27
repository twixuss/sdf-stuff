#include "common.h"

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

	for (u32 lx = 0; lx < size.x-1; ++lx) {
		#if !RESERVE_ALL
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

			#if CALCULATE_NORMALS == FROM_SDF
			__m128 normal = _mm_or_ps(_mm_or_ps(
				_mm_dp_ps(_mm_sub_ps(d3210, d7654), f1111, 0b1111'0001),
				_mm_dp_ps(_mm_sub_ps(d5410, d7632), f1111, 0b1111'0010)),
				_mm_dp_ps(_mm_sub_ps(d6420, d7531), f1111, 0b1111'0100));
				
			normal = _mm_mul_ps(normal, _mm_permute_ps(_mm_rsqrt_ss(_mm_dp_ps(normal, normal, 0b0111'0001)), _MM_SHUFFLE(0,0,0,0)));
			_mm_storeu_ps((float *)&vertex->normal, normal);
			#endif
			#if CALCULATE_NORMALS == FROM_TRIANGLES
			memset(&vertex->normal, 0, sizeof(vertex->normal));
			#endif
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
		#if CALCULATE_NORMALS == FROM_TRIANGLES
		for (umm i = prev_layer_vertices_begin; i < prev_layer_vertices_end; ++i) {
			vertices.data[i].normal = normalize(vertices.data[i].normal);
		}
		prev_layer_vertices_begin = prev_layer_vertices_end;
		prev_layer_vertices_end = vertices.count;
		#endif
	}
	
	indices.count = indices_count;
}
