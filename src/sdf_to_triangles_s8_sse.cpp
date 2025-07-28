#include "common.h"
#include <tl/console.h>

__m128i sign_epi8(__m128i m) {
	//__m128i h = _mm_and_si128(_mm_srai_epi16(m, 7), _mm_set1_epi16(0xff00));
	//__m128i l = _mm_and_si128(_mm_srai_epi16(_mm_srai_epi16(m, 8), 7), _mm_set1_epi16(0x00ff));
	//return _mm_or_si128(l, h);

	return _mm_blendv_epi8(_mm_set1_epi8(0), _mm_set1_epi8(-1), m);
}

template <int count>
__m128i slli_epi8(__m128i m) {
	return _mm_and_si128(_mm_slli_epi16(m, count), _mm_set1_epi8(0xff << count));
}
template <int count>
__m128i srli_epi8(__m128i m) {
	return _mm_and_si128(_mm_srli_epi16(m, count), _mm_set1_epi8(0xff >> count));
}

u8 div_epu8(u8 a, u8 b) {
    if (b == 0) {
        return 255;
    }

	u8 y = 1;
	while ((b << 1) < a) {
		b <<= 1;
		y <<= 1;
	}

	u8 r = 0;
	while (y) {
		if (a < b) {
			b >>= 1;
			y >>= 1;
		} else {
			a -= b;
			r += y;
		}
	}
	return r;
}


__m128i cmplt_epu8(__m128i a, __m128i b) {
	return _mm_cmplt_epi8(_mm_xor_si128(a, _mm_set1_epi8(0x80)), _mm_xor_si128(b, _mm_set1_epi8(0x80)));
}

__m128i div_epu8(__m128i a, __m128i b) {
	__m128i dbz = _mm_cmpeq_epi8(b, _mm_setzero_si128());
	__m128i y = _mm_set1_epi8(1);
	while (1) {
		__m128i s = slli_epi8<1>(b);
		__m128i c = cmplt_epu8(s, a);
		if (_mm_movemask_epi8(c) == 0)
			break;
		b = _mm_blendv_epi8(b, s, c);
		y = _mm_blendv_epi8(y, slli_epi8<1>(y), c);
	}
	__m128i r = _mm_setzero_si128();
	while (1) {
		__m128i c = _mm_cmpeq_epi8(y, _mm_setzero_si128());
		if (_mm_movemask_epi8(c) == 0)
			break;

		c = cmplt_epu8(a, b);
		b = _mm_blendv_epi8(b, slli_epi8<1>(b), c);
		y = _mm_blendv_epi8(y, slli_epi8<1>(y), c);
		a = _mm_blendv_epi8(_mm_sub_epi8(a, b), a, c);
		r = _mm_blendv_epi8(_mm_add_epi8(r, y), r, c);
	}
	return r;
}

void sdf_to_triangles_s8_sse(v3u size, Span<s8> sdf, Span<u8> negative_bitfield, Span<u8> surface_bitfield, GList<Vertex> &vertices, GList<u32> &indices, GList<u32> &index_grid) {
    __m128i a = _mm_set_epi8(160,150,140,130,120,110,100,90,80,70,60,50,40,30,20,10);
    __m128i b = _mm_set1_epi8(10);
    __m128i c = div_epu8(a, b);
    print("{} ", div_epu8(_mm_extract_epi8(a, 15), 10));
    print("{} ", div_epu8(_mm_extract_epi8(a, 14), 10));
    print("{} ", div_epu8(_mm_extract_epi8(a, 13), 10));
    print("{} ", div_epu8(_mm_extract_epi8(a, 12), 10));
    print("{} ", div_epu8(_mm_extract_epi8(a, 11), 10));
    print("{} ", div_epu8(_mm_extract_epi8(a, 10), 10));
    print("{} ", div_epu8(_mm_extract_epi8(a, 9), 10));
    print("{} ", div_epu8(_mm_extract_epi8(a, 8), 10)); 
    print("{} ", div_epu8(_mm_extract_epi8(a, 7), 10));
    print("{} ", div_epu8(_mm_extract_epi8(a, 6), 10));
    print("{} ", div_epu8(_mm_extract_epi8(a, 5), 10));
    print("{} ", div_epu8(_mm_extract_epi8(a, 4), 10));
    print("{} ", div_epu8(_mm_extract_epi8(a, 3), 10));
    print("{} ", div_epu8(_mm_extract_epi8(a, 2), 10));
    print("{} ", div_epu8(_mm_extract_epi8(a, 1), 10));
    print("{} ", div_epu8(_mm_extract_epi8(a, 0), 10));
	println();
    print("{} ", _mm_extract_epi8(c, 15));
    print("{} ", _mm_extract_epi8(c, 14));
    print("{} ", _mm_extract_epi8(c, 13));
    print("{} ", _mm_extract_epi8(c, 12));
    print("{} ", _mm_extract_epi8(c, 11));
    print("{} ", _mm_extract_epi8(c, 10));
    print("{} ", _mm_extract_epi8(c, 9));
    print("{} ", _mm_extract_epi8(c, 8));
    print("{} ", _mm_extract_epi8(c, 7));
    print("{} ", _mm_extract_epi8(c, 6));
    print("{} ", _mm_extract_epi8(c, 5));
    print("{} ", _mm_extract_epi8(c, 4));
    print("{} ", _mm_extract_epi8(c, 3));
    print("{} ", _mm_extract_epi8(c, 2));
    print("{} ", _mm_extract_epi8(c, 1));
    print("{} ", _mm_extract_epi8(c, 0));

	exit(1);

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
			__m128i r11 = _mm_loadu_si128((__m128i *)&at(sdf.data, lx+0, ly+0, lz+0));
			__m128i r12 = _mm_loadu_si128((__m128i *)&at(sdf.data, lx+0, ly+1, lz+0));
			__m128i r21 = _mm_loadu_si128((__m128i *)&at(sdf.data, lx+1, ly+0, lz+0));
			__m128i r22 = _mm_loadu_si128((__m128i *)&at(sdf.data, lx+1, ly+1, lz+0));
			// ? ? ? ? ? ? ? ? ? ? ? ? ? ? 1 0
			// ? ? ? ? ? ? ? ? ? ? ? ? ? ? 3 2
			// ? ? ? ? ? ? ? ? ? ? ? ? ? ? 5 4
			// ? ? ? ? ? ? ? ? ? ? ? ? ? ? 7 6

			__m128i d = _mm_or_si128(_mm_or_si128(r11, _mm_slli_si128(r12, 2)), _mm_or_si128(_mm_slli_si128(r21, 4), _mm_slli_si128(r22, 6)));
			// ? ? ? ? ? ? ? ? 7 6 5 4 3 2 1 0

			at(index_grid.data, lx & 1, ly, lz) = vertices.count;
			Vertex *vertex = &vertices.data[vertices.count++];

			__m128i d642054103210 = _mm_shuffle_epi8(d, _mm_set_epi8(-1,-1,-1,-1,6,4,2,0,5,4,1,0,3,2,1,0));
			__m128i d753176327654 = _mm_shuffle_epi8(d, _mm_set_epi8(-1,-1,-1,-1,7,5,3,1,7,6,3,2,7,6,5,4));
			__m128i c = sign_epi8(_mm_xor_si128(d642054103210, d753176327654));

			int cmask = _mm_movemask_epi8(c);

			__m128 d3210 = _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_shuffle_epi8(d, _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,3,2,1,0))));
			#if 0
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
			#endif
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
