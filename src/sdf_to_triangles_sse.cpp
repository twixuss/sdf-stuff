#include "common.h"

// 
// 3 2 1 0
// 
// d c b a  m0
// h g f e  m1
// l k j i  m2
// p o n m  m3
//
//  =>
//
// m i e a  m0
// n j f b  m1
// o k g c  m2
// p l h d  m3
//
forceinline static void transpose(__m128 &ponm, __m128 &lkji, __m128 &hgfe, __m128 &dcba) {
	__m128 fbea = _mm_unpacklo_ps(dcba, hgfe);
	__m128 hdgc = _mm_unpackhi_ps(dcba, hgfe);
	__m128 njmi = _mm_unpacklo_ps(lkji, ponm);
	__m128 plok = _mm_unpackhi_ps(lkji, ponm);
	__m128 miea = _mm_movelh_ps(fbea, njmi);
	__m128 njfb = _mm_movehl_ps(njmi, fbea);
	__m128 okgc = _mm_movelh_ps(hdgc, plok);
	__m128 plhd = _mm_movehl_ps(plok, hdgc);
	dcba = miea;
	hgfe = njfb;
	lkji = okgc;
	ponm = plhd;
}

#define pshuf_ps(a, b) _mm_castsi128_ps(_mm_shuffle_epi8(_mm_castps_si128(a), b))

#if SIMD_METHOD == DEFERRED
struct Deferred {
	__m128 d0;
	__m128 d1;
	__m128 d2;
	__m128 d3;
	__m128 d4;
	__m128 d5;
	__m128 d6;
	__m128 d7;
	__m128 lx;
	__m128 ly;
	__m128 lz;
};
extern "C" void flush_deferred_sse_asm(Deferred *deferred, GList<Vertex> *vertices);
#endif

void verify_sdf_aux_bit_fields(v3u size, Span<f32> sdf, Span<u8> negative_bitfield, Span<u8> surface_bitfield);

void sdf_to_triangles_sse(v3u size, Span<f32> sdf, Span<u8> negative_bitfield, Span<u8> surface_bitfield, GList<Vertex> &vertices, GList<u32> &indices, GList<u32> &index_grid) {
	auto at = [&](auto &arr, u32 x, u32 y, u32 z) -> decltype(auto) {
		return arr[x * (size.y*size.z) + y * (size.z) + z];
	};

	// from 1.35 to 1.81 gb/s ??? when assert expands to nothing?
	#if 0
	assert(size.x * size.y * size.z <= sdf.count);
	assert(size.x * size.y * size.z / 8 <= negative_bitfield.count);
	assert(size.x * size.y * size.z / 8 <= surface_bitfield.count);
	verify_sdf_aux_bit_fields(size, sdf, negative_bitfield, surface_bitfield);
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
	Deferred deferred;
	u8 deferred_index = 0;

	#pragma push_macro("forceinline")
	#undef forceinline
	auto write_deferred_vertices = [&] () [[msvc::forceinline]] {
	#pragma pop_macro("forceinline")
		__m128 d0 = deferred.d0;
		__m128 d1 = deferred.d1;
		__m128 d2 = deferred.d2;
		__m128 d3 = deferred.d3;
		__m128 d4 = deferred.d4;
		__m128 d5 = deferred.d5;
		__m128 d6 = deferred.d6;
		__m128 d7 = deferred.d7;
		
		// tried removing these shifts and replacing latter ands with blends. became bit slower. almost the same.
		// NOTE: these are 32 bit booleans, very wasteful. I tried compressing them into 8 bits when writing to Deferred,
		//       but conversion between 8 - 32 bits is very slow, not worth for reducing number of xors.
		__m128 s0 = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(d0), 31));
		__m128 s1 = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(d1), 31));
		__m128 s2 = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(d2), 31));
		__m128 s3 = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(d3), 31));
		__m128 s4 = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(d4), 31));
		__m128 s5 = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(d5), 31));
		__m128 s6 = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(d6), 31));
		__m128 s7 = _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(d7), 31));

		__m128 c0 = _mm_xor_ps(s0, s1);
		__m128 c1 = _mm_xor_ps(s2, s3);
		__m128 c2 = _mm_xor_ps(s4, s5);
		__m128 c3 = _mm_xor_ps(s6, s7);
		__m128 c4 = _mm_xor_ps(s0, s2);
		__m128 c5 = _mm_xor_ps(s1, s3);
		__m128 c6 = _mm_xor_ps(s4, s6);
		__m128 c7 = _mm_xor_ps(s5, s7);
		__m128 c8 = _mm_xor_ps(s0, s4);
		__m128 c9 = _mm_xor_ps(s1, s5);
		__m128 ca = _mm_xor_ps(s2, s6);
		__m128 cb = _mm_xor_ps(s3, s7);

		__m128 t0 = _mm_and_ps(_mm_div_ps(d0, _mm_sub_ps(d0, d1)), c0);
		__m128 t1 = _mm_and_ps(_mm_div_ps(d2, _mm_sub_ps(d2, d3)), c1);
		__m128 t2 = _mm_and_ps(_mm_div_ps(d4, _mm_sub_ps(d4, d5)), c2);
		__m128 t3 = _mm_and_ps(_mm_div_ps(d6, _mm_sub_ps(d6, d7)), c3);
		__m128 t4 = _mm_and_ps(_mm_div_ps(d0, _mm_sub_ps(d0, d2)), c4);
		__m128 t5 = _mm_and_ps(_mm_div_ps(d1, _mm_sub_ps(d1, d3)), c5);
		__m128 t6 = _mm_and_ps(_mm_div_ps(d4, _mm_sub_ps(d4, d6)), c6);
		__m128 t7 = _mm_and_ps(_mm_div_ps(d5, _mm_sub_ps(d5, d7)), c7);
		__m128 t8 = _mm_and_ps(_mm_div_ps(d0, _mm_sub_ps(d0, d4)), c8);
		__m128 t9 = _mm_and_ps(_mm_div_ps(d1, _mm_sub_ps(d1, d5)), c9);
		__m128 ta = _mm_and_ps(_mm_div_ps(d2, _mm_sub_ps(d2, d6)), ca);
		__m128 tb = _mm_and_ps(_mm_div_ps(d3, _mm_sub_ps(d3, d7)), cb);

		__m128 f1111 = _mm_set1_ps(1);
		c0 = _mm_and_ps(c0, f1111);
		c1 = _mm_and_ps(c1, f1111);
		c2 = _mm_and_ps(c2, f1111);
		c3 = _mm_and_ps(c3, f1111);
		c4 = _mm_and_ps(c4, f1111);
		c5 = _mm_and_ps(c5, f1111);
		c6 = _mm_and_ps(c6, f1111);
		c7 = _mm_and_ps(c7, f1111);
		c8 = _mm_and_ps(c8, f1111);
		c9 = _mm_and_ps(c9, f1111);
		ca = _mm_and_ps(ca, f1111);
		cb = _mm_and_ps(cb, f1111);

		
		__m128 px = _mm_add_ps(_mm_add_ps(_mm_add_ps(t8, t9), _mm_add_ps(ta, tb)), _mm_add_ps(_mm_add_ps(c2, c3), _mm_add_ps(c6, c7)));
		__m128 py = _mm_add_ps(_mm_add_ps(_mm_add_ps(t4, t5), _mm_add_ps(t6, t7)), _mm_add_ps(_mm_add_ps(c1, c3), _mm_add_ps(ca, cb)));
		__m128 pz = _mm_add_ps(_mm_add_ps(_mm_add_ps(t0, t1), _mm_add_ps(t2, t3)), _mm_add_ps(_mm_add_ps(c5, c9), _mm_add_ps(c7, cb)));
		
		__m128 c = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(c0, c1),
		                                            _mm_add_ps(c2, c3)),
		                                 _mm_add_ps(_mm_add_ps(c4, c5),
		                                            _mm_add_ps(c6, c7))),
			                             _mm_add_ps(_mm_add_ps(c8, c9),
		                                            _mm_add_ps(ca, cb)));
		
		px = _mm_add_ps(_mm_div_ps(px, c), deferred.lx);
		py = _mm_add_ps(_mm_div_ps(py, c), deferred.ly);
		pz = _mm_add_ps(_mm_div_ps(pz, c), deferred.lz);
		
		__m128 pw;
		transpose(pw, pz, py, px);
		Vertex *vertex = vertices.data + vertices.count;
		_mm_storeu_ps(vertex++->position.s, px);
		_mm_storeu_ps(vertex++->position.s, py);
		_mm_storeu_ps(vertex++->position.s, pz);
		_mm_storeu_ps(vertex++->position.s, pw);

		__m128 nx = _mm_add_ps(_mm_add_ps(_mm_sub_ps(d0, d4), _mm_sub_ps(d1, d5)), _mm_add_ps(_mm_sub_ps(d2, d6), _mm_sub_ps(d3, d7)));
		__m128 ny = _mm_add_ps(_mm_add_ps(_mm_sub_ps(d0, d2), _mm_sub_ps(d1, d3)), _mm_add_ps(_mm_sub_ps(d4, d6), _mm_sub_ps(d5, d7)));
		__m128 nz = _mm_add_ps(_mm_add_ps(_mm_sub_ps(d0, d1), _mm_sub_ps(d2, d3)), _mm_add_ps(_mm_sub_ps(d4, d5), _mm_sub_ps(d6, d7)));
		__m128 il = _mm_rsqrt_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(nx,nx), _mm_mul_ps(ny,ny)), _mm_mul_ps(nz,nz)));
		nx = _mm_mul_ps(nx, il);
		ny = _mm_mul_ps(ny, il);
		nz = _mm_mul_ps(nz, il);

		__m128 nw;
		transpose(nw, nz, ny, nx);
		vertex = vertices.data + vertices.count;
		memcpy(&vertex++->normal, &nx, 12);
		memcpy(&vertex++->normal, &ny, 12);
		memcpy(&vertex++->normal, &nz, 12);
		memcpy(&vertex++->normal, &nw, 12);
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
		//// 4 VERTICES ////
		////////////////////
		////////////////////
			
		((f32 *)&deferred.d0)[deferred_index] = at(sdf.data, lx+0, ly+0, lz+0);
		((f32 *)&deferred.d1)[deferred_index] = at(sdf.data, lx+0, ly+0, lz+1);
		((f32 *)&deferred.d2)[deferred_index] = at(sdf.data, lx+0, ly+1, lz+0);
		((f32 *)&deferred.d3)[deferred_index] = at(sdf.data, lx+0, ly+1, lz+1);
		((f32 *)&deferred.d4)[deferred_index] = at(sdf.data, lx+1, ly+0, lz+0);
		((f32 *)&deferred.d5)[deferred_index] = at(sdf.data, lx+1, ly+0, lz+1);
		((f32 *)&deferred.d6)[deferred_index] = at(sdf.data, lx+1, ly+1, lz+0);
		((f32 *)&deferred.d7)[deferred_index] = at(sdf.data, lx+1, ly+1, lz+1);
		((f32 *)&deferred.lx)[deferred_index] = fx;
		((f32 *)&deferred.ly)[deferred_index] = fy;
		((f32 *)&deferred.lz)[deferred_index] = fz;

		at(index_grid.data, lx & 1, ly, lz) = vertices.count + deferred_index;

		++deferred_index;

		if (deferred_index == 4) {
			//flush_deferred_sse_asm(&deferred, &vertices);

			write_deferred_vertices();

			vertices.count += deferred_index;
			deferred_index = 0;
		}

		#else
			
		////////////////////
		////////////////////
		//// ONE VERTEX ////
		////////////////////
		////////////////////
		
		assert(r != 0 && (r & 0x03030303) != 0x03030303);

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

		#if SIMD_METHOD == HORIZONTAL

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
				
		__m128 invlen = _mm_rsqrt_ss(_mm_dp_ps(normal, normal, 0b0111'0001));
		normal = _mm_mul_ps(normal, _mm_shuffle_ps(invlen, invlen, _MM_SHUFFLE(0,0,0,0)));
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
