#include "common.h"

void sdf_to_triangles_starting(v3u size, Span<f32> sdf, GList<Vertex> &vertices, GList<u32> &indices, GList<u32> &index_grid) {
	assert(size.x * size.y * size.z == sdf.count);
	defer {
		for (auto index : indices) {
			assert(index < vertices.count);
		}
	};
	
	#if !RESERVE_ALL
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
			#if !RESERVE_ALL
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
			#if !RESERVE_ALL
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
