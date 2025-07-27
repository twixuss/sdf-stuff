#pragma once
#include <tl/common.h>
#include <tl/math.h>

using namespace tl;
using String = Span<utf8>;

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

#define RESERVE_ALL 1

#define FROM_SDF       0
#define FROM_TRIANGLES 1
#define CALCULATE_NORMALS FROM_SDF
