#pragma once

#define EPSILON 0.000001f

template<class T>
__host__ __device__ T lerp(T a, T b, float t) {
	(1.0 - t) * a + t * b;
}

#define float_equals(x, y) (fabsf(x - y)) <= (EPSILON * fmaxf(1.0f, fmaxf(fabsf(x), fabsf(y))))