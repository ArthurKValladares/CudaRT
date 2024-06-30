#pragma once

#include <curand_kernel.h>
#include <math_constants.h>
#define _USE_MATH_DEFINES
#include <math.h>

#define EPSILON 0.000001f
#define float_equals(x, y) (fabsf(x - y)) <= (EPSILON * fmaxf(1.0f, fmaxf(fabsf(x), fabsf(y))))

template<class T>
__host__ __device__ T lerp(T a, T b, float t) {
	(1.0 - t) * a + t * b;
}

__host__ __device__ inline double degrees_to_radians(double degrees) {
	return degrees * M_PI / 180.0;
}

__device__ float random_float(curandState* rand_state) {
	return curand_uniform(rand_state);
}

__device__ float random_float(curandState* rand_state, float min, float max) {
	return lerp(min, max, random_float(rand_state));
}