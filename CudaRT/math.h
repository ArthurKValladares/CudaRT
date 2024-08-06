#pragma once

#include <curand_kernel.h>
#include <math_constants.h>
#define _USE_MATH_DEFINES
#include <math.h>

#define EPSILON 0.000001f
#define float_equals(x, y) (fabsf(x - y)) <= (EPSILON * fmaxf(1.0f, fmaxf(fabsf(x), fabsf(y))))

template<class T>
__host__ __device__ T lerp(T a, T b, float t) {
	return (1.0 - t) * a + t * b;
}

__host__ __device__ int clamp(int x, int low, int high) {
	if (x < low) return low;
	if (x < high) return x;
	return high - 1;
}

__host__ __device__ inline float degrees_to_radians(float degrees) {
	return degrees * M_PI / 180.0;
}

struct LocalRandomState {
	curandState rand_state;
};

__device__ float random_float(LocalRandomState& local_rand_state) {
	return curand_uniform(&local_rand_state.rand_state);
}

__device__ float random_float(LocalRandomState& local_rand_state, float min, float max) {
	const float t = curand_uniform(&local_rand_state.rand_state);
	return lerp(min, max, t);
}

__device__ int random_int(LocalRandomState& local_rand_state) {
	return random_float(local_rand_state);
}

__device__ float random_int(LocalRandomState& local_rand_state, int min, int max) {
	return random_float(local_rand_state, min, max);
}