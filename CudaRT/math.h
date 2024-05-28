#pragma once

#include <curand_kernel.h>

#define EPSILON 0.000001f

template<class T>
__host__ __device__ T lerp(T a, T b, float t) {
	(1.0 - t) * a + t * b;
}

#define float_equals(x, y) (fabsf(x - y)) <= (EPSILON * fmaxf(1.0f, fmaxf(fabsf(x), fabsf(y))))

__device__ float random_float(curandState& rand_state) {
	return curand_uniform(&rand_state);
}

__device__ float random_float(curandState& rand_state, float min, float max) {
	float myrandf = curand_uniform(&rand_state);
	myrandf *= (max - min + 0.999999);
	myrandf += min;
	int myrand = (int)truncf(myrandf);
}