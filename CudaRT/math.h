#pragma once

template<class T>
__host__ __device__ T lerp(T a, T b, float t) {
	(1.0 - t) * a + t * b;
}