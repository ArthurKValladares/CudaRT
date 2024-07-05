#pragma once

#include "math.h"

struct Interval {
	__device__ Interval() : min(+INFINITY), max(-INFINITY) {}
	__device__ Interval(float min, float max) : min(min), max(max) {}
	__device__ Interval(const Interval& a, const Interval& b) {
		min = a.min <= b.min ? a.min : b.min;
		max = a.max >= b.max ? a.max : b.max;
	}

	__device__ float clamp(float x) const {
		if (x < min) return min;
		if (x > max) return max;
		return x;
	}

	__device__ Interval expand(float delta) const {
		auto padding = delta / 2;
		return Interval(min - padding, max + padding);
	}

	float min, max;
};