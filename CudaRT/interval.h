#pragma once

#include "math.h"

struct Interval {
	__device__ Interval() : min(+INFINITY), max(-INFINITY) {}
	__device__ Interval(float min, float max) : min(min), max(max) {}
	__device__ Interval(const Interval& a, const Interval& b) {
		min = a.min <= b.min ? a.min : b.min;
		max = a.max >= b.max ? a.max : b.max;
	}

	float min, max;
};