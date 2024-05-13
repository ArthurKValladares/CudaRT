#pragma once

#include <assert.h>

#include "math.h"
#include "vec3.h"
#include "ray.h"

class hit_record {
public:
	vec3 p;
	vec3 normal;
	float t;
	bool front_face;

	__device__ void set_face_normal(const ray& r, const vec3& outward_normal) {
		assert(float_equals(outward_normal.length(), 1.0));

		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};