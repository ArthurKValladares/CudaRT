#pragma once

#include <assert.h>

#include "defs.h"
#include "math.h"
#include "vec3.h"
#include "ray.h"

struct Material;
class HitRecord {
public:
	Vec3f32 p;
	Vec3f32 normal;
	float t;
	bool front_face;
	const Material* material;

	__device__ void set_face_normal(const Ray& r, const Vec3f32& outward_normal) {
		// TODO: Figure this out
		//assert(float_equals(outward_normal.length(), 1.0));

		front_face = dot(r.direction(), outward_normal) < 0;
		normal = front_face ? outward_normal : -outward_normal;
	}
};