#pragma once

#include "hittable.h"
#include "sphere.h"

class HittableList {
public:
	__device__ HittableList() = delete;

	__device__ HittableList(Sphere** hittables, int size)
		: hittables(hittables)
		, list_size(size)
	{
	}

	__device__ bool HittableList::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
		HitRecord temp_rec;
		bool hit_anything = false;
		float closest_so_far = t_max;
		for (int i = 0; i < list_size; i++) {
			if (hittables[i]->hit(r, t_min, closest_so_far, temp_rec)) {
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}
		return hit_anything;
	}

private:
	Sphere** hittables;
	int list_size;
};