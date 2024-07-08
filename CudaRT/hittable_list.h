#pragma once

#include "hittable.h"
#include "renderable.h"

struct HittableList {
	__device__ HittableList() = delete;

	__device__ HittableList(Renderable* hittables, int size)
		: hittables(hittables)
		, list_size(size)
	{
		m_bounding_box = AABB();
		for (int i = 0; i < list_size; ++i) {
			m_bounding_box = AABB(m_bounding_box, hittables[i].bounding_box());
		}
	}

	__device__ bool HittableList::hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
		HitRecord temp_rec;
		bool hit_anything = false;
		float closest_so_far = t_max;
		for (int i = 0; i < list_size; i++) {
			if (hittables[i].hit(r, Interval(t_min, closest_so_far), temp_rec)) {
				hit_anything = true;
				closest_so_far = temp_rec.t;
				rec = temp_rec;
			}
		}
		return hit_anything;
	}

	Renderable* hittables;
	int list_size;
	AABB m_bounding_box;
};