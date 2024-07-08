#pragma once

#include "hittable_list.h"
#include "renderable.h"

#include <vector>
#include <algorithm>

struct BvhNode {
	// TODO: This part is actually  kinda hard, try to figure it out better later.
	// For now, im  assuming all this data in on the CPU, and that I can modify it.
	__host__ BvhNode(std::vector<Renderable*>& objects, int start, int end) {
		// TODO: Make actually random
		const int axis = 0;

		const auto comparator = (axis == 0) ? box_x_compare
			: (axis == 1) ? box_y_compare
			: box_z_compare;

		size_t object_span = end - start;

		if (object_span == 1) {
			left = right = objects[start];
		}
		else if (object_span == 2) {
			left = objects[start];
			right = objects[start + 1];
		}
		else {
			std::sort(objects.begin() + start, objects.begin() + end, comparator);

			auto mid = start + object_span / 2;

			left = (Renderable*)malloc(sizeof(Renderable));
			// TODO: I know for sure this is wrong, just writting some stuff down
			*left = Renderable::BvhNode(BvhNode(objects, start, mid));

			right = (Renderable*)malloc(sizeof(Renderable));
			// TODO: I know for sure this is wrong, just writting some stuff down
			*right = Renderable::BvhNode(BvhNode(objects, mid, end));
		}

		bbox = AABB(left->bounding_box(), right->bounding_box());
	}

	__device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
		if (!bbox.hit(r, ray_t)) {
			return false;
		}

		
		bool hit_left = left->hit(r, ray_t, rec);
		const Interval ray_interval = Interval(ray_t.min, hit_left ? rec.t : ray_t.max);
		bool hit_right = right->hit(r, ray_interval, rec);

		return hit_left || hit_right;
	}

	__device__ AABB bounding_box() const {
		return bbox;
	}

private:
	Renderable* left;
	Renderable* right;
	AABB bbox;

	static bool box_compare(
		const Renderable* a, const Renderable* b, int axis_index
	) {
		auto a_axis_interval = a->bounding_box().axis_interval(axis_index);
		auto b_axis_interval = b->bounding_box().axis_interval(axis_index);
		return a_axis_interval.min < b_axis_interval.min;
	}

	static bool box_x_compare(const Renderable* a, const Renderable* b) {
		return box_compare(a, b, 0);
	}

	static bool box_y_compare(const Renderable* a, const Renderable* b) {
		return box_compare(a, b, 1);
	}

	static bool box_z_compare(const Renderable* a, const Renderable* b) {
		return box_compare(a, b, 2);
	}
};