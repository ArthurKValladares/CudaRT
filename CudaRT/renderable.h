#pragma once

#include "sphere.h"

enum class RenderableType {
	Sphere
};

union RenderablePayload {
	Sphere sphere;
};

struct RenderableData {
	RenderableType type;
	RenderablePayload payload;
};

struct Renderable {
	Renderable() = delete;

	__device__ Renderable(RenderableData data) : data(data) {}

	__device__ static Renderable Sphere(const Vec3f32& center, float radius, Material material) {
		RenderablePayload payload = RenderablePayload{
			 Sphere::Sphere(center, radius, material)
		};
		return Renderable {
			RenderableData {
				RenderableType::Sphere,
				payload
			}
		};
	}

	__device__ static Renderable Sphere(const Vec3f32& center1, const Vec3f32& center2, float radius, Material material) {
		RenderablePayload payload = RenderablePayload{
			Sphere::Sphere(center1, center2, radius, material)
		};
		return Renderable{
			RenderableData {
				RenderableType::Sphere,
				payload
			}
		};
	}


	__device__ bool hit(const Ray& r, float t_min, float t_max, HitRecord& rec) const {
		switch (data.type) {
			case RenderableType::Sphere : {
				return data.payload.sphere.hit(r, t_min, t_max, rec);
			}
		}
	}

private:
	RenderableData data;
};