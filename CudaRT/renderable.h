#pragma once

#include "sphere.h"

enum class RenderableType {
	Sphere
};

union RenderablePayload {
	Sphere sphere;

	__device__ RenderablePayload& operator=(const RenderablePayload& payload) {
		memcpy(this, &payload, sizeof(RenderablePayload));

		return *this;
	}
};

struct RenderableData {
	RenderableType type;
	RenderablePayload payload;

	__device__ RenderableData& operator=(const RenderableData& data) {
		type = data.type;
		payload = data.payload;

		return *this;
	}
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

	__device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
		switch (data.type) {
			case RenderableType::Sphere : {
				return data.payload.sphere.hit(r, ray_t, rec);
			}
		}
	}

	__device__ Renderable& operator=(const Renderable& renderable) {
		data = renderable.data;

		return *this;
	}

private:
	RenderableData data;
};