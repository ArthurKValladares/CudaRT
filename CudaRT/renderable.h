#pragma once

#include "sphere.h"
#include "quad.h"

enum class RenderableType {
	Sphere,
	Quad
};

union RenderablePayload {
	Sphere sphere;
	Quad quad;

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

	__device__ static Renderable Quad(const Vec3f32& Q, const Vec3f32& u, const Vec3f32& v, Material mat) {
		RenderablePayload payload = RenderablePayload{};
		payload.quad = Quad::Quad(Q, u, v, mat);
		return Renderable{
			RenderableData {
				RenderableType::Quad,
				payload
			}
		};
	}

	__device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
		switch (data.type) {
			case RenderableType::Sphere : {
				return data.payload.sphere.hit(r, ray_t, rec);
			}
			case RenderableType::Quad: {
				return data.payload.quad.hit(r, ray_t, rec);
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