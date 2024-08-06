#pragma once

#include "sphere.h"
#include "quad.h"
#include "transforms.h"

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

		Vec3f32 origin = r.origin();
		//origin[0] = rotation.m_cos_theta * r.origin()[0] - rotation.m_sin_theta * r.origin()[2];
		//origin[2] = rotation.m_sin_theta * r.origin()[0] + rotation.m_cos_theta * r.origin()[2];
		origin -= translation.m_offset;

		Vec3f32 direction = r.direction();
		//direction[0] = rotation.m_cos_theta * r.direction()[0] - rotation.m_sin_theta * r.direction()[2];
		//direction[2] = rotation.m_sin_theta * r.direction()[0] + rotation.m_cos_theta * r.direction()[2];

		const Ray transformed_ray = Ray(origin, direction, r.time());

		bool hit = false;
		switch (data.type) {
		case RenderableType::Sphere: {
			hit = data.payload.sphere.hit(transformed_ray, ray_t, rec);
			break;
		}
		case RenderableType::Quad: {
			hit = data.payload.quad.hit(transformed_ray, ray_t, rec);
			break;
		}
		default: {
			assert(false);
			return false;
		}
		}

		Vec3f32 p = rec.p;
		//p[0] = rotation.m_cos_theta * rec.p[0] + rotation.m_sin_theta * rec.p[2];
		//p[2] = -rotation.m_sin_theta * rec.p[0] + rotation.m_cos_theta * rec.p[2];
		p += translation.m_offset;

		Vec3f32 normal = rec.normal;
		//normal[0] = rotation.m_cos_theta * rec.normal[0] + rotation.m_sin_theta * rec.normal[2];
		//normal[2] = -rotation.m_sin_theta * rec.normal[0] + rotation.m_cos_theta * rec.normal[2];

		rec.p = p;
		rec.normal = normal;

		return hit;
	}

	__device__ Renderable& operator=(const Renderable& renderable) {
		data = renderable.data;

		return *this;
	}

	__device__ void set_rotation(float angle) {
		rotation = Rotation(angle);
	}

	__device__ void set_translation(Vec3f32 offset) {
		translation = Translation(offset);
	}

private:
	RenderableData data;

	Translation translation;
	Rotation rotation;
};