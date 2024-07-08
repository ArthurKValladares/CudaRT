#pragma once

#include "vec3.h"
#include "ray.h"
#include "hittable.h"
#include "material.h"
#include "aabb.h"

class Sphere {
public:
    __device__ __host__ Sphere(const Vec3f32& center, float radius, Material material)
		: center(center)
		, radius(radius)
        , material(material)
        , is_moving(false)
	{
        const Vec3f32 rvec = Vec3f32(radius, radius, radius);
        m_bounding_box = AABB(center - rvec, center + rvec);
    }

    __device__ __host__ Sphere(const Vec3f32& center1, const Vec3f32& center2, float radius, Material material)
        : center(center1)
        , radius(radius)
        , material(material)
        , is_moving(true)
    {
        center_vec = center2 - center1;

        const Vec3f32 rvec = Vec3f32(radius, radius, radius);
        const AABB box1(center1 - rvec, center1 + rvec);
        const AABB box2(center2 - rvec, center2 + rvec);
        m_bounding_box = AABB(box1, box2);
    }

    __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        const float ray_tmin = ray_t.min;
        const float ray_tmax = ray_t.max;
        Vec3f32 oc = (is_moving ? sphere_center(r.time()) : center) - r.origin();
        auto a = r.direction().squared_length();
        auto h = dot(r.direction(), oc);
        auto c = oc.squared_length() - radius * radius;

        auto discriminant = h * h - a * c;
        if (discriminant < 0) {
            return false;
        }
        auto sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root) {
            root = (h + sqrtd) / a;
            if (root <= ray_tmin || ray_tmax <= root)
                return false;
        }

        rec.t = root;
        rec.p = r.at(rec.t);
        // TODO: Do i need to normalize this here?
        Vec3f32 outward_normal = unit_vector((rec.p - center) / radius);
        rec.set_face_normal(r, outward_normal);
        rec.material = &material;

        return true;
	}

    __device__ AABB bounding_box() const {
        return m_bounding_box;
    }

private:
    __device__ Vec3f32 sphere_center(float time) const {
        return center + center_vec * time;
    }

	Vec3f32 center;
	float radius;
    Material material;
    bool is_moving;
    Vec3f32 center_vec;
    AABB m_bounding_box;
};