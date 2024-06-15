#pragma once

#include "vec3.h"
#include "ray.h"
#include "hittable.h"
#include "material.h"

class Sphere {
public:
    __device__ Sphere(const Vec3f32& center, float radius, Material material)
		: center(center)
		, radius(radius)
        , material(material)
	{}

    __device__ bool hit(const Ray& r, double ray_tmin, double ray_tmax, HitRecord& rec) const {
        Vec3f32 oc = center - r.origin();
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

private:
	Vec3f32 center;
	float radius;
    Material material;
};