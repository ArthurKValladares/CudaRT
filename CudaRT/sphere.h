#pragma once

#include "vec3.h"
#include "ray.h"
#include "hittable.h"

class Sphere {
public:
    __device__ Sphere(const vec3& center, float radius)
		: center(center)
		, radius(radius)
	{}

    __device__ bool hit(const ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const {
        vec3 oc = center - r.origin();
        auto a = r.direction().squared_length();
        auto h = dot(r.direction(), oc);
        auto c = oc.squared_length() - radius * radius;

        auto discriminant = h * h - a * c;
        if (discriminant < 0)
            return false;

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
        vec3 outward_normal = (rec.p - center) / radius;
        rec.set_face_normal(r, outward_normal);

        return true;
	}

private:
	vec3 center;
	float radius;
};