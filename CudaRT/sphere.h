#pragma once

#include "vec3.h"
#include "ray.h"
#include "hittable.h"
#include "material.h"
#include "interval.h"

class Sphere {
public:
    __device__ __host__ Sphere()
    {
    }

    __device__ __host__ Sphere(const Vec3f32& center, float radius, Material material)
		: center(center)
		, radius(radius)
        , material(material)
        , is_moving(false)
	{
    }

    __device__ __host__ Sphere(const Vec3f32& center1, const Vec3f32& center2, float radius, Material material)
        : center(center1)
        , radius(radius)
        , material(material)
        , is_moving(true)
    {
        center_vec = center2 - center1;
    }

    __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        const float ray_tmin = ray_t.min;
        const float ray_tmax = ray_t.max;
        Vec3f32 oc = (is_moving ? sphere_center(r.time()) : center) - r.origin();
        const float a = r.direction().squared_length();
        const float h = dot(r.direction(), oc);
        const float c = oc.squared_length() - radius * radius;

        const float discriminant = h * h - a * c;
        if (discriminant < 0) {
            return false;
        }
        const float sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        float root = (h - sqrtd) / a;
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
        get_sphere_uv(outward_normal, rec.u, rec.v);
        rec.material = &material;

        return true;
	}

private:
    __device__ Vec3f32 sphere_center(float time) const {
        return center + center_vec * time;
    }

    __device__ static void get_sphere_uv(const Vec3f32& p, float& u, float& v) {
        // p: a given point on the sphere of radius one, centered at the origin.
        // u: returned value [0,1] of angle around the Y axis from X=-1.
        // v: returned value [0,1] of angle from Y=-1 to Y=+1.
        //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
        //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
        //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

        const float theta = acos(-p.y());
        const float phi = atan2(-p.z(), p.x()) + M_PI;

        u = phi / (2 * M_PI);
        v = theta / M_PI;
    }

	Vec3f32 center;
	float radius;
    Material material;
    bool is_moving;
    Vec3f32 center_vec;
};