#pragma once

#include "vec3.h"
#include "aabb.h"
#include "material.h"

struct Quad {
    __device__ Quad(const Vec3f32& Q, const Vec3f32& u, const Vec3f32& v, Material mat)
        : Q(Q), u(u), v(v), mat(mat)
    {
        const Vec3f32 n = cross(u, v);
        normal = unit_vector(n);
        D = dot(normal, Q);
        w = n / dot(n, n);
    }

    __device__ bool hit(const Ray& r, Interval ray_t, HitRecord& rec) const {
        const float denom = dot(normal, r.direction());

        if (fabs(denom) < 1e-8) {
            return false;
        }

        const float t = (D - dot(normal, r.origin())) / denom;
        if (!ray_t.contains(t)) {
            return false;
        }

        const Vec3f32 intersection = r.at(t);
        const Vec3f32 planar_hitpt_vector = intersection - Q;
        const float alpha = dot(w, cross(planar_hitpt_vector, v));
        const float beta = dot(w, cross(u, planar_hitpt_vector));

        if (!is_interior(alpha, beta, rec)) {
            return false;
        }

        rec.t = t;
        rec.p = intersection;
        rec.material = &mat;
        rec.set_face_normal(r, normal);

        return true;
    }

    __device__ bool is_interior(float a, float b, HitRecord& rec) const {
        const Interval unit_interval = Interval(0, 1);

        if (!unit_interval.contains(a) || !unit_interval.contains(b)) {
            return false;
        }

        rec.u = a;
        rec.v = b;
        return true;
    }

private:
    Vec3f32 Q;
    Vec3f32 u, v;
    Vec3f32 w;
    // TODO: Needs to be a pointer
    Material mat;
    Vec3f32 normal;
    float D;
};