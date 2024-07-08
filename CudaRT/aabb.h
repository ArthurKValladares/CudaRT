#pragma once

#include "interval.h"
#include "vec3.h"

struct AABB {
    __device__ __host__ AABB() {}
    __device__ __host__ AABB(const Interval& X, const Interval& Y, const Interval& Z)
		: X(X), Y(Y), Z(Z) {}
    __device__ __host__ AABB(const Vec3f32& a, const Vec3f32& b) {
        X = (a[0] <= b[0]) ? Interval(a[0], b[0]) : Interval(b[0], a[0]);
        Y = (a[1] <= b[1]) ? Interval(a[1], b[1]) : Interval(b[1], a[1]);
        Z = (a[2] <= b[2]) ? Interval(a[2], b[2]) : Interval(b[2], a[2]);
    }
    __device__ __host__ AABB(const AABB& box0, const AABB& box1) {
        X = Interval(box0.X, box1.X);
        Y = Interval(box0.Y, box1.Y);
        Z = Interval(box0.Z, box1.Z);
    }

    __device__ const Interval& axis_interval(int n) const {
        if (n == 1) return Y;
        if (n == 2) return Z;
        return X;
    }

    __device__ bool hit(const Ray& r, Interval ray_t) const {
        const Vec3f32& ray_orig = r.origin();
        const Vec3f32& ray_dir = r.direction();

        for (int axis = 0; axis < 3; axis++) {
            const Interval& ax = axis_interval(axis);
            const double adinv = 1.0 / ray_dir[axis];

            auto t0 = (ax.min - ray_orig[axis]) * adinv;
            auto t1 = (ax.max - ray_orig[axis]) * adinv;

            if (t0 < t1) {
                if (t0 > ray_t.min) ray_t.min = t0;
                if (t1 < ray_t.max) ray_t.max = t1;
            }
            else {
                if (t1 > ray_t.min) ray_t.min = t1;
                if (t0 < ray_t.max) ray_t.max = t0;
            }

            if (ray_t.max <= ray_t.min)
                return false;
        }
        return true;
    }

	Interval X, Y, Z;
};