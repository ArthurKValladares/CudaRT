#pragma once

#include "vec3.h"

class Ray
{
public:
    __device__ Ray() {}
    __device__ Ray(const Vec3f32& a, const Vec3f32& b) { A = a; B = b; }
    __device__ Vec3f32 origin() const { return A; }
    __device__ Vec3f32 direction() const { return B; }
    __device__ Vec3f32 at(float t) const { return A + t * B; }

    Vec3f32 A;
    Vec3f32 B;
};