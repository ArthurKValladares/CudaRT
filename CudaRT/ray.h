#pragma once

#include "vec3.h"

// NOTE: for now, we are rendering frame time "individually" where 0.0 = start, and 1.0 = end.

class Ray
{
public:
    __device__ Ray() {}
    __device__ Ray(const Vec3f32& a, const Vec3f32& b) : m_A(a), m_B(b), m_time(0.0) {}
    __device__ Ray(const Vec3f32& a, const Vec3f32& b, float time) : m_A(a), m_B(b), m_time(time) {}

    __device__ Vec3f32 origin() const { return m_A; }
    __device__ Vec3f32 direction() const { return m_B; }
    __device__ float time() const { return m_time; }

    __device__ Vec3f32 at(float t) const { return m_A + t * m_B; }

    Vec3f32 m_A;
    Vec3f32 m_B;
    float m_time;
};