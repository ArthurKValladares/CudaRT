#pragma once

#include <math.h>
#include <stdlib.h>
#include <iostream>

#include "math.h"

class Vec3f32 {


public:
    __host__ __device__ Vec3f32() {}
    __host__ __device__ Vec3f32(float e0, float e1, float e2) { e[0] = e0; e[1] = e1; e[2] = e2; }

    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline Vec3f32 with_x(float v) const {
        Vec3f32 temp = *this; 
        temp[0] = v; 
        return temp;
    }
    __host__ __device__ inline Vec3f32 with_y(float v) const {
        Vec3f32 temp = *this;
        temp[1] = v; 
        return temp;
    }
    __host__ __device__ inline Vec3f32 with_z(float v) const {
        Vec3f32 temp = *this;
        temp[2] = v; 
        return temp;
    }
    __host__ __device__ inline Vec3f32 with_r(float v) const {
        return with_x(v);
    }
    __host__ __device__ inline Vec3f32 with_g(float v) const {
        return with_y(v);
    }
    __host__ __device__ inline Vec3f32 with_b(float v) const {
        return with_z(v);
    }

    __host__ __device__ inline const Vec3f32& operator+() const { return *this; }
    __host__ __device__ inline Vec3f32 operator-() const { return Vec3f32(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; };

    __host__ __device__ inline Vec3f32& operator+=(const Vec3f32& v2);
    __host__ __device__ inline Vec3f32& operator-=(const Vec3f32& v2);
    __host__ __device__ inline Vec3f32& operator*=(const Vec3f32& v2);
    __host__ __device__ inline Vec3f32& operator/=(const Vec3f32& v2);
    __host__ __device__ inline Vec3f32& operator*=(const float t);
    __host__ __device__ inline Vec3f32& operator/=(const float t);

    __host__ __device__ inline float length() const { return sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]); }
    __host__ __device__ inline float squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }
    __host__ __device__ inline void make_unit_vector();

    __host__ __device__ bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        auto s = 1e-8;
        return (fabs(e[0]) < s) && (fabs(e[1]) < s) && (fabs(e[2]) < s);
    }
 
    float e[3];
};



inline std::istream& operator>>(std::istream& is, Vec3f32& t) {
    is >> t.e[0] >> t.e[1] >> t.e[2];
    return is;
}

inline std::ostream& operator<<(std::ostream& os, const Vec3f32& t) {
    os << t.e[0] << " " << t.e[1] << " " << t.e[2];
    return os;
}

__host__ __device__ inline void Vec3f32::make_unit_vector() {
    float k = 1.0 / sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ inline Vec3f32 operator+(const Vec3f32& v1, const Vec3f32& v2) {
    return Vec3f32(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
}

__host__ __device__ inline Vec3f32 operator-(const Vec3f32& v1, const Vec3f32& v2) {
    return Vec3f32(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
}

__host__ __device__ inline Vec3f32 operator*(const Vec3f32& v1, const Vec3f32& v2) {
    return Vec3f32(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
}

__host__ __device__ inline Vec3f32 operator/(const Vec3f32& v1, const Vec3f32& v2) {
    return Vec3f32(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
}

__host__ __device__ inline Vec3f32 operator*(float t, const Vec3f32& v) {
    return Vec3f32(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline Vec3f32 operator/(Vec3f32 v, float t) {
    return Vec3f32(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}

__host__ __device__ inline Vec3f32 operator*(const Vec3f32& v, float t) {
    return Vec3f32(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline float dot(const Vec3f32& v1, const Vec3f32& v2) {
    return v1.e[0] * v2.e[0] + v1.e[1] * v2.e[1] + v1.e[2] * v2.e[2];
}

__host__ __device__ inline Vec3f32 cross(const Vec3f32& v1, const Vec3f32& v2) {
    return Vec3f32((v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1]),
        (-(v1.e[0] * v2.e[2] - v1.e[2] * v2.e[0])),
        (v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]));
}


__host__ __device__ inline Vec3f32& Vec3f32::operator+=(const Vec3f32& v) {
    e[0] += v.e[0];
    e[1] += v.e[1];
    e[2] += v.e[2];
    return *this;
}

__host__ __device__ inline Vec3f32& Vec3f32::operator*=(const Vec3f32& v) {
    e[0] *= v.e[0];
    e[1] *= v.e[1];
    e[2] *= v.e[2];
    return *this;
}

__host__ __device__ inline Vec3f32& Vec3f32::operator/=(const Vec3f32& v) {
    e[0] /= v.e[0];
    e[1] /= v.e[1];
    e[2] /= v.e[2];
    return *this;
}

__host__ __device__ inline Vec3f32& Vec3f32::operator-=(const Vec3f32& v) {
    e[0] -= v.e[0];
    e[1] -= v.e[1];
    e[2] -= v.e[2];
    return *this;
}

__host__ __device__ inline Vec3f32& Vec3f32::operator*=(const float t) {
    e[0] *= t;
    e[1] *= t;
    e[2] *= t;
    return *this;
}

__host__ __device__ inline Vec3f32& Vec3f32::operator/=(const float t) {
    float k = 1.0 / t;

    e[0] *= k;
    e[1] *= k;
    e[2] *= k;
    return *this;
}

__host__ __device__ inline Vec3f32 unit_vector(Vec3f32 v) {
    return v / v.length();
}

// TODO: Make these host as well, need a host random_float function
__device__ Vec3f32 random_vec(LocalRandomState& local_rand_state) {
    return Vec3f32(random_float(local_rand_state), random_float(local_rand_state), random_float(local_rand_state));
}

__device__ Vec3f32 random_vec(LocalRandomState& local_rand_state, double min, double max) {
    return Vec3f32(random_float(local_rand_state, min, max), random_float(local_rand_state, min, max), random_float(local_rand_state, min, max));
}

__device__ Vec3f32 random_in_unit_sphere(LocalRandomState& local_rand_state) {
    Vec3f32 p;
    do {
        p = 2.0f * random_vec(local_rand_state) - Vec3f32(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

__device__ Vec3f32 random_unit_vector(LocalRandomState& local_rand_state) {
    return unit_vector(random_in_unit_sphere(local_rand_state));
}

__device__ Vec3f32 random_on_hemisphere(LocalRandomState& local_rand_state, const Vec3f32& normal) {
    Vec3f32 on_unit_sphere = random_unit_vector(local_rand_state);
    // In the same hemisphere as the normal
    if (dot(on_unit_sphere, normal) > 0.0) {
        return on_unit_sphere;
    }
    else {
        return -on_unit_sphere;
    }
}

__device__ Vec3f32 random_in_unit_disk(LocalRandomState& local_rand_state) {
    Vec3f32 p;
    while (true) {
        p = Vec3f32(random_float(local_rand_state, -1, 1), random_float(local_rand_state, -1, 1), 0);
        if (p.squared_length() < 1) {
            return p;
        }
    }
}

__device__ Vec3f32 reflect(const Vec3f32& v, const Vec3f32& n) {
    return v - 2 * dot(v, n) * n;
}

__device__ Vec3f32 refract(const Vec3f32& uv, const Vec3f32& n, float etai_over_etat) {
    auto cos_theta = fminf(dot(-uv, n), 1.0);
    Vec3f32 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    Vec3f32 r_out_parallel = -sqrt(fabs(1.0 - r_out_perp.squared_length())) * n;
    return r_out_perp + r_out_parallel;
}