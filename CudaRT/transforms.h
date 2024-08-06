#pragma once

#include "vec3.h"
#include "math.h"
#include "aabb.h"

struct Translation {
    __device__ Translation(Vec3f32 offset) {
        m_offset = offset;
    }

    __device__ Translation() : Translation(Vec3f32(0.0, 0.0, 0.0)) {
    }

    __device__ Vec3f32 offset() const {
        return m_offset;
    }

private:
	Vec3f32 m_offset;
};

struct Rotation {
	__device__ Rotation(float angle) {
        const float radians = degrees_to_radians(angle);

        m_sin_theta = std::sin(radians);
        m_cos_theta = std::cos(radians);
	}

    __device__ Rotation() : Rotation(0.0) {}

private:
	float m_sin_theta;
	float m_cos_theta;
};