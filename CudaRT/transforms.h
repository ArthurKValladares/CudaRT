#pragma once

#include "vec3.h"
#include "math.h"
#include "aabb.h"

struct Translation {
    __device__ Translation(Vec3f32 offset, AABB bbox) {
        m_offset = offset;
        m_bbox = bbox;
    }

private:
	Vec3f32 m_offset;
    AABB m_bbox;
};

struct Rotation {
	__device__ Rotation(float angle, AABB bbox) {
        const float radians = degrees_to_radians(angle);

        m_sin_theta = std::sin(radians);
        m_cos_theta = std::cos(radians);

        Vec3f32 min(INFINITY, INFINITY, INFINITY);
        Vec3f32 max(-INFINITY, -INFINITY, -INFINITY);

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    const float x = i * bbox.x.max + (1 - i) * bbox.x.min;
                    const float y = j * bbox.y.max + (1 - j) * bbox.y.min;
                    const float z = k * bbox.z.max + (1 - k) * bbox.z.min;

                    const float newx = m_cos_theta * x + m_sin_theta * z;
                    const float newz = -m_sin_theta * x + m_cos_theta * z;

                    const Vec3f32 tester(newx, y, newz);

                    for (int c = 0; c < 3; c++) {
                        min[c] = std::fmin(min[c], tester[c]);
                        max[c] = std::fmax(max[c], tester[c]);
                    }
                }
            }
        }

        m_bbox = AABB(min, max);
	}

private:
	float m_sin_theta;
	float m_cos_theta;
    AABB m_bbox;
};