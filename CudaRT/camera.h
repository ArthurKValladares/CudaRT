#pragma once

#include "vec3.h"
#include "ray.h"

class camera {
public:
    __device__ camera(vec3 origin, vec3 look_at, vec3 vup, float vfov, float aspect_ratio) {
        const float theta = vfov * CUDART_PI_F / 180;
        const float half_height = tan(theta / 2);
        const float half_width = aspect_ratio * half_height;
        const vec3 w = unit_vector(origin - look_at);
        const vec3 u = unit_vector(cross(vup, w));
        const vec3 v = cross(w, u);

        m_origin = origin;
        m_lower_left_corner = origin - half_width * u - half_height * v - w;
        m_horizontal = 2 * half_width * u;
        m_vertical = 2 * half_height * v;
    }

    __device__ ray get_ray(float u, float v) {
        return ray(m_origin, m_lower_left_corner + u * m_horizontal + v * m_vertical - m_origin);
    }

    vec3 m_origin;
    vec3 m_lower_left_corner;
    vec3 m_horizontal;
    vec3 m_vertical;
};