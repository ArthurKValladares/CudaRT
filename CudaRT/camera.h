#pragma once

#include "vec3.h"
#include "ray.h"

class Camera {
public:
    __device__ Camera(Vec3f32 lookfrom, Vec3f32 lookat, Vec3f32 vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
        lens_radius = aperture / 2.0f;
        float theta = vfov * ((float)M_PI) / 180.0f;
        float half_height = tan(theta / 2.0f);
        float half_width = aspect * half_height;
        origin = lookfrom;
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);
        lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
        horizontal = 2.0f * half_width * focus_dist * u;
        vertical = 2.0f * half_height * focus_dist * v;
    }

    __device__ void update_position(Vec3f32 disp) {
        origin += disp;
    }

    __device__ Ray get_ray(float s, float t, curandState* rand_state) {
        Vec3f32 rd = lens_radius * random_in_unit_disk(rand_state);
        Vec3f32 offset = u * rd.x() + v * rd.y();
        return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
    }

    Vec3f32 origin;
    Vec3f32 lower_left_corner;
    Vec3f32 horizontal;
    Vec3f32 vertical;
    Vec3f32 u, v, w;
    float lens_radius;
};