#pragma once

#include "vec3.h"
#include "ray.h"

class Camera {
public:
    __device__ Camera(Vec3f32 lookfrom, Vec3f32 lookat, Vec3f32 vup, float vfov, float aspect, float aperture, float focus_dist) { // vfov is top to bottom in degrees
        lens_radius = aperture / 2.0f;
        const float theta = vfov * ((float)M_PI) / 180.0f;
        const float half_height = tan(theta / 2.0f);
        const float half_width = aspect * half_height;
        origin = lookfrom;
        front = unit_vector(lookfrom - lookat);
        right = unit_vector(cross(vup, front));
        up = cross(front, right);
        lower_left_corner = origin - half_width * focus_dist * right - half_height * focus_dist * up - focus_dist * front;
        horizontal = 2.0f * half_width * focus_dist * right;
        vertical = 2.0f * half_height * focus_dist * up;
    }

    __device__ Vec3f32 front_movement_vector() const {
        return unit_vector(front.with_y(0.));
    }

    __device__ Vec3f32 rioght_movement_vector() const {
        return unit_vector(right.with_y(0.));
    }

    __device__ void update_position(Vec3f32 disp) {
        origin += disp;
    }

    __device__ Ray get_ray(float s, float t, curandState* rand_state) {
        const Vec3f32 rd = lens_radius * random_in_unit_disk(rand_state);
        const Vec3f32 offset = right * rd.x() + up * rd.y();
        const float ray_time = random_float(rand_state);
        return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset, ray_time);
    }

    Vec3f32 origin;
    Vec3f32 lower_left_corner;
    Vec3f32 horizontal;
    Vec3f32 vertical;
    Vec3f32 right, up, front;
    float lens_radius;
};