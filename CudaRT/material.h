#pragma once

#include "ray.h"
#include "hittable.h"

enum class MaterialType {
	Lambertian,
    Metal,
    Dieletric,
};

struct LambertianData {
    vec3 albedo;
};

struct MetalData{
    vec3 albedo;
    float fuzz;
};

struct DieletricData {
    float refraction_index;
};

union MaterialPayload {
    LambertianData lambertian;
    MetalData metal;
    DieletricData dieletric;
};

struct MaterialData {
    MaterialType type;
    MaterialPayload payload;
};

struct Material {

    Material() = delete;

    __device__ Material(MaterialData data) : data(data) {}

    __device__ static Material lambertian(vec3 albedo) {
        MaterialPayload payload = {};
        payload.lambertian.albedo = albedo;
        return Material {
            MaterialData {
                MaterialType::Lambertian,
                payload
            }
        };
    }

    __device__ static Material metal(vec3 albedo, float fuzz) {
        MaterialPayload payload = {};
        payload.metal.albedo = albedo;
        payload.metal.fuzz = fuzz;
        return Material{
            MaterialData {
                MaterialType::Metal,
                payload
            }
        };
    }

    __device__ static Material dieletric(float refraction_index) {
        MaterialPayload payload = {};
        payload.dieletric.refraction_index = refraction_index;
        return Material{
            MaterialData {
                MaterialType::Dieletric,
                payload
            }
        };
    }

    __device__ bool scatter(
        const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState* rand_state
    ) const {
        switch (data.type) {
        case MaterialType::Lambertian: {
            auto scatter_direction = rec.normal + random_unit_vector(rand_state);

            // Catch degenerate scatter direction
            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;

            scattered = ray(rec.p, scatter_direction);
            attenuation = data.payload.lambertian.albedo;
            return true;
        }
        case MaterialType::Metal: {
            vec3 reflected = reflect(r_in.direction(), rec.normal);
            reflected = unit_vector(reflected) + (data.payload.metal.fuzz * random_unit_vector(rand_state));
            scattered = ray(rec.p, reflected);
            attenuation = data.payload.metal.albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }
        case MaterialType::Dieletric: {
            attenuation = vec3(1.0, 1.0, 1.0);
            double ri = rec.front_face ? (1.0 / data.payload.dieletric.refraction_index) : data.payload.dieletric.refraction_index;

            vec3 unit_direction = unit_vector(r_in.direction());
            double cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            bool cannot_refract = ri * sin_theta > 1.0;
            vec3 direction;

            if (cannot_refract || reflectance(cos_theta, ri) > random_float(rand_state)) {
                direction = reflect(unit_direction, rec.normal);
            }
            else {
                direction = refract(unit_direction, rec.normal, ri);
            }

            scattered = ray(rec.p, direction);
            return true;
        }
        }
    }

private:
    MaterialData data;

    __device__ static float reflectance(float cosine, float refraction_index) {
        // Use Schlick's approximation for reflectance.
        auto r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};