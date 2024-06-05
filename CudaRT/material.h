#pragma once

#include "ray.h"
#include "hittable.h"

enum class MaterialType {
	Lambertian,
    Metal,
};

struct LambertianData {
    vec3 albedo;
};

struct MetalData{
    vec3 albedo;
    float fuzz;
};

union MaterialPayload {
    LambertianData lambertian;
    MetalData metal;
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

    __device__ bool scatter(
        const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState& rand_state
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
        }
    }

private:
    MaterialData data;
};