#pragma once

#include "ray.h"
#include  "hittable.h"

enum class MaterialType {
	Lambertian,
};

struct LambertianData {
    vec3 albedo;
};

struct MaterialData {
    MaterialType type;
    union {
        LambertianData lambertian;
    };
};

class Material {

    Material() = delete;

    __device__ Material(MaterialData data)
        : data(data) 
    {}

    __device__ bool scatter(
        const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState& rand_state
    ) const {
        switch (data.type) {
        case MaterialType::Lambertian: {
            auto scatter_direction = rec.normal + random_unit_vector(rand_state);
            scattered = ray(rec.p, scatter_direction);
            attenuation = data.lambertian.albedo;
            return true;
        }
        }
    }

private:
    MaterialData data;
};