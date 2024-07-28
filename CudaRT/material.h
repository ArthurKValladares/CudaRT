#pragma once

#include "ray.h"
#include "hittable.h"
#include "texture.h"
#include "light.h"

enum class MaterialType {
	Lambertian,
    Metal,
    Dieletric,
    DiffuseLight,
};

struct LambertianData {
    Texture texture;
};

struct MetalData{
    Vec3f32 albedo;
    float fuzz;
};

struct DieletricData {
    float refraction_index;
};

struct LightData {
    DiffuseLight light;
};

union MaterialPayload {
    LambertianData lambertian;
    MetalData metal;
    DieletricData dieletric;
    DiffuseLight light;

    __device__ MaterialPayload& operator=(const MaterialPayload& payload) {
        memcpy(this, &payload, sizeof(MaterialPayload));

        return *this;
    }
};

struct MaterialData {
    MaterialType type;
    MaterialPayload payload;

    __device__ MaterialData() :
        type(MaterialType::Lambertian),
        payload(MaterialPayload{ LambertianData{Texture::SolidColor(1.0, 0.0, 1.0)}})
    {
    }

    __device__ MaterialData(MaterialType type, MaterialPayload payload) :
        type(type),
        payload(payload)
    {}

    __device__ MaterialData& operator=(const MaterialData& data) {
        type = data.type;
        payload = data.payload;

        return *this;
    }
};

struct Material {

    __device__ Material()
    {
    }

    __device__ Material(MaterialData data) : data(data) {}

    __device__ static Material lambertian(Texture texture) {
        MaterialPayload payload = {};
        payload.lambertian.texture = texture;
        return Material {
            MaterialData (
                MaterialType::Lambertian,
                payload
            )
        };
    }

    __device__ static Material lambertian(float r, float g, float b) {
        MaterialPayload payload = {};
        payload.lambertian.texture = Texture::SolidColor(r, g, b);
        return Material{
            MaterialData(
                MaterialType::Lambertian,
                payload
            )
        };
    }

    __device__ static Material metal(Vec3f32 albedo, float fuzz) {
        MaterialPayload payload = {};
        payload.metal.albedo = albedo;
        payload.metal.fuzz = fuzz;
        return Material{
            MaterialData(
                MaterialType::Metal,
                payload
            )
        };
    }

    __device__ static Material dieletric(float refraction_index) {
        MaterialPayload payload = {};
        payload.dieletric.refraction_index = refraction_index;
        return Material{
            MaterialData (
                MaterialType::Dieletric,
                payload
            )
        };
    }

    __device__ static Material diffuse_light(Vec3f32 color) {
        MaterialPayload payload = {};
        payload.light = DiffuseLight(color);
        return Material{
            MaterialData(
                MaterialType::DiffuseLight,
                payload
            )
        };
    }

    __device__ Material& operator=(const Material& mat) {
        data = mat.data;

        return *this;
    }

    __device__ bool scatter(
        const Ray& r_in, const HitRecord& rec, Vec3f32& attenuation, Ray& scattered, LocalRandomState& local_rand_state
    ) const {
        switch (data.type) {
        case MaterialType::Lambertian: {
            Vec3f32 scatter_direction = rec.normal + random_unit_vector(local_rand_state);

            // Catch degenerate scatter direction
            if (scatter_direction.near_zero())
                scatter_direction = rec.normal;

            scattered = Ray(rec.p, scatter_direction, r_in.time());
            attenuation = data.payload.lambertian.texture.value(rec.u, rec.v, rec.p);
            return true;
        }
        case MaterialType::Metal: {
            Vec3f32 reflected = reflect(r_in.direction(), rec.normal);
            reflected = unit_vector(reflected) + (data.payload.metal.fuzz * random_unit_vector(local_rand_state));
            scattered = Ray(rec.p, reflected, r_in.time());
            attenuation = data.payload.metal.albedo;
            return (dot(scattered.direction(), rec.normal) > 0);
        }
        case MaterialType::Dieletric: {
            attenuation = Vec3f32(1.0, 1.0, 1.0);
            double ri = rec.front_face ? (1.0 / data.payload.dieletric.refraction_index) : data.payload.dieletric.refraction_index;

            Vec3f32 unit_direction = unit_vector(r_in.direction());
            double cos_theta = fminf(dot(-unit_direction, rec.normal), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta * cos_theta);

            bool cannot_refract = ri * sin_theta > 1.0;
            Vec3f32 direction;

            if (cannot_refract || reflectance(cos_theta, ri) > random_float(local_rand_state)) {
                direction = reflect(unit_direction, rec.normal);
            }
            else {
                direction = refract(unit_direction, rec.normal, ri);
            }

            scattered = Ray(rec.p, direction, r_in.time());
            return true;
        }
        case MaterialType::DiffuseLight: {
            return false;
        }
        }
    }

    __device__ Vec3f32 emitted(double u, double v, const Vec3f32& p) const {
        switch (data.type) {
        case MaterialType::Lambertian: {
            return Vec3f32(0.0, 0.0, 0.0);
        }
        case MaterialType::Metal: {
            return Vec3f32(0.0, 0.0, 0.0);
        }
        case MaterialType::Dieletric: {
            return Vec3f32(0.0, 0.0, 0.0);
        }
        case MaterialType::DiffuseLight: {
            return data.payload.light.emitted(u, v, p);
        }
        }
    }

private:
    MaterialData data;

    __device__ static float reflectance(float cosine, float refraction_index) {
        // Use Schlick's approximation for reflectance.
        float r0 = (1 - refraction_index) / (1 + refraction_index);
        r0 = r0 * r0;
        return r0 + (1 - r0) * pow((1 - cosine), 5);
    }
};