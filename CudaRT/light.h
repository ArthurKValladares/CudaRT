#pragma once

#include "texture.h"
#include "vec3.h"

struct DiffuseLight {
    __device__ DiffuseLight(Texture tex) 
        : tex(tex)
    {}

    __device__ DiffuseLight(const Vec3f32& emit) 
        : tex(Texture::SolidColor(emit))
    {}

    __device__ Vec3f32 emitted(double u, double v, const Vec3f32& p) const {
        return tex.value(u, v, p);
    }

private:
    Texture tex;
};