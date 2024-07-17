#pragma once

#include "math.h"
#include "vec3.h"

struct Perlin {
    __device__ Perlin() {}

    __device__ void init(LocalRandomState& local_rand_state) {
        for (int i = 0; i < point_count; i++) {
            randfloat[i] = random_float(local_rand_state);
        }

        perlin_generate_perm(local_rand_state, perm_x);
        perlin_generate_perm(local_rand_state, perm_y);
        perlin_generate_perm(local_rand_state, perm_z);
    }

    __device__ double noise(const Vec3f32& p) const {
        auto i = int(4 * p.x()) & 255;
        auto j = int(4 * p.y()) & 255;
        auto k = int(4 * p.z()) & 255;

        return randfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
    }

private:
    static const int point_count = 256;
    double* randfloat;
    int* perm_x;
    int* perm_y;
    int* perm_z;

    __device__ void perlin_generate_perm(LocalRandomState& local_rand_state, int* p) {
        auto p = new int[point_count];

        for (int i = 0; i < point_count; i++)
            p[i] = i;

        permute(local_rand_state, p, point_count);
    }

    __device__ static void permute(LocalRandomState& local_rand_state, int* p, int n) {
        for (int i = n - 1; i > 0; i--) {
            int target = random_int(local_rand_state, 0, i);
            int tmp = p[i];
            p[i] = p[target];
            p[target] = tmp;
        }
    }
};