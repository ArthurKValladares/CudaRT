#pragma once

#include "math.h"
#include "vec3.h"

struct Perlin {
    __device__ __host__ Perlin() {}

    __device__ void init(LocalRandomState& local_rand_state) {
        for (int i = 0; i < point_count; i++) {
            randfloat[i] = random_float(local_rand_state);
        }

        perlin_generate_perm(local_rand_state, perm_x);
        perlin_generate_perm(local_rand_state, perm_y);
        perlin_generate_perm(local_rand_state, perm_z);
    }

    __device__ float noise(const Vec3f32& p) const {
        float u = p.x() - floor(p.x());
        float v = p.y() - floor(p.y());
        float w = p.z() - floor(p.z());
        u = u * u * (3 - 2 * u);
        v = v * v * (3 - 2 * v);
        w = w * w * (3 - 2 * w);

        const int i = int(floor(p.x()));
        const int j = int(floor(p.y()));
        const int k = int(floor(p.z()));

        float c[2][2][2];

        for (int di = 0; di < 2; di++) {
            for (int dj = 0; dj < 2; dj++) {
                for (int dk = 0; dk < 2; dk++) {
                    c[di][dj][dk] = randfloat[
                        perm_x[(i + di) & 255] ^
                        perm_y[(j + dj) & 255] ^
                        perm_z[(k + dk) & 255]
                    ];
                }
            }
        }

        return trilinear_interp(c, u, v, w);
    }

    static const int point_count = 256;

    float* randfloat;
    int* perm_x;
    int* perm_y;
    int* perm_z;

private:
    __device__ void perlin_generate_perm(LocalRandomState& local_rand_state, int* p) {

        for (int i = 0; i < point_count; i++) {
            p[i] = i;
        }

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

    __device__ static float trilinear_interp(float c[2][2][2], float u, double v, float w) {
        auto accum = 0.0;
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    accum += (i * u + (1 - i) * (1 - u))
                        * (j * v + (1 - j) * (1 - v))
                        * (k * w + (1 - k) * (1 - w))
                        * c[i][j][k];
                }
            }
        }

        return accum;
    }
};