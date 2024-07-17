#pragma once

#include "math.h"
#include "vec3.h"

struct Perlin {
    __device__ __host__ Perlin() {}

    __device__ void init(LocalRandomState& local_rand_state) {
        for (int i = 0; i < point_count; i++) {
            randvec[i] = unit_vector(random_vec(local_rand_state));
        }

        perlin_generate_perm(local_rand_state, perm_x);
        perlin_generate_perm(local_rand_state, perm_y);
        perlin_generate_perm(local_rand_state, perm_z);
    }

    __device__ float noise(const Vec3f32& p) const {
        const float u = p.x() - floor(p.x());
        const float v = p.y() - floor(p.y());
        const float w = p.z() - floor(p.z());

        const int i = int(floor(p.x()));
        const int j = int(floor(p.y()));
        const int k = int(floor(p.z()));

        Vec3f32 c[2][2][2];

        for (int di = 0; di < 2; di++) {
            for (int dj = 0; dj < 2; dj++) {
                for (int dk = 0; dk < 2; dk++) {
                    c[di][dj][dk] = randvec[
                        perm_x[(i + di) & 255] ^
                        perm_y[(j + dj) & 255] ^
                        perm_z[(k + dk) & 255]
                    ];
                }
            }
        }

        return perlin_interp(c, u, v, w);
    }

    __device__ float turb(const Vec3f32& p, int depth) const {
        auto accum = 0.0;
        auto temp_p = p;
        auto weight = 1.0;

        for (int i = 0; i < depth; i++) {
            accum += weight * noise(temp_p);
            weight *= 0.5;
            temp_p *= 2;
        }

        return fabs(accum);
    }

    static const int point_count = 256;

    Vec3f32* randvec;
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

    __device__ static float perlin_interp(const Vec3f32 c[2][2][2], float u, float v, float w) {
        const float uu = u * u * (3 - 2 * u);
        const float vv = v * v * (3 - 2 * v);
        const float ww = w * w * (3 - 2 * w);
        auto accum = 0.0;

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    Vec3f32 weight_v(u - i, v - j, w - k);
                    accum += (i * uu + (1 - i) * (1 - uu))
                        * (j * vv + (1 - j) * (1 - vv))
                        * (k * ww + (1 - k) * (1 - ww))
                        * dot(c[i][j][k], weight_v);
                }
            }
        }

        return accum;
    }
};