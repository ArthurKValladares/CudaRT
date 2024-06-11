#include <iostream>
#include <time.h>
#include <fstream>

#include <curand_kernel.h>

#include "defs.h"
#include "vec3.h"
#include "ray.h"
#include "math.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"

#define SDL_MAIN_HANDLED
#include "SDL.h"

#define MAX_BOUNCE_DEPTH 25
#define SAMPLES_PER_PIXEL 25

__device__ vec3 color(curandState& local_rand_state, hittable_list** hittables, const ray& r) {
    ray cur_ray = r;
    vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
    for (int i = 0; i < 50; i++) {
        hit_record rec;
        if ((*hittables)->hit(cur_ray, 0.001f, FLT_MAX, rec)) {
            ray scattered;
            vec3 attenuation;
            if (rec.material->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = lerp(vec3(1.0, 1.0, 1.0), vec3(0.5, 0.7, 1.0), t);
            return cur_attenuation * c;
        }
    }
    return vec3(0.0, 0.0, 0.0); // exceeded recursion
}

__device__ double linear_to_gamma(double linear_component)
{
    if (linear_component > 0)
        return sqrt(linear_component);

    return 0;
}

__device__ vec3 linear_to_gamma(vec3 linear_vec)
{
   return vec3(linear_to_gamma(linear_vec.x()), linear_to_gamma(linear_vec.y()), linear_to_gamma(linear_vec.z()));
}

__device__ Uint32 vec3_to_color(vec3 color) {
    color = linear_to_gamma(color);

    color *= 255.99;

    const Uint8 R = color.r();
    const Uint8 G = color.g();
    const Uint8 B = color.b();

    return (0xFF << 24) | (R << 16) | (G << 8) | (B);
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;
    const int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int flipped_j = max_y - 1 - j;
    int pixel_index = flipped_j * max_x + i;
    const unsigned long long seed = 1984;
    const unsigned long long offset = 0;
    curand_init(seed, pixel_index, offset, &rand_state[pixel_index]);
}

__global__ void render(hittable_list** hittables, curandState* rand_state, int ns, Uint32* fb, int max_x, int max_y, camera** cam) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if ((i >= max_x) || (j >= max_y)) return;
    int flipped_j = max_y - 1 - j;
    int pixel_index = flipped_j * max_x + i;
    curandState& local_rand_state = rand_state[pixel_index];

    vec3 col(0., 0., 0.);
    for (int s = 0; s < ns; s++) {
        float u = float(i + random_float(local_rand_state)) / float(max_x);
        float v = float(j + random_float(local_rand_state)) / float(max_y);
        ray r = (*cam)->get_ray(u, v);
        col += color(local_rand_state, hittables, r);
    }

    fb[pixel_index] = vec3_to_color(col / float(ns));
}

__global__ void create_world(Sphere** spheres, int num_hittables, hittable_list** hittables, camera** d_camera, int nx, int ny) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *(spheres) = new Sphere(vec3(0, 0, -1), 0.5, Material::lambertian(vec3(0.1, 0.2, 0.5)));
        *(spheres + 1) = new Sphere(vec3(0, -100.5, -1), 100, Material::lambertian((vec3(0.8, 0.8, 0.0))));
        *(spheres + 2) = new Sphere(vec3(1, 0, -1), 0.5, Material::metal(vec3(0.8, 0.6, 0.2), 0.0));
        *(spheres + 3) = new Sphere(vec3(-1, 0, -1), 0.5, Material::dieletric(1.5));
        *(spheres + 4) = new Sphere(vec3(-1, 0, -1), -0.45, Material::dieletric(1.5));
        *hittables = new hittable_list(spheres, num_hittables);
        *d_camera = new camera(
            vec3(-2, 2, 1),
            vec3(0, 0, -1),
            vec3(0, 1, 0),
            20.0,
            float(nx) / float(ny)
        );
    }
}

__global__ void free_world(Sphere** spheres, hittable_list** hittables, camera** d_camera) {
    delete* (spheres);
    delete* (spheres+1);
    delete* hittables;
    delete* d_camera;
}

int main() {
    clock_t start, stop;

    int nx = 2400 / 2;
    int ny = 1200 / 2;

    int num_pixels = nx * ny;
    size_t fb_size = num_pixels * sizeof(vec3);

    const size_t surface_buffer_size = nx * ny * sizeof(Uint32);

    // Block stuff
    int tx = 8;
    int ty = 8;
    dim3 blocks(nx / tx + 1, ny / ty + 1);
    dim3 threads(tx, ty);

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        printf("SDL could not be initialized!\n"
            "SDL_Error: %s\n",
            SDL_GetError());
        return 0;
    }
    SDL_Window* window = SDL_CreateWindow(
        "Basic C SDL project", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        nx, ny, SDL_WINDOW_SHOWN);
    if (!window) {
        printf("Window could not be created!\n"
            "SDL_Error: %s\n",
            SDL_GetError());
        return 0;
    }
    SDL_Surface* surface = SDL_CreateRGBSurface(0, nx, ny, 32,
        0x00FF0000,
        0x0000FF00,
        0x000000FF,
        0xFF000000
    );
    if (!surface) {
        printf("Surface could not be created!\n"
            "SDL_Error: %s\n",
            SDL_GetError());
        return 0;
    }
    SDL_SetSurfaceBlendMode(surface, SDL_BLENDMODE_NONE);

    Uint32* surface_buffer;
    checkCudaErrors(cudaMalloc(&surface_buffer, surface_buffer_size));

    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void**)&d_rand_state, num_pixels * sizeof(curandState)));

    render_init << <blocks, threads >> > (nx, ny, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // TODO: Better create/free_world functions
    const int sphere_count = 5;
    Sphere** spheres;
    checkCudaErrors(cudaMalloc(&spheres, sphere_count * sizeof(Sphere*)));
    hittable_list** hittables;
    checkCudaErrors(cudaMalloc(&hittables, sizeof(hittable_list*)));
    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

    create_world << <1, 1 >> > (spheres, sphere_count, hittables, d_camera, nx, ny);

    bool quit = false;
    while (!quit) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            switch (e.type) {
                case SDL_KEYDOWN: {
                    switch (e.key.keysym.sym) {
                        case SDLK_ESCAPE: {
                            quit = true;
                            break;
                        }
                        default: {
                            break;
                        }
                    }
                    break;
                }
                case SDL_QUIT: {
                    quit = true;
                    break;
                }
                default: {
                    break;
                }
            }
        }

        SDL_LockSurface(surface);
        {
            start = clock();

            render << <blocks, threads >> > (
                hittables,
                d_rand_state, SAMPLES_PER_PIXEL,
                surface_buffer, nx, ny,
                d_camera
            );
            checkCudaErrors(cudaGetLastError());
            checkCudaErrors(cudaDeviceSynchronize());
            checkCudaErrors(cudaMemcpy(surface->pixels, surface_buffer, surface_buffer_size, cudaMemcpyDeviceToHost));

            stop = clock();
            double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
            std::cerr << "took " << timer_seconds << " seconds.\n";
        }
        SDL_UnlockSurface(surface);

        if (SDL_BlitScaled(surface, nullptr, SDL_GetWindowSurface(window), nullptr))
        {
            printf("SDL_BlitScaled %s", SDL_GetError());
            exit(1);
        }

        SDL_UpdateWindowSurface(window);
    }

    // Cleanup
    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (spheres, hittables, d_camera);
    checkCudaErrors(cudaFree(surface_buffer));
    checkCudaErrors(cudaFree(spheres));

    SDL_DestroyWindow(window);
    SDL_Quit();

    return 0;
}